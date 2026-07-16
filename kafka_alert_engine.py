from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from v2x_task5_common import (
    FEATURE_SCALER_PATH,
    INTERSECTION_ID,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    NEAR_JUNCTION_RADIUS_M,
    PROJECT_ROOT,
    RISK_PRIORITY,
    TARGET_SCALER_PATH,
    TrajectoryPredictor,
    VehicleHistoryStore,
    build_pair_alerts,
    euclidean_distance,
    infer_vehicle_group,
    start_traci,
    utc_now_iso,
)


DEFAULT_SUMO_CONFIG = PROJECT_ROOT / "osm.sumocfg"
DEFAULT_KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_ALERT_TOPIC = "v2x.alerts"
DEFAULT_PREDICTION_TOPIC = "v2x.predictions"
SENT_ALERT_LOG_PATH = PROJECT_ROOT / "data" / "kafka_sent_alert_log.csv"
SENT_ALERT_LOG_FIELDS = [
    "protocol",
    "alert_id",
    "simulation_time",
    "risk_level",
    "vehicle_1",
    "vehicle_2",
    "sent_perf_time",
    "sent_wall_time_utc",
    "sent_wall_time_cest",
]
LOCAL_TZ = ZoneInfo("Europe/Rome")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V2X prediction engine and publish alerts through Kafka."
    )
    parser.add_argument("--sumo-binary", default="sumo-gui", help="SUMO executable, e.g. sumo or sumo-gui.")
    parser.add_argument("--sumo-config", type=Path, default=DEFAULT_SUMO_CONFIG, help="SUMO config file.")
    parser.add_argument("--junction-id", default=INTERSECTION_ID, help="Target intersection/junction ID.")
    parser.add_argument(
        "--near-radius-m",
        type=float,
        default=NEAR_JUNCTION_RADIUS_M,
        help="Only vehicles inside this radius are considered for alerts.",
    )
    parser.add_argument(
        "--vehicle-groups",
        choices=["all", "targeted", "background"],
        default="targeted",
        help="Limit alert evaluation to all vehicles, targeted vehicles, or background vehicles.",
    )
    parser.add_argument("--kafka-bootstrap-servers", default=DEFAULT_KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=DEFAULT_ALERT_TOPIC, help="Kafka alert topic.")
    parser.add_argument("--prediction-topic", default=DEFAULT_PREDICTION_TOPIC, help="Kafka prediction topic.")
    parser.add_argument("--client-id", default="v2x-kafka-alert-engine", help="Kafka producer client ID.")
    parser.add_argument(
        "--min-risk",
        choices=["LOW", "HIGH"],
        default="LOW",
        help="Minimum risk level that should be published as an alert.",
    )
    parser.add_argument(
        "--risk-source",
        choices=["arrival"],
        default="arrival",
        help="Risk rule used for publishing alerts. This project uses arrival-time difference only.",
    )
    parser.add_argument(
        "--publish-predictions",
        action="store_true",
        help="Publish each vehicle trajectory prediction to the Kafka prediction topic.",
    )
    parser.add_argument(
        "--max-alerts-per-cycle",
        type=int,
        default=5,
        help="Publish at most N alerts per prediction cycle after severity sorting. Use 0 for no limit.",
    )
    parser.add_argument(
        "--prediction-interval-steps",
        type=int,
        default=10,
        help="Run model inference every N SUMO steps.",
    )
    parser.add_argument(
        "--alert-mode",
        choices=["episode", "cooldown", "all"],
        default="episode",
        help="episode publishes once per risk episode; cooldown repeats after --alert-cooldown-s; all publishes every detection.",
    )
    parser.add_argument("--alert-cooldown-s", type=float, default=1.0)
    parser.add_argument("--episode-reset-s", type=float, default=3600.0)
    parser.add_argument("--status-interval-steps", type=int, default=50)
    parser.add_argument("--gui-view-radius-m", type=float, default=260.0)
    parser.add_argument("--gui-refresh-steps", type=int, default=20)
    parser.add_argument(
        "--traci-num-clients",
        type=int,
        default=1,
        help="Number of TraCI clients SUMO should wait for. Use 2 for subscriber-controller mode.",
    )
    parser.add_argument(
        "--traci-client-order",
        type=int,
        default=1,
        help="TraCI execution order for this engine client.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--real-time", dest="real_time", action="store_true", default=True)
    parser.add_argument("--no-real-time", dest="real_time", action="store_false")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--send-timeout-s", type=float, default=10.0, help="Kafka delivery wait timeout.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Saved Keras trajectory model.")
    parser.add_argument("--feature-scaler", type=Path, default=FEATURE_SCALER_PATH)
    parser.add_argument("--target-scaler", type=Path, default=TARGET_SCALER_PATH)
    parser.add_argument("--metadata", type=Path, default=MODEL_METADATA_PATH)
    return parser.parse_args()


def import_runtime_dependencies() -> tuple[Any, Any]:
    try:
        import traci
    except Exception as exc:
        raise RuntimeError("TraCI is required. Install SUMO and ensure its Python tools are available.") from exc

    try:
        from kafka import KafkaProducer
    except Exception as exc:
        raise RuntimeError("kafka-python is required. Install it with: pip install kafka-python") from exc

    return traci, KafkaProducer


def make_kafka_producer(KafkaProducer: Any, args: argparse.Namespace) -> Any:
    return KafkaProducer(
        bootstrap_servers=args.kafka_bootstrap_servers,
        client_id=args.client_id,
        value_serializer=lambda payload: json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        key_serializer=lambda value: value.encode("utf-8") if value else None,
        linger_ms=0,
        acks="all",
    )


def publish_json(
    producer: Any,
    topic: str,
    payload: dict[str, Any],
    key: str | None = None,
    timeout_s: float = 10.0,
) -> None:
    future = producer.send(topic, key=key, value=payload)
    future.get(timeout=timeout_s)


def collect_vehicle_state(traci: Any, vehicle_id: str, sim_time: float, jx: float, jy: float) -> dict[str, Any]:
    x, y = traci.vehicle.getPosition(vehicle_id)
    speed = traci.vehicle.getSpeed(vehicle_id)
    acceleration = traci.vehicle.getAcceleration(vehicle_id)
    angle = traci.vehicle.getAngle(vehicle_id)
    lane_position = traci.vehicle.getLanePosition(vehicle_id)
    distance = euclidean_distance(x, y, jx, jy)

    return {
        "time": float(sim_time),
        "vehicle_id": str(vehicle_id),
        "vehicle_group": infer_vehicle_group(vehicle_id),
        "x": float(x),
        "y": float(y),
        "speed_mps": float(speed),
        "acceleration_mps2": float(acceleration),
        "angle_deg": float(angle),
        "lane_position_m": float(lane_position),
        "target_junction_x": float(jx),
        "target_junction_y": float(jy),
        "distance_to_junction_center_m": float(distance),
        "is_near_target_junction": 1 if distance <= NEAR_JUNCTION_RADIUS_M else 0,
    }


def vehicle_group_allowed(vehicle_group: str, selected_groups: str) -> bool:
    return selected_groups == "all" or vehicle_group == selected_groups


def alert_pair_key(alert: dict[str, Any]) -> tuple[str, str]:
    pair = sorted([str(alert["vehicle_1"]), str(alert["vehicle_2"])])
    return pair[0], pair[1]


def should_publish_alert(
    last_publish_times: dict[tuple[str, str, str], float],
    alert: dict[str, Any],
    cooldown_s: float,
) -> bool:
    pair = alert_pair_key(alert)
    key = (pair[0], pair[1], str(alert["risk_level"]))
    sim_time = float(alert["simulation_time"])
    last_time = last_publish_times.get(key)
    if last_time is not None and (sim_time - last_time) < cooldown_s:
        return False
    last_publish_times[key] = sim_time
    return True


def update_episode_state_for_alert(
    active_episode_states: dict[tuple[str, str], dict[str, Any]],
    alert: dict[str, Any],
    episode_reset_s: float,
) -> dict[str, str] | None:
    pair_key = alert_pair_key(alert)
    sim_time = float(alert["simulation_time"])
    risk_level = str(alert["risk_level"])
    current_priority = RISK_PRIORITY.get(risk_level, 0)
    previous = active_episode_states.get(pair_key)

    if previous is None or (sim_time - float(previous["last_seen_sim_time"])) > episode_reset_s:
        episode_id = f"{pair_key[0]}|{pair_key[1]}|{sim_time:.1f}"
        active_episode_states[pair_key] = {
            "episode_id": episode_id,
            "risk_level": risk_level,
            "max_risk_priority": current_priority,
            "last_seen_sim_time": sim_time,
        }
        return {"episode_id": episode_id, "episode_status": f"NEW_{risk_level}"}

    previous_max_priority = int(previous.get("max_risk_priority", RISK_PRIORITY.get(str(previous["risk_level"]), 0)))
    previous["risk_level"] = risk_level
    previous["max_risk_priority"] = max(previous_max_priority, current_priority)
    previous["last_seen_sim_time"] = sim_time
    if current_priority > previous_max_priority:
        return {
            "episode_id": str(previous.get("episode_id", f"{pair_key[0]}|{pair_key[1]}")),
            "episode_status": f"ESCALATED_{risk_level}",
        }
    return None


def prune_episode_states(
    active_episode_states: dict[tuple[str, str], dict[str, Any]],
    simulation_time: float,
    episode_reset_s: float,
) -> None:
    for pair_key, state in list(active_episode_states.items()):
        if (float(simulation_time) - float(state["last_seen_sim_time"])) > max(episode_reset_s, 0.0):
            del active_episode_states[pair_key]


def alert_sort_key(alert: dict[str, Any]) -> tuple[Any, ...]:
    risk_priority = RISK_PRIORITY.get(str(alert.get("risk_level")), 0)
    arrival_diff = alert.get("arrival_time_difference_s")
    arrival_value = float(arrival_diff) if arrival_diff is not None else 999999.0
    return (
        -risk_priority,
        arrival_value,
        str(alert.get("vehicle_1", "")),
        str(alert.get("vehicle_2", "")),
    )


def sorted_limited_alerts(alerts: list[dict[str, Any]], max_alerts_per_cycle: int) -> list[dict[str, Any]]:
    sorted_alerts = sorted(alerts, key=alert_sort_key)
    if max_alerts_per_cycle > 0:
        return sorted_alerts[:max_alerts_per_cycle]
    return sorted_alerts


def is_sumo_gui_binary(sumo_binary: str | Path) -> bool:
    return Path(str(sumo_binary)).name == "sumo-gui"


def refresh_sumo_gui_view(traci: Any, center_x: float, center_y: float, radius_m: float) -> None:
    radius = max(float(radius_m), 50.0)
    view_id = traci.gui.DEFAULT_VIEW
    try:
        traci.gui.setSchema(view_id, "standard")
        traci.gui.setZoom(view_id, 500.0)
        traci.gui.setOffset(view_id, float(center_x), float(center_y))
        traci.gui.setBoundary(
            view_id,
            float(center_x) - radius,
            float(center_y) - radius,
            float(center_x) + radius,
            float(center_y) + radius,
        )
    except Exception as exc:
        print(f"Warning: could not refresh SUMO-GUI view: {exc}", file=sys.stderr)


def pace_real_time(sim_time: float, sim_start_time: float, wall_start_time: float, realtime_factor: float) -> None:
    if realtime_factor <= 0:
        return
    target_wall_elapsed = (float(sim_time) - float(sim_start_time)) / realtime_factor
    actual_wall_elapsed = time.monotonic() - wall_start_time
    sleep_s = target_wall_elapsed - actual_wall_elapsed
    if sleep_s > 0:
        time.sleep(sleep_s)


def format_cest_time(value: Any) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone(LOCAL_TZ)
    return f"{local_dt:%H:%M:%S}.{local_dt.microsecond // 1000:03d} {local_dt.tzname()}"


def add_kafka_comparison_fields(payload: dict[str, Any]) -> dict[str, Any]:
    payload["protocol"] = "kafka"
    payload["sent_perf_time"] = time.perf_counter()
    payload["sent_wall_time_utc"] = utc_now_iso()
    payload["sent_wall_time_cest"] = format_cest_time(payload["sent_wall_time_utc"])
    return payload


def open_sent_alert_log(path: Path = SENT_ALERT_LOG_PATH) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=SENT_ALERT_LOG_FIELDS)
    writer.writeheader()
    handle.flush()
    return handle, writer


def write_sent_alert_log(writer: csv.DictWriter, handle: Any, alert: dict[str, Any]) -> None:
    writer.writerow(
        {
            "protocol": alert.get("protocol", "kafka"),
            "alert_id": alert.get("alert_id", ""),
            "simulation_time": alert.get("simulation_time", ""),
            "risk_level": alert.get("risk_level", ""),
            "vehicle_1": alert.get("vehicle_1", ""),
            "vehicle_2": alert.get("vehicle_2", ""),
            "sent_perf_time": alert.get("sent_perf_time", ""),
            "sent_wall_time_utc": alert.get("sent_wall_time_utc", ""),
            "sent_wall_time_cest": alert.get("sent_wall_time_cest", ""),
        }
    )
    handle.flush()


def main() -> int:
    args = parse_args()

    if not args.sumo_config.exists():
        print(f"SUMO config not found: {args.sumo_config}", file=sys.stderr)
        return 1

    try:
        traci, KafkaProducer = import_runtime_dependencies()
        predictor = TrajectoryPredictor(
            model_path=args.model,
            feature_scaler_path=args.feature_scaler,
            target_scaler_path=args.target_scaler,
            metadata_path=args.metadata,
        )
        producer = make_kafka_producer(KafkaProducer, args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    using_sumo_gui = is_sumo_gui_binary(args.sumo_binary)
    if using_sumo_gui:
        sumo_cmd = [
            args.sumo_binary,
            "-c",
            str(args.sumo_config),
            "--num-clients",
            str(args.traci_num_clients),
            "--start",
            "--disable-textures",
            "--window-size",
            "1360,820",
            "--window-pos",
            "30,30",
        ]
    else:
        sumo_cmd = [
            args.sumo_binary,
            "-c",
            str(args.sumo_config),
            "--num-clients",
            str(args.traci_num_clients),
            "--start",
            "--quit-on-end",
        ]

    history_store = VehicleHistoryStore(input_len=predictor.input_len)
    last_publish_times: dict[tuple[str, str, str], float] = {}
    active_episode_states: dict[tuple[str, str], dict[str, Any]] = {}
    step = 0
    total_alerts = 0
    sent_log_handle, sent_log_writer = open_sent_alert_log()

    try:
        print(f"Publishing Kafka alerts to {args.topic} on {args.kafka_bootstrap_servers}")
        start_traci(traci, sumo_cmd)
        traci.setOrder(int(args.traci_client_order))

        junction_ids = set(traci.junction.getIDList())
        if args.junction_id not in junction_ids:
            raise RuntimeError(f"Junction '{args.junction_id}' was not found in the SUMO network.")

        jx, jy = traci.junction.getPosition(args.junction_id)
        if using_sumo_gui:
            refresh_sumo_gui_view(traci, float(jx), float(jy), args.gui_view_radius_m)

        sim_start_time = float(traci.simulation.getTime())
        wall_start_time = time.monotonic()

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = float(traci.simulation.getTime())

            if using_sumo_gui and step < max(args.gui_refresh_steps, 0):
                refresh_sumo_gui_view(traci, float(jx), float(jy), args.gui_view_radius_m)

            active_vehicle_ids = set(map(str, traci.vehicle.getIDList()))
            history_store.prune_missing(active_vehicle_ids)

            candidate_histories: dict[str, list[dict[str, Any]]] = {}
            predictions_by_vehicle: dict[str, list[dict[str, Any]]] = {}

            for vehicle_id in active_vehicle_ids:
                state = collect_vehicle_state(traci, vehicle_id, sim_time, float(jx), float(jy))
                history_store.add(vehicle_id, state)

                if not vehicle_group_allowed(str(state["vehicle_group"]), args.vehicle_groups):
                    continue
                if state["distance_to_junction_center_m"] > args.near_radius_m:
                    continue
                if not history_store.is_ready(vehicle_id):
                    continue

                candidate_histories[vehicle_id] = history_store.history(vehicle_id)

            should_predict = (
                args.prediction_interval_steps <= 1
                or step % args.prediction_interval_steps == 0
            )

            if should_predict and candidate_histories:
                predictions_by_vehicle = predictor.predict_many(candidate_histories)

            if args.publish_predictions and predictions_by_vehicle:
                for vehicle_id, predictions in predictions_by_vehicle.items():
                    prediction_payload = add_kafka_comparison_fields(
                        {
                            "event_type": "trajectory_prediction",
                            "simulation_time": round(sim_time, 4),
                            "vehicle_id": vehicle_id,
                            "predictions": predictions,
                        }
                    )
                    publish_json(
                        producer,
                        args.prediction_topic,
                        prediction_payload,
                        key=vehicle_id,
                        timeout_s=args.send_timeout_s,
                    )

            alerts = build_pair_alerts(
                predictions_by_vehicle,
                simulation_time=sim_time,
                min_risk_level=args.min_risk,
            )
            alerts = sorted_limited_alerts(alerts, args.max_alerts_per_cycle)

            for alert in alerts:
                if args.alert_mode == "episode":
                    episode_update = update_episode_state_for_alert(
                        active_episode_states,
                        alert,
                        args.episode_reset_s,
                    )
                    should_publish = episode_update is not None
                    if episode_update is not None:
                        alert.update(episode_update)
                elif args.alert_mode == "cooldown":
                    should_publish = should_publish_alert(last_publish_times, alert, args.alert_cooldown_s)
                    alert.setdefault("episode_status", "COOLDOWN_DETECTION")
                else:
                    should_publish = True
                    alert.setdefault("episode_status", "DETECTION")

                if not should_publish:
                    continue

                add_kafka_comparison_fields(alert)
                publish_json(
                    producer,
                    args.topic,
                    alert,
                    key=str(alert.get("pair_id", "")),
                    timeout_s=args.send_timeout_s,
                )
                write_sent_alert_log(sent_log_writer, sent_log_handle, alert)
                total_alerts += 1
                print(
                    f"Kafka alert published | risk={alert['risk_level']} | "
                    f"sim={alert['simulation_time']} | pair={alert['vehicle_1']} -> {alert['vehicle_2']}"
                )

            if args.alert_mode == "episode":
                prune_episode_states(active_episode_states, sim_time, args.episode_reset_s)

            if args.real_time:
                pace_real_time(sim_time, sim_start_time, wall_start_time, args.realtime_factor)

            if args.status_interval_steps > 0 and step % args.status_interval_steps == 0:
                print(
                    f"Kafka engine status | sim={sim_time:.1f} | step={step} | "
                    f"active={len(active_vehicle_ids)} | ready={len(predictions_by_vehicle)} | "
                    f"alerts={total_alerts}"
                )

            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break

    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    finally:
        try:
            traci.close()
        except Exception:
            pass
        producer.flush()
        producer.close()
        sent_log_handle.close()

    print(f"Kafka alert engine finished. Steps={step}, alerts_published={total_alerts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
