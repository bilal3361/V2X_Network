from __future__ import annotations

import sys
import time
from typing import Any

import kafka_alert_engine as base_engine
from mqtt_alert_engine_protect_collision import (
    APPROACH_ZONE_RADIUS_M,
    CLEAR_ZONE_RADIUS_M,
    COLLISION_DIAGNOSIS_LOG_PATH as MQTT_COLLISION_DIAGNOSIS_LOG_PATH,
    COOLDOWN_S,
    CONFLICT_ENTRY_GUARD_DISTANCE_M,
    CONFLICT_ZONE_RADIUS_M,
    MAX_WAIT_S,
    PREPARE_SPEED_MPS,
    PREPARE_TO_WAIT_DISTANCE_M,
    PROTECTED_MIN_GAP_M,
    PROTECTED_SPEED_MODE,
    PROTECTED_TAU_S,
    SAFE_TIME_GAP_S,
    SPEED_COMMAND_REFRESH_S,
    STOP_ZONE_RADIUS_M,
    WAIT_AT_ENTRY_DISTANCE_M,
    WAIT_SPEED_MPS,
    build_sumo_command,
    configure_protected_vehicle_types,
    maintain_intersection_gate_controls,
    open_collision_diagnosis_log,
    open_protection_log,
    release_gate_vehicle,
    update_alert_history,
    write_collision_diagnosis_rows,
)
from v2x_task5_common import PROJECT_ROOT, TrajectoryPredictor, VehicleHistoryStore, build_pair_alerts, utc_now_iso


PROTECTION_LOG_PATH = PROJECT_ROOT / "data" / "scenario1_kafka_protection_log.csv"
COLLISION_DIAGNOSIS_LOG_PATH = PROJECT_ROOT / "data" / "scenario1_kafka_collision_diagnosis_log.csv"


def parse_args() -> Any:
    args = base_engine.parse_args()
    if args.client_id == "v2x-kafka-alert-engine":
        args.client_id = "v2x-scenario1-protected-kafka"
    return args


def publish_status(producer: Any, args: Any, payload: dict[str, Any]) -> None:
    status_payload = base_engine.add_kafka_comparison_fields(payload)
    base_engine.publish_json(
        producer,
        args.topic,
        status_payload,
        key=str(payload.get("event_type", "scenario1_protected_kafka_status")),
        timeout_s=args.send_timeout_s,
    )


def main() -> int:
    args = parse_args()

    if not args.sumo_config.exists():
        print(f"SUMO config not found: {args.sumo_config}", file=sys.stderr)
        return 1

    protection_handle, protection_writer = open_protection_log(PROTECTION_LOG_PATH)
    diagnosis_handle, diagnosis_writer = open_collision_diagnosis_log(COLLISION_DIAGNOSIS_LOG_PATH)
    sent_log_handle = None
    producer = None

    try:
        traci, KafkaProducer = base_engine.import_runtime_dependencies()
        predictor = TrajectoryPredictor(
            model_path=args.model,
            feature_scaler_path=args.feature_scaler,
            target_scaler_path=args.target_scaler,
            metadata_path=args.metadata,
        )
        producer = base_engine.make_kafka_producer(KafkaProducer, args)
    except Exception as exc:
        protection_handle.close()
        diagnosis_handle.close()
        print(str(exc), file=sys.stderr)
        return 1

    using_sumo_gui = base_engine.is_sumo_gui_binary(args.sumo_binary)
    history_store = VehicleHistoryStore(input_len=predictor.input_len)
    last_publish_times: dict[tuple[str, str, str], float] = {}
    active_episode_states: dict[tuple[str, str], dict[str, Any]] = {}
    alert_history: dict[tuple[str, str], dict[str, Any]] = {}
    gate_state: dict[str, Any] = {
        "current_priority_vehicle_id": "",
        "reservation_queue": [],
        "conflict_group_vehicle_ids": [],
        "conflict_zone_vehicle_ids": [],
        "protected_vehicle_ids": set(),
        "controlled": {},
        "release_cooldowns": {},
        "last_release_time_s": -999999.0,
        "last_logged_signature": None,
    }
    step = 0
    total_alerts = 0
    high_alerts_seen = 0
    protections_applied = 0
    releases = 0
    collision_diagnoses = 0

    try:
        sent_log_handle, sent_log_writer = base_engine.open_sent_alert_log()
        print("Starting Scenario1 protected Kafka engine.")
        print(f"SUMO config: {args.sumo_config}")
        print(f"Protection log: {PROTECTION_LOG_PATH}")
        print(f"Collision diagnosis log: {COLLISION_DIAGNOSIS_LOG_PATH}")
        print(
            "FCFS Dynamic Target Gate: Kafka publishes alerts while the same "
            "MQTT-tuned reservation logic controls Scenario1 vehicles."
        )

        base_engine.start_traci(traci, build_sumo_command(args))
        configure_protected_vehicle_types(traci)

        junction_ids = set(traci.junction.getIDList())
        if args.junction_id not in junction_ids:
            raise RuntimeError(f"Junction '{args.junction_id}' was not found in the SUMO network.")

        jx, jy = traci.junction.getPosition(args.junction_id)
        if using_sumo_gui:
            base_engine.refresh_sumo_gui_view(traci, float(jx), float(jy), args.gui_view_radius_m)

        sim_start_time = float(traci.simulation.getTime())
        wall_start_time = time.monotonic()

        publish_status(
            producer,
            args,
            {
                "event_type": "scenario1_protected_kafka_started",
                "generated_at_utc": utc_now_iso(),
                "junction_id": args.junction_id,
                "junction_x": float(jx),
                "junction_y": float(jy),
                "model": str(args.model),
                "metadata": str(args.metadata),
                "min_risk": args.min_risk,
                "vehicle_groups": args.vehicle_groups,
                "prepare_speed_mps": PREPARE_SPEED_MPS,
                "wait_speed_mps": WAIT_SPEED_MPS,
                "safe_time_gap_s": SAFE_TIME_GAP_S,
                "cooldown_s": COOLDOWN_S,
                "speed_command_refresh_s": SPEED_COMMAND_REFRESH_S,
                "approach_zone_radius_m": APPROACH_ZONE_RADIUS_M,
                "stop_zone_radius_m": STOP_ZONE_RADIUS_M,
                "conflict_zone_radius_m": CONFLICT_ZONE_RADIUS_M,
                "clear_zone_radius_m": CLEAR_ZONE_RADIUS_M,
                "prepare_to_wait_distance_m": PREPARE_TO_WAIT_DISTANCE_M,
                "wait_at_entry_distance_m": WAIT_AT_ENTRY_DISTANCE_M,
                "conflict_entry_guard_distance_m": CONFLICT_ENTRY_GUARD_DISTANCE_M,
                "protected_speed_mode": PROTECTED_SPEED_MODE,
                "protected_min_gap_m": PROTECTED_MIN_GAP_M,
                "protected_tau_s": PROTECTED_TAU_S,
                "max_wait_s": MAX_WAIT_S,
                "real_time": bool(args.real_time),
                "realtime_factor": args.realtime_factor,
                "collision_diagnosis_log": str(COLLISION_DIAGNOSIS_LOG_PATH),
                "mqtt_reference_collision_diagnosis_log": str(MQTT_COLLISION_DIAGNOSIS_LOG_PATH),
            },
        )

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = float(traci.simulation.getTime())

            if using_sumo_gui and step < max(args.gui_refresh_steps, 0):
                base_engine.refresh_sumo_gui_view(traci, float(jx), float(jy), args.gui_view_radius_m)

            active_vehicle_ids = set(map(str, traci.vehicle.getIDList()))
            history_store.prune_missing(active_vehicle_ids)

            current_states: dict[str, dict[str, Any]] = {}
            candidate_histories: dict[str, list[dict[str, Any]]] = {}
            predictions_by_vehicle: dict[str, list[dict[str, Any]]] = {}

            for vehicle_id in active_vehicle_ids:
                state = base_engine.collect_vehicle_state(traci, vehicle_id, sim_time, float(jx), float(jy))
                current_states[vehicle_id] = state
                history_store.add(vehicle_id, state)

                if not base_engine.vehicle_group_allowed(str(state["vehicle_group"]), args.vehicle_groups):
                    continue
                if float(state["distance_to_junction_center_m"]) > float(args.near_radius_m):
                    continue
                if not history_store.is_ready(vehicle_id):
                    continue

                candidate_histories[vehicle_id] = history_store.history(vehicle_id)

            should_predict = args.prediction_interval_steps <= 1 or step % args.prediction_interval_steps == 0
            if should_predict and candidate_histories:
                predictions_by_vehicle = predictor.predict_many(candidate_histories)

            if args.publish_predictions and predictions_by_vehicle:
                for vehicle_id, predictions in predictions_by_vehicle.items():
                    prediction_payload = base_engine.add_kafka_comparison_fields(
                        {
                            "event_type": "trajectory_prediction",
                            "simulation_time": round(sim_time, 4),
                            "vehicle_id": vehicle_id,
                            "predictions": predictions,
                        }
                    )
                    base_engine.publish_json(
                        producer,
                        args.prediction_topic,
                        prediction_payload,
                        key=vehicle_id,
                        timeout_s=args.send_timeout_s,
                    )

            all_alerts = build_pair_alerts(
                predictions_by_vehicle,
                simulation_time=sim_time,
                min_risk_level=args.min_risk,
            )
            protection_candidates = base_engine.sorted_limited_alerts(all_alerts, 0)
            alerts = base_engine.sorted_limited_alerts(all_alerts, args.max_alerts_per_cycle)

            high_alerts = [alert for alert in protection_candidates if str(alert.get("risk_level")) == "HIGH"]
            high_alerts_seen += len(high_alerts)

            update_alert_history(alert_history, protection_candidates, sim_time)
            collision_diagnoses += write_collision_diagnosis_rows(
                traci,
                diagnosis_writer,
                diagnosis_handle,
                sim_time,
                alert_history,
                gate_state,
                float(jx),
                float(jy),
            )

            new_protections, new_releases = maintain_intersection_gate_controls(
                traci,
                protection_writer,
                protection_handle,
                gate_state,
                active_vehicle_ids,
                current_states,
                sim_time,
                float(jx),
                float(jy),
                high_alerts,
            )
            protections_applied += new_protections
            releases += new_releases

            for alert in alerts:
                if args.alert_mode == "episode":
                    episode_update = base_engine.update_episode_state_for_alert(
                        active_episode_states,
                        alert,
                        args.episode_reset_s,
                    )
                    should_publish = episode_update is not None
                    if episode_update is not None:
                        alert.update(episode_update)
                elif args.alert_mode == "cooldown":
                    should_publish = base_engine.should_publish_alert(
                        last_publish_times,
                        alert,
                        args.alert_cooldown_s,
                    )
                    alert.setdefault("episode_status", "COOLDOWN_DETECTION")
                else:
                    should_publish = True
                    alert.setdefault("episode_status", "DETECTION")

                if not should_publish:
                    continue

                total_alerts += 1
                base_engine.add_kafka_comparison_fields(alert)
                base_engine.publish_json(
                    producer,
                    args.topic,
                    alert,
                    key=str(alert.get("pair_id", "")),
                    timeout_s=args.send_timeout_s,
                )
                base_engine.write_sent_alert_log(sent_log_writer, sent_log_handle, alert)

            if args.alert_mode == "episode":
                base_engine.prune_episode_states(active_episode_states, sim_time, args.episode_reset_s)

            if args.real_time:
                base_engine.pace_real_time(sim_time, sim_start_time, wall_start_time, args.realtime_factor)

            if args.status_interval_steps > 0 and step % args.status_interval_steps == 0:
                print(
                    f"Protected Kafka status | sim={sim_time:.1f} | step={step} | "
                    f"active={len(active_vehicle_ids)} | ready={len(predictions_by_vehicle)} | "
                    f"alerts={total_alerts} | high_seen={high_alerts_seen} | "
                    f"protections={protections_applied} | releases={releases} | "
                    f"collisions_logged={collision_diagnoses}"
                )

            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break

        publish_status(
            producer,
            args,
            {
                "event_type": "scenario1_protected_kafka_stopped",
                "generated_at_utc": utc_now_iso(),
                "steps": step,
                "alerts_published": total_alerts,
                "high_alerts_seen": high_alerts_seen,
                "protections_applied": protections_applied,
                "releases": releases,
                "collision_diagnoses": collision_diagnoses,
                "priority_vehicle_id": gate_state.get("current_priority_vehicle_id", ""),
                "reservation_queue": gate_state.get("reservation_queue", []),
                "conflict_zone_vehicle_ids": gate_state.get("conflict_zone_vehicle_ids", []),
            },
        )

    except Exception as exc:
        try:
            if producer is not None:
                publish_status(
                    producer,
                    args,
                    {
                        "event_type": "scenario1_protected_kafka_error",
                        "generated_at_utc": utc_now_iso(),
                        "error": str(exc),
                    },
                )
        except Exception:
            pass
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        try:
            active_vehicle_ids = set(map(str, traci.vehicle.getIDList()))
            for vehicle_id in list(gate_state.get("controlled", {})):
                if vehicle_id in active_vehicle_ids:
                    try:
                        release_gate_vehicle(traci, gate_state, vehicle_id, float(traci.simulation.getTime()))
                    except Exception:
                        pass
            traci.close()
        except Exception:
            pass
        if producer is not None:
            try:
                producer.flush()
                producer.close()
            except Exception:
                pass
        if sent_log_handle is not None:
            sent_log_handle.close()
        protection_handle.close()
        diagnosis_handle.close()

    print(
        "Scenario1 protected Kafka engine finished. "
        f"Steps={step}, alerts_published={total_alerts}, high_alerts_seen={high_alerts_seen}, "
        f"protections_applied={protections_applied}, releases={releases}, "
        f"collision_diagnoses={collision_diagnoses}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
