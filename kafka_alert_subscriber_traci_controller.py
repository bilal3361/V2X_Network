from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import mqtt_alert_engine_protect_collision as protection
from mqtt_alert_subscriber import format_compact_alert
from mqtt_alert_subscriber_traci_controller import (
    collect_current_states,
    connect_traci_with_retry,
    expire_alert_visuals,
    apply_alert_visual,
    calculate_latency_ms,
    normalize_pair,
    queue_alert_visual,
)
from v2x_task5_common import DATA_DIR, INTERSECTION_ID, utc_now_iso


DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "v2x.alerts"
DEFAULT_TRACI_PORT = 8873
DEFAULT_PROTECTION_LOG_PATH = DATA_DIR / "scenario1_kafka_subscriber_controller_protection_log.csv"
DEFAULT_RECEIVED_LOG_PATH = DATA_DIR / "scenario1_kafka_subscriber_controller_received_alert_log.csv"

RECEIVED_LOG_FIELDS = [
    "received_at_utc",
    "protocol",
    "alert_id",
    "topic",
    "simulation_time",
    "risk_level",
    "episode_status",
    "vehicle_1",
    "vehicle_2",
    "arrival_time_difference_s",
    "latency_ms",
    "controller_action",
    "payload_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to Kafka alerts and control SUMO vehicles through TraCI."
    )
    parser.add_argument("--kafka-bootstrap-servers", default=DEFAULT_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--group-id", default="v2x-scenario1-kafka-subscriber-controller")
    parser.add_argument("--client-id", default="v2x-scenario1-kafka-subscriber-controller")
    parser.add_argument("--auto-offset-reset", choices=["latest", "earliest"], default="latest")
    parser.add_argument("--traci-host", default="localhost")
    parser.add_argument("--traci-port", type=int, default=DEFAULT_TRACI_PORT)
    parser.add_argument("--traci-client-order", type=int, default=2)
    parser.add_argument("--junction-id", default=INTERSECTION_ID)
    parser.add_argument("--alert-memory-s", type=float, default=12.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--protection-log", type=Path, default=DEFAULT_PROTECTION_LOG_PATH)
    parser.add_argument("--received-log", type=Path, default=DEFAULT_RECEIVED_LOG_PATH)
    args, _unknown = parser.parse_known_args()
    return args


def import_runtime_dependencies() -> tuple[Any, Any]:
    try:
        import traci
    except Exception as exc:
        raise RuntimeError("TraCI is required. Install SUMO and ensure its Python tools are available.") from exc

    try:
        from kafka import KafkaConsumer
    except Exception as exc:
        raise RuntimeError("kafka-python is required. Install it with: pip install kafka-python") from exc

    return traci, KafkaConsumer


def open_received_log(path: Path) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=RECEIVED_LOG_FIELDS)
    writer.writeheader()
    handle.flush()
    return handle, writer


def main() -> int:
    args = parse_args()

    protection.TARGET_JUNCTION_ID = str(args.junction_id)
    protection.LANE_LEADS_TO_TARGET_CACHE = {}

    try:
        traci, KafkaConsumer = import_runtime_dependencies()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    protection_handle, protection_writer = protection.open_protection_log(args.protection_log)
    received_handle, received_writer = open_received_log(args.received_log)

    alert_lock = threading.Lock()
    stop_event = threading.Event()
    high_alerts_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    visual_alert_events: list[dict[str, Any]] = []
    consumer_holder: dict[str, Any] = {}

    def handle_payload(payload_text: str, topic: str) -> None:
        received_at = utc_now_iso()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {"message": payload_text}

        latency_ms = calculate_latency_ms(payload)
        risk_level = str(payload.get("risk_level", ""))
        vehicle_1 = str(payload.get("vehicle_1", ""))
        vehicle_2 = str(payload.get("vehicle_2", ""))
        controller_action = "LOG_ONLY"

        if risk_level in {"HIGH", "LOW"} and vehicle_1 and vehicle_2:
            with alert_lock:
                queue_alert_visual(visual_alert_events, payload, risk_level, vehicle_1, vehicle_2)
            print(format_compact_alert(received_at, topic, payload, latency_ms))

        if risk_level == "HIGH" and vehicle_1 and vehicle_2:
            with alert_lock:
                payload["_controller_received_perf_time"] = time.perf_counter()
                high_alerts_by_pair[normalize_pair(vehicle_1, vehicle_2)] = payload
            controller_action = "QUEUED_FOR_TRACI_CONTROL"

        received_writer.writerow(
            {
                "received_at_utc": received_at,
                "protocol": payload.get("protocol", "kafka"),
                "alert_id": payload.get("alert_id", ""),
                "topic": topic,
                "simulation_time": payload.get("simulation_time", ""),
                "risk_level": risk_level,
                "episode_status": payload.get("episode_status", ""),
                "vehicle_1": vehicle_1,
                "vehicle_2": vehicle_2,
                "arrival_time_difference_s": payload.get("arrival_time_difference_s", ""),
                "latency_ms": "" if latency_ms is None else round(float(latency_ms), 4),
                "controller_action": controller_action,
                "payload_json": json.dumps(payload, separators=(",", ":")),
            }
        )
        received_handle.flush()

    def consume_alerts() -> None:
        consumer = None
        try:
            consumer = KafkaConsumer(
                args.topic,
                bootstrap_servers=args.kafka_bootstrap_servers,
                group_id=args.group_id,
                client_id=args.client_id,
                auto_offset_reset=args.auto_offset_reset,
                enable_auto_commit=True,
                value_deserializer=lambda payload: payload.decode("utf-8", errors="replace"),
            )
            consumer_holder["consumer"] = consumer
            print(f"Scenario1 Kafka subscriber-controller subscribed to {args.topic}")
            for message in consumer:
                if stop_event.is_set():
                    break
                handle_payload(str(message.value), str(message.topic))
        except Exception as exc:
            if not stop_event.is_set():
                print(str(exc), file=sys.stderr)
        finally:
            if consumer is not None:
                try:
                    consumer.close()
                except Exception:
                    pass

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
    visual_state: dict[str, Any] = {
        "vehicle_alert_labels": {},
    }

    step = 0
    protections_applied = 0
    releases = 0
    consumer_thread = threading.Thread(target=consume_alerts, daemon=True)

    try:
        consumer_thread.start()

        traci = connect_traci_with_retry(traci, args.traci_host, args.traci_port)
        traci.setOrder(int(args.traci_client_order))

        junction_ids = set(traci.junction.getIDList())
        if args.junction_id not in junction_ids:
            raise RuntimeError(f"Junction '{args.junction_id}' was not found in the SUMO network.")

        junction_x, junction_y = traci.junction.getPosition(args.junction_id)
        protection.configure_protected_vehicle_types(traci)

        print(
            "Scenario1 Kafka subscriber-controller connected to TraCI. "
            "It will control vehicles only after receiving HIGH Kafka alerts."
        )

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = float(traci.simulation.getTime())
            active_vehicle_ids = set(map(str, traci.vehicle.getIDList()))

            with alert_lock:
                high_alerts: list[dict[str, Any]] = []
                new_visual_alerts = list(visual_alert_events)
                visual_alert_events.clear()
                now_perf = time.perf_counter()
                for pair, alert in list(high_alerts_by_pair.items()):
                    age_s = now_perf - float(alert.get("_controller_received_perf_time", now_perf))
                    if age_s > args.alert_memory_s:
                        del high_alerts_by_pair[pair]
                        continue
                    if str(alert.get("vehicle_1", "")) in active_vehicle_ids or str(alert.get("vehicle_2", "")) in active_vehicle_ids:
                        high_alerts.append(dict(alert))

            current_states = collect_current_states(traci, active_vehicle_ids, sim_time, float(junction_x), float(junction_y))

            for alert in new_visual_alerts:
                apply_alert_visual(
                    traci,
                    visual_state,
                    alert,
                    active_vehicle_ids,
                    sim_time,
                    float(junction_x),
                    float(junction_y),
                )
            expire_alert_visuals(traci, visual_state, active_vehicle_ids, sim_time)

            if high_alerts:
                new_protections, new_releases = protection.maintain_intersection_gate_controls(
                    traci,
                    protection_writer,
                    protection_handle,
                    gate_state,
                    active_vehicle_ids,
                    current_states,
                    sim_time,
                    float(junction_x),
                    float(junction_y),
                    high_alerts,
                )
                protections_applied += new_protections
                releases += new_releases
            else:
                for vehicle_id in list(gate_state.get("controlled", {})):
                    if vehicle_id in active_vehicle_ids:
                        protection.release_gate_vehicle(traci, gate_state, vehicle_id, sim_time)
                        releases += 1

            if step % 20 == 0:
                print(
                    f"Kafka subscriber-controller status | active={len(active_vehicle_ids)} | "
                    f"high_alert_pairs={len(high_alerts)} | protections={protections_applied} | releases={releases}"
                )

            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break

        print(
            "Scenario1 Kafka subscriber-controller finished. "
            f"Steps={step}, protections_applied={protections_applied}, releases={releases}"
        )
        return 0
    except KeyboardInterrupt:
        print("Scenario1 Kafka subscriber-controller stopped.")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        stop_event.set()
        try:
            consumer = consumer_holder.get("consumer")
            if consumer is not None:
                consumer.close()
        except Exception:
            pass
        try:
            traci.close()
        except Exception:
            pass
        protection_handle.close()
        received_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
