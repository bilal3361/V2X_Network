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

from mqtt_alert_subscriber import format_compact_alert
from v2x_task5_common import DATA_DIR, utc_now_iso


DEFAULT_KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "v2x.alerts"
DEFAULT_CSV_PATH = DATA_DIR / "kafka_alert_log.csv"
LOCAL_TZ = ZoneInfo("Europe/Rome")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subscribe to Kafka V2X alerts and log latency.")
    parser.add_argument("--kafka-bootstrap-servers", default=DEFAULT_KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--group-id", default="v2x-kafka-alert-subscriber")
    parser.add_argument("--client-id", default="v2x-kafka-alert-subscriber")
    parser.add_argument("--auto-offset-reset", choices=["latest", "earliest"], default="latest")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--append-csv", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def import_kafka_consumer() -> Any:
    try:
        from kafka import KafkaConsumer
    except Exception as exc:
        raise RuntimeError("kafka-python is required. Install it with: pip install kafka-python") from exc
    return KafkaConsumer


def ensure_csv_writer(path: Path, append: bool = False) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    is_new = not append or not path.exists() or path.stat().st_size == 0
    handle = path.open(mode, newline="", encoding="utf-8")
    fieldnames = [
        "received_at_utc",
        "received_at_cest",
        "generated_at_cest",
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
        "payload_json",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
        handle.flush()
    return handle, writer


def format_value(value: Any, suffix: str = "", precision: int = 2) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def format_cest_time(value: Any) -> str:
    if not value:
        return "NA"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone(LOCAL_TZ)
    return f"{local_dt:%H:%M:%S}.{local_dt.microsecond // 1000:03d} {local_dt.tzname()}"


def format_alert_generated_time(payload: dict[str, Any]) -> str:
    return format_cest_time(payload.get("generated_at_utc") or payload.get("sent_wall_time_utc"))


def calculate_latency_ms(payload: dict[str, Any]) -> float | None:
    sent_perf_time = payload.get("sent_perf_time")
    try:
        return (time.perf_counter() - float(sent_perf_time)) * 1000.0
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()

    try:
        KafkaConsumer = import_kafka_consumer()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    csv_handle = None
    csv_writer = None
    if not args.no_csv:
        csv_handle, csv_writer = ensure_csv_writer(args.csv, append=args.append_csv)

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.kafka_bootstrap_servers,
        group_id=args.group_id,
        client_id=args.client_id,
        auto_offset_reset=args.auto_offset_reset,
        enable_auto_commit=True,
        value_deserializer=lambda payload: payload.decode("utf-8", errors="replace"),
    )

    print(f"Subscribed to Kafka topic {args.topic} on {args.kafka_bootstrap_servers}")

    try:
        for message in consumer:
            received_at = utc_now_iso()
            payload_text = message.value
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                payload = {"message": payload_text}

            latency_ms = calculate_latency_ms(payload)
            print(format_compact_alert(received_at, message.topic, payload, latency_ms))

            if csv_writer is not None and csv_handle is not None:
                csv_writer.writerow(
                    {
                        "received_at_utc": received_at,
                        "received_at_cest": format_cest_time(received_at),
                        "generated_at_cest": format_alert_generated_time(payload),
                        "protocol": payload.get("protocol", "kafka"),
                        "alert_id": payload.get("alert_id", ""),
                        "topic": message.topic,
                        "simulation_time": payload.get("simulation_time", ""),
                        "risk_level": payload.get("risk_level", ""),
                        "episode_status": payload.get("episode_status", ""),
                        "vehicle_1": payload.get("vehicle_1", ""),
                        "vehicle_2": payload.get("vehicle_2", ""),
                        "arrival_time_difference_s": payload.get("arrival_time_difference_s", ""),
                        "latency_ms": "" if latency_ms is None else round(float(latency_ms), 4),
                        "payload_json": json.dumps(payload, separators=(",", ":")),
                    }
                )
                csv_handle.flush()

    except KeyboardInterrupt:
        print("Kafka subscriber stopped.")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        consumer.close()
        if csv_handle is not None:
            csv_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
