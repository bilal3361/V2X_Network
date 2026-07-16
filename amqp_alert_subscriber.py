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


DEFAULT_AMQP_HOST = "localhost"
DEFAULT_AMQP_PORT = 5672
DEFAULT_EXCHANGE = "v2x"
DEFAULT_EXCHANGE_TYPE = "topic"
DEFAULT_QUEUE = "v2x.alerts"
DEFAULT_ROUTING_KEYS = ["alerts.high", "alerts.low"]
DEFAULT_CSV_PATH = DATA_DIR / "amqp_alert_log.csv"
LOCAL_TZ = ZoneInfo("Europe/Rome")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subscribe to AMQP/RabbitMQ V2X alerts and log latency.")
    parser.add_argument("--amqp-host", default=DEFAULT_AMQP_HOST, help="RabbitMQ host.")
    parser.add_argument("--amqp-port", type=int, default=DEFAULT_AMQP_PORT, help="RabbitMQ AMQP port.")
    parser.add_argument("--exchange", default=DEFAULT_EXCHANGE, help="RabbitMQ topic exchange.")
    parser.add_argument("--exchange-type", default=DEFAULT_EXCHANGE_TYPE, help="RabbitMQ exchange type.")
    parser.add_argument("--queue", default=DEFAULT_QUEUE, help="RabbitMQ queue name.")
    parser.add_argument(
        "--routing-key",
        action="append",
        dest="routing_keys",
        help="Routing key to bind. Repeat for multiple keys. Defaults to alerts.high and alerts.low.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--append-csv", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def import_pika() -> Any:
    try:
        import pika
    except Exception as exc:
        raise RuntimeError("pika is required for AMQP subscription. Install it with: pip install pika") from exc
    return pika


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
        "exchange",
        "routing_key",
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
    routing_keys = args.routing_keys or DEFAULT_ROUTING_KEYS

    try:
        pika = import_pika()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    csv_handle = None
    csv_writer = None
    if not args.no_csv:
        csv_handle, csv_writer = ensure_csv_writer(args.csv, append=args.append_csv)

    connection = None
    try:
        parameters = pika.ConnectionParameters(host=args.amqp_host, port=args.amqp_port)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.exchange_declare(exchange=args.exchange, exchange_type=args.exchange_type, durable=True)
        channel.queue_declare(queue=args.queue, durable=True)
        for routing_key in routing_keys:
            channel.queue_bind(exchange=args.exchange, queue=args.queue, routing_key=routing_key)

        print(
            f"Subscribed to AMQP exchange {args.exchange} queue {args.queue} "
            f"on {args.amqp_host}:{args.amqp_port}"
        )

        def on_message(channel: Any, method: Any, properties: Any, body: bytes) -> None:
            received_at = utc_now_iso()
            payload_text = body.decode("utf-8", errors="replace")
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                payload = {"message": payload_text}

            latency_ms = calculate_latency_ms(payload)
            print(format_compact_alert(received_at, method.routing_key, payload, latency_ms))

            if csv_writer is not None and csv_handle is not None:
                csv_writer.writerow(
                    {
                        "received_at_utc": received_at,
                        "received_at_cest": format_cest_time(received_at),
                        "generated_at_cest": format_alert_generated_time(payload),
                        "protocol": payload.get("protocol", "amqp"),
                        "alert_id": payload.get("alert_id", ""),
                        "exchange": method.exchange,
                        "routing_key": method.routing_key,
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

            channel.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_qos(prefetch_count=100)
        channel.basic_consume(queue=args.queue, on_message_callback=on_message, auto_ack=False)
        channel.start_consuming()

    except KeyboardInterrupt:
        print("AMQP subscriber stopped.")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            try:
                if connection.is_open:
                    connection.close()
            except Exception:
                pass
        if csv_handle is not None:
            csv_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
