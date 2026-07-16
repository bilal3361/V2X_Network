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
#These libraries are used for command-line options, saving CSV files,reading JSON MQTT messages, printing errors, and handling file paths.

from v2x_task5_common import DATA_DIR, utc_now_iso
#This is used when saving received alert time in the CSV file.


DEFAULT_CSV_PATH = DATA_DIR / "mqtt_alert_log.csv" #All received MQTT alerts can be saved there.
LOCAL_TZ = ZoneInfo("Europe/Rome")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subscribe to task 5 MQTT V2X alerts.")
    parser.add_argument("--broker-host", default="localhost", help="MQTT broker host.")
    parser.add_argument("--broker-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--client-id", default="v2x-task5-alert-subscriber", help="MQTT client ID.")
    parser.add_argument("--topic", default="v2x/alerts", help="MQTT topic filter.")
    parser.add_argument(
        "--display-mode",
        choices=["compact", "summary", "plain", "json"],
        default="compact",
        help="How alerts should be printed in the terminal.",
    )
    parser.add_argument(
        "--summary-interval-s",
        type=float,
        default=5.0,
        help="Wall-clock seconds between grouped summary prints when --display-mode summary is used.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="CSV log path. Use --no-csv to disable CSV logging.",
    )
    parser.add_argument(
        "--append-csv",
        action="store_true",
        help="Append to the CSV log instead of replacing it at subscriber startup.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV logging.")
    return parser.parse_args()


def import_mqtt() -> Any:                  #       import mqtt libraries
    try:
        import paho.mqtt.client as mqtt # This library allows Python to connect to an MQTT broker.
    except Exception as exc:
        raise RuntimeError(
            "paho-mqtt is required for MQTT subscription. Install it with:\n"
            "  pip install paho-mqtt"
        ) from exc
    return mqtt


def make_mqtt_client(mqtt: Any, client_id: str) -> Any: #create client   #This client will connect to the broker and subscribe to topics..
    if hasattr(mqtt, "CallbackAPIVersion"):
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    return mqtt.Client(client_id=client_id)


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
        "risk_source",
        "vehicle_1",
        "vehicle_2",
        "arrival_time_difference_s",
        "latency_ms",
        "recommendation",
        "message",
        "payload_json",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
        handle.flush()
    return handle, writer


def format_value(value: Any, suffix: str = "", precision: int = 2) -> str: #Makes values clean before printing.
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def calculate_latency_ms(payload: dict[str, Any]) -> float | None:
    sent_perf_time = payload.get("sent_perf_time")
    try:
        return (time.perf_counter() - float(sent_perf_time)) * 1000.0
    except (TypeError, ValueError):
        return None


def format_latency(latency_ms: float | None) -> str:
    if latency_ms is None:
        return "NA"
    return f"{latency_ms:.2f}ms"


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


def format_compact_alert(received_at: str, topic: str, payload: dict[str, Any], latency_ms: float | None) -> str: #Creates the short alert line you see in terminal.
    risk = str(payload.get("risk_level", "UNKNOWN"))
    status = str(payload.get("episode_status", "DETECTED"))
    vehicle_1 = str(payload.get("vehicle_1", ""))
    vehicle_2 = str(payload.get("vehicle_2", ""))
    pair = f"{vehicle_1} -> {vehicle_2}"
    arrival_diff = format_value(payload.get("arrival_time_difference_s"), "s")
    prediction_time = format_value(payload.get("prediction_time_s", payload.get("simulation_time")), "s")
    predicted_collision_time = format_value(payload.get("predicted_collision_time_s"), "s")
    generated_at = format_alert_generated_time(payload)

    return (
        f"{risk:<4} | {status:<14} | "
        f"prediction_time={prediction_time} | "
        f"predicted_collision_time={predicted_collision_time} | "
        f"pair={pair:<34} | arrival_diff={arrival_diff} | "
        f"generated={generated_at} | latency={format_latency(latency_ms)}"
    )

def alert_sort_key(payload: dict[str, Any]) -> tuple[object, ...]: #Used in summary mode to sort alerts. high risk shown first
    risk_priority = {"HIGH": 2, "LOW": 1}.get(str(payload.get("risk_level")), 0)
    arrival_diff = payload.get("arrival_time_difference_s")

    try:
        arrival_value = float(arrival_diff)
    except (TypeError, ValueError):
        arrival_value = 999999.0

    return (
        -risk_priority,
        arrival_value,
        str(payload.get("vehicle_1", "")),
        str(payload.get("vehicle_2", "")),
    )


def flush_summary(summary_buffer: list[dict[str, Any]]) -> None:
    if not summary_buffer:
        return

    sorted_alerts = sorted(summary_buffer, key=alert_sort_key)
    high_count = sum(1 for item in summary_buffer if item.get("risk_level") == "HIGH")
    low_count = sum(1 for item in summary_buffer if item.get("risk_level") == "LOW")

    print(
        f"\nAlert summary | HIGH={high_count} LOW={low_count} total={len(summary_buffer)}"
    )

    for payload in sorted_alerts[:5]:
        risk = str(payload.get("risk_level", "UNKNOWN"))
        status = str(payload.get("episode_status", "DETECTED"))
        vehicle_1 = str(payload.get("vehicle_1", ""))
        vehicle_2 = str(payload.get("vehicle_2", ""))
        arrival_diff = format_value(payload.get("arrival_time_difference_s"), "s")
        prediction_time = format_value(payload.get("prediction_time_s", payload.get("simulation_time")), "s")
        predicted_collision_time = format_value(payload.get("predicted_collision_time_s"), "s")
        print(
            f"  {risk:<4} | {status:<14} | {vehicle_1} -> {vehicle_2} | "
            f"prediction_time={prediction_time} | "
            f"predicted_collision_time={predicted_collision_time} | "
            f"arrival_diff={arrival_diff}"
        )

    summary_buffer.clear()


def main() -> int: #This is where the script actually starts running.
    args = parse_args()

    try:
        mqtt = import_mqtt()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    csv_handle = None
    csv_writer = None
    if not args.no_csv:
        csv_handle, csv_writer = ensure_csv_writer(args.csv, append=args.append_csv)

    client = make_mqtt_client(mqtt, args.client_id)
    summary_buffer: list[dict[str, Any]] = []
    last_summary_flush_time = time.monotonic()

    def on_connect(client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        client.subscribe(args.topic)                        ##########                               Subscribes to alert topic
        print(f"Subscribed to {args.topic} on {args.broker_host}:{args.broker_port}")

    def on_message(client: Any, userdata: Any, msg: Any) -> None:
        nonlocal last_summary_flush_time

        received_at = utc_now_iso()
        payload_text = msg.payload.decode("utf-8", errors="replace")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {"message": payload_text}

        latency_ms = calculate_latency_ms(payload)
        protocol = payload.get("protocol", "mqtt")
        alert_id = payload.get("alert_id", "")
        risk = payload.get("risk_level", "UNKNOWN")
        vehicle_1 = payload.get("vehicle_1", "")
        vehicle_2 = payload.get("vehicle_2", "")
        sim_time = payload.get("simulation_time", "")
        message = payload.get("message", "")
        prediction_time = payload.get("prediction_time_s", sim_time)
        predicted_collision_time = payload.get("predicted_collision_time_s", "")

        if args.display_mode == "json":
            print(json.dumps(payload, indent=2))
        elif args.display_mode == "plain":
            print(
                f"[{format_cest_time(received_at)}] {msg.topic} | {risk} | "
                f"prediction_time={prediction_time}s | "
                f"predicted_collision_time={predicted_collision_time}s | "
                f"{vehicle_1} {vehicle_2} | {message} | "
                f"generated={format_alert_generated_time(payload)} | latency={format_latency(latency_ms)}"
            )
        elif args.display_mode == "summary":
            summary_buffer.append(payload)
            now = time.monotonic()
            if (now - last_summary_flush_time) >= args.summary_interval_s:
                flush_summary(summary_buffer)
                last_summary_flush_time = now
        else:
            print(format_compact_alert(received_at, msg.topic, payload, latency_ms))

        if csv_writer is not None and csv_handle is not None:
            csv_writer.writerow(
                {
                    "received_at_utc": received_at,
                    "received_at_cest": format_cest_time(received_at),
                    "generated_at_cest": format_alert_generated_time(payload),
                    "protocol": protocol,
                    "alert_id": alert_id,
                    "topic": msg.topic,
                    "simulation_time": sim_time,
                    "risk_level": risk,
                    "episode_status": payload.get("episode_status", ""),
                    "risk_source": payload.get("risk_source", ""),
                    "vehicle_1": vehicle_1,
                    "vehicle_2": vehicle_2,
                    "arrival_time_difference_s": payload.get("arrival_time_difference_s", ""),
                    "latency_ms": "" if latency_ms is None else round(float(latency_ms), 4),
                    "recommendation": payload.get("recommendation", ""),
                    "message": message,
                    "payload_json": json.dumps(payload, separators=(",", ":")),
                }
            )
            csv_handle.flush()

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.broker_host, args.broker_port, keepalive=60)      #Connects to Mosquitto broker.
        client.loop_forever()                                                    #Keeps the subscriber running.
    except KeyboardInterrupt:
        print("Subscriber stopped.")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if args.display_mode == "summary":
            flush_summary(summary_buffer)
        if csv_handle is not None:
            csv_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
