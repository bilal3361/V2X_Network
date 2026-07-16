from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


SCENARIO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCENARIO_DIR.parent
DATA_DIR = SCENARIO_DIR / "data"
PLOTS_DIR = SCENARIO_DIR / "plots"
DOCS_DIR = SCENARIO_DIR / "docs"

PROTOCOL_LOGS = {
    "mqtt": {
        "sent": DATA_DIR / "mqtt_sent_alert_log.csv",
        "received": DATA_DIR / "mqtt_alert_log.csv",
    },
    "kafka": {
        "sent": DATA_DIR / "kafka_sent_alert_log.csv",
        "received": DATA_DIR / "kafka_alert_log.csv",
    },
    "amqp": {
        "sent": DATA_DIR / "amqp_sent_alert_log.csv",
        "received": DATA_DIR / "amqp_alert_log.csv",
    },
}

SUMMARY_CSV_PATH = DATA_DIR / "protocol_comparison_summary.csv"
SUMMARY_MD_PATH = DOCS_DIR / "protocol_comparison_summary.md"
LATENCY_PLOT_PATH = PLOTS_DIR / "protocol_comparison_latency.png"
STABILITY_PLOT_PATH = PLOTS_DIR / "protocol_comparison_latency_stability.png"
DELIVERY_PLOT_PATH = PLOTS_DIR / "protocol_comparison_delivery_success.png"

SUMMARY_FIELDS = [
    "protocol",
    "sent_log_file",
    "received_log_file",
    "sent_alert_count",
    "received_alert_count",
    "matched_alert_count",
    "missing_alert_count",
    "duplicate_received_count",
    "delivery_success_rate_percent",
    "matched_delivery_success_rate_percent",
    "delivery_success_note",
    "latency_sample_count",
    "average_latency_ms",
    "median_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "std_latency_ms",
    "latency_range_ms",
    "coefficient_of_variation",
    "high_alert_count",
    "low_alert_count",
]

PROTOCOL_NOTES = {
    "mqtt": "MQTT/Mosquitto is lightweight and suitable for simple real-time IoT/V2X alerts.",
    "kafka": "Kafka/Apache Kafka is suitable for high-throughput event streaming and replay/history.",
    "amqp": "AMQP/RabbitMQ is suitable for reliable queue-based delivery and routing.",
}


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def fmt(value: float | int | str | None, precision: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def rel(path: Path) -> str:
    return str(path.relative_to(SCENARIO_DIR))


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def payload_json(row: dict[str, Any]) -> dict[str, Any]:
    payload_text = row.get("payload_json", "")
    if not payload_text:
        return {}
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def value_from_row_or_payload(row: dict[str, Any], field: str) -> Any:
    value = row.get(field)
    if value not in (None, ""):
        return value
    return payload_json(row).get(field)


def extract_alert_id(row: dict[str, Any]) -> str | None:
    value = value_from_row_or_payload(row, "alert_id")
    if value in (None, ""):
        return None
    return str(value)


def alert_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [alert_id for row in rows if (alert_id := extract_alert_id(row)) is not None]


def matched_alert_count(sent_ids: list[str], received_ids: list[str]) -> int | None:
    if not sent_ids or not received_ids:
        return None
    sent_counter = Counter(sent_ids)
    received_counter = Counter(received_ids)
    return sum(min(sent_count, received_counter.get(alert_id, 0)) for alert_id, sent_count in sent_counter.items())


def delivery_metrics(
    sent_path: Path,
    received_path: Path,
    sent_rows: list[dict[str, Any]],
    received_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sent_exists = sent_path.exists()
    received_exists = received_path.exists()

    sent_count = len(sent_rows) if sent_exists else None
    received_count = len(received_rows) if received_exists else None

    sent_ids = alert_ids(sent_rows)
    received_ids = alert_ids(received_rows)
    duplicate_received_count = (len(received_ids) - len(set(received_ids))) if received_ids else None

    if not sent_exists:
        delivery_success_rate: float | str | None = "not available - sent log missing"
        delivery_note = "Run the protocol engine to create the sent log."
    elif not received_exists:
        delivery_success_rate = "not available - received log missing"
        delivery_note = "Run the protocol subscriber to create the received log."
    elif sent_count == 0:
        delivery_success_rate = "not available - sent log empty"
        delivery_note = "Sent log exists but contains no alert rows."
    else:
        delivery_success_rate = (float(received_count or 0) / float(sent_count)) * 100.0
        delivery_note = "calculated as received_alert_count / sent_alert_count * 100"

    matched_count = None
    missing_count = None
    matched_rate = None
    if sent_exists and received_exists and sent_count and sent_ids and received_ids:
        matched_count = matched_alert_count(sent_ids, received_ids)
        missing_count = sent_count - int(matched_count or 0)
        matched_rate = (float(matched_count or 0) / float(sent_count)) * 100.0

    return {
        "sent_alert_count": sent_count,
        "received_alert_count": received_count,
        "matched_alert_count": matched_count,
        "missing_alert_count": missing_count,
        "duplicate_received_count": duplicate_received_count,
        "delivery_success_rate_percent": delivery_success_rate,
        "matched_delivery_success_rate_percent": matched_rate,
        "delivery_success_note": delivery_note,
    }


def latency_metrics(received_rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        latency
        for row in received_rows
        if (latency := parse_float(row.get("latency_ms"))) is not None
    ]
    if not latencies:
        return {
            "latency_sample_count": 0,
            "average_latency_ms": None,
            "median_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "std_latency_ms": None,
            "latency_range_ms": None,
            "coefficient_of_variation": None,
        }

    average_latency = mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_latency = pstdev(latencies) if len(latencies) > 1 else 0.0
    return {
        "latency_sample_count": len(latencies),
        "average_latency_ms": average_latency,
        "median_latency_ms": median(latencies),
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "std_latency_ms": std_latency,
        "latency_range_ms": max_latency - min_latency,
        "coefficient_of_variation": std_latency / average_latency if average_latency else None,
    }


def count_risks(rows: list[dict[str, Any]]) -> tuple[int, int]:
    high_count = 0
    low_count = 0
    for row in rows:
        risk = str(value_from_row_or_payload(row, "risk_level") or "").upper()
        if risk == "HIGH":
            high_count += 1
        elif risk == "LOW":
            low_count += 1
    return high_count, low_count


def summarize_protocol(protocol: str, sent_path: Path, received_path: Path) -> dict[str, Any] | None:
    if not sent_path.exists() and not received_path.exists():
        return None

    sent_rows = load_rows(sent_path)
    received_rows = load_rows(received_path)
    high_count, low_count = count_risks(received_rows)

    summary: dict[str, Any] = {
        "protocol": protocol,
        "sent_log_file": rel(sent_path) if sent_path.exists() else "",
        "received_log_file": rel(received_path) if received_path.exists() else "",
        "high_alert_count": high_count,
        "low_alert_count": low_count,
    }
    summary.update(delivery_metrics(sent_path, received_path, sent_rows, received_rows))
    summary.update(latency_metrics(received_rows))
    return summary


def write_summary_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: fmt(summary.get(field)) for field in SUMMARY_FIELDS})


def numeric_summaries(summaries: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return [
        summary
        for summary in summaries
        if isinstance(summary.get(metric), (int, float)) and math.isfinite(float(summary[metric]))
    ]


def lowest_metric_protocol(summaries: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = numeric_summaries(summaries, metric)
    return min(candidates, key=lambda summary: float(summary[metric])) if candidates else None


def highest_metric_protocol(summaries: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = numeric_summaries(summaries, metric)
    return max(candidates, key=lambda summary: float(summary[metric])) if candidates else None


def delivery_chart_value(summary: dict[str, Any]) -> float | None:
    matched = summary.get("matched_delivery_success_rate_percent")
    if isinstance(matched, (int, float)):
        return float(matched)
    raw = summary.get("delivery_success_rate_percent")
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def best_delivery_protocol(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    matched_candidates = numeric_summaries(summaries, "matched_delivery_success_rate_percent")
    if matched_candidates:
        return max(matched_candidates, key=lambda summary: float(summary["matched_delivery_success_rate_percent"])), "matched_delivery_success_rate_percent"
    raw_candidates = numeric_summaries(summaries, "delivery_success_rate_percent")
    if raw_candidates:
        return max(raw_candidates, key=lambda summary: float(summary["delivery_success_rate_percent"])), "delivery_success_rate_percent"
    return None, "delivery_success_rate_percent"


def markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| Protocol | Sent | Received | Delivery % | Matched % | Matched | Missing | Duplicates | Avg latency (ms) | Std (ms) | Range (ms) | CV | HIGH | LOW |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {protocol} | {sent} | {received} | {delivery} | {matched_rate} | {matched} | {missing} | {dupes} | {avg} | {std} | {rangev} | {cv} | {high} | {low} |".format(
                protocol=str(summary["protocol"]).upper(),
                sent=fmt(summary["sent_alert_count"]) or "-",
                received=fmt(summary["received_alert_count"]) or "-",
                delivery=fmt(summary["delivery_success_rate_percent"]) or "-",
                matched_rate=fmt(summary["matched_delivery_success_rate_percent"]) or "-",
                matched=fmt(summary["matched_alert_count"]) or "-",
                missing=fmt(summary["missing_alert_count"]) or "-",
                dupes=fmt(summary["duplicate_received_count"]) or "-",
                avg=fmt(summary["average_latency_ms"]) or "-",
                std=fmt(summary["std_latency_ms"]) or "-",
                rangev=fmt(summary["latency_range_ms"]) or "-",
                cv=fmt(summary["coefficient_of_variation"]) or "-",
                high=summary["high_alert_count"],
                low=summary["low_alert_count"],
            )
        )
    return "\n".join(lines)


def best_text(summary: dict[str, Any] | None, metric: str, unit: str = "") -> str:
    if summary is None:
        return "not available"
    value = fmt(summary.get(metric))
    suffix = f" {unit}" if unit and value else ""
    return f"{summary['protocol'].upper()} ({value}{suffix})"


def write_markdown_report(
    summaries: list[dict[str, Any]],
    missing_protocols: list[str],
    path: Path,
) -> None:
    best_latency = lowest_metric_protocol(summaries, "average_latency_ms")
    most_stable = lowest_metric_protocol(summaries, "coefficient_of_variation")
    if most_stable is None:
        most_stable = lowest_metric_protocol(summaries, "std_latency_ms")
    best_delivery, delivery_metric = best_delivery_protocol(summaries)

    missing_text = "None" if not missing_protocols else ", ".join(protocol.upper() for protocol in missing_protocols)
    available_notes = [
        f"- {PROTOCOL_NOTES[protocol]}"
        for protocol in ("mqtt", "kafka", "amqp")
        if any(summary["protocol"] == protocol for summary in summaries)
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Protocol Comparison Summary",
                "",
                "## Method",
                "",
                "This report compares saved sent-alert logs and subscriber logs for MQTT, Kafka, and AMQP/RabbitMQ. "
                "The same SUMO scenario, same trained LSTM trajectory model, same risk logic, same vehicle pair, "
                "and same alert payload were used. Only the communication protocol changed: "
                "MQTT/Mosquitto, Kafka/Apache Kafka, and AMQP/RabbitMQ.",
                "",
                "The script does not run SUMO, Docker, MQTT, Kafka, or AMQP. It only analyzes existing CSV logs.",
                f"Protocols with no sent or received logs: {missing_text}.",
                "",
                "## Metric Meaning",
                "",
                "- Latency is the primary metric because V2X collision alerts must arrive before the collision.",
                "- Delivery success rate is important because alerts must not be lost.",
                "- Latency stability is important because consistent delivery is safer than unstable delivery.",
                "- Throughput is intentionally removed from this comparison so the report focuses on delivery success and latency for the same Scenario1 traffic inputs.",
                "- Delivery success is calculated from sent-alert logs and subscriber received logs when both are available.",
                "",
                "## Results",
                "",
                markdown_table(summaries) if summaries else "No protocol logs were available.",
                "",
                "## Recommendation From This Local Experiment",
                "",
                f"- Best latency protocol: {best_text(best_latency, 'average_latency_ms', 'ms')}.",
                f"- Most stable protocol: {best_text(most_stable, 'coefficient_of_variation')}.",
                f"- Best delivery protocol: {best_text(best_delivery, delivery_metric, '%')}.",
                "- Best overall protocol should consider latency, stability, delivery success, reliability features, setup complexity, and deployment constraints.",
                "- Do not claim any protocol is always best; this recommendation is based only on the measured local experiment.",
                "",
                "## Protocol Notes",
                "",
                *available_notes,
                "",
                "## Outputs",
                "",
                f"- CSV summary: `{SUMMARY_CSV_PATH.relative_to(REPO_ROOT)}`",
                f"- Average latency chart: `{LATENCY_PLOT_PATH.relative_to(REPO_ROOT)}`",
                f"- Latency stability chart: `{STABILITY_PLOT_PATH.relative_to(REPO_ROOT)}`",
                f"- Delivery success chart: `{DELIVERY_PLOT_PATH.relative_to(REPO_ROOT)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_matplotlib() -> Any:
    cache_root = Path(tempfile.gettempdir()) / "v2x_matplotlib_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_bar_chart(
    summaries: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt = prepare_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    chart_summaries = numeric_summaries(summaries, metric)
    labels = [summary["protocol"].upper() for summary in chart_summaries]
    values = [float(summary[metric]) for summary in chart_summaries]

    write_chart_values(plt, labels, values, ylabel, title, path)


def write_delivery_chart(summaries: list[dict[str, Any]], path: Path) -> None:
    plt = prepare_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    values = []
    for summary in summaries:
        value = delivery_chart_value(summary)
        if value is None:
            continue
        labels.append(summary["protocol"].upper())
        values.append(value)

    write_chart_values(plt, labels, values, "Delivery success (%)", "Protocol Delivery Success", path)


def write_chart_values(
    plt: Any,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if values:
        bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"][: len(values)])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax.text(0.5, 0.5, "No numeric data available", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    summaries: list[dict[str, Any]] = []
    missing_protocols: list[str] = []

    for protocol, paths in PROTOCOL_LOGS.items():
        summary = summarize_protocol(protocol, paths["sent"], paths["received"])
        if summary is None:
            missing_protocols.append(protocol)
            continue
        summaries.append(summary)

    write_summary_csv(summaries, SUMMARY_CSV_PATH)
    write_markdown_report(summaries, missing_protocols, SUMMARY_MD_PATH)
    write_bar_chart(
        summaries,
        "average_latency_ms",
        "Average latency (ms)",
        "Protocol Average Alert Latency",
        LATENCY_PLOT_PATH,
    )
    write_bar_chart(
        summaries,
        "std_latency_ms",
        "Latency standard deviation (ms)",
        "Protocol Latency Stability",
        STABILITY_PLOT_PATH,
    )
    write_delivery_chart(summaries, DELIVERY_PLOT_PATH)

    print(f"Wrote {SUMMARY_CSV_PATH.relative_to(SCENARIO_DIR)}")
    print(f"Wrote {SUMMARY_MD_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATENCY_PLOT_PATH.relative_to(SCENARIO_DIR)}")
    print(f"Wrote {STABILITY_PLOT_PATH.relative_to(SCENARIO_DIR)}")
    print(f"Wrote {DELIVERY_PLOT_PATH.relative_to(SCENARIO_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
