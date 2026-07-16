# Protocol Comparison Summary

## Method

This report compares saved sent-alert logs and subscriber logs for MQTT, Kafka, and AMQP/RabbitMQ. The same SUMO scenario, same trained LSTM trajectory model, same risk logic, same vehicle pair, and same alert payload were used. Only the communication protocol changed: MQTT/Mosquitto, Kafka/Apache Kafka, and AMQP/RabbitMQ.

The script does not run SUMO, Docker, MQTT, Kafka, or AMQP. It only analyzes existing CSV logs.
Protocols with no sent or received logs: None.

## Metric Meaning

- Latency is the primary metric because V2X collision alerts must arrive before the collision.
- Delivery success rate is important because alerts must not be lost.
- Latency stability is important because consistent delivery is safer than unstable delivery.
- Throughput is intentionally removed from this comparison so the report focuses on delivery success and latency for the same Scenario1 traffic inputs.
- Delivery success is calculated from sent-alert logs and subscriber received logs when both are available.

## Results

| Protocol | Sent | Received | Delivery % | Matched % | Matched | Missing | Duplicates | Avg latency (ms) | Std (ms) | Range (ms) | CV | HIGH | LOW |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MQTT | 26 | 26 | 100.0000 | 100.0000 | 26 | 0 | 0 | 1.6126 | 0.3721 | 1.5415 | 0.2307 | 18 | 8 |
| KAFKA | 25 | 25 | 100.0000 | 100.0000 | 25 | 0 | 0 | 4.0362 | 2.0224 | 10.8397 | 0.5011 | 17 | 8 |
| AMQP | 25 | 25 | 100.0000 | 100.0000 | 25 | 0 | 0 | 2.7353 | 1.1799 | 4.7172 | 0.4314 | 17 | 8 |

## Recommendation From This Local Experiment

- Best latency protocol: MQTT (1.6126 ms).
- Most stable protocol: MQTT (0.2307).
- Best delivery protocol: MQTT (100.0000 %).
- Best overall protocol should consider latency, stability, delivery success, reliability features, setup complexity, and deployment constraints.
- Do not claim any protocol is always best; this recommendation is based only on the measured local experiment.

## Protocol Notes

- MQTT/Mosquitto is lightweight and suitable for simple real-time IoT/V2X alerts.
- Kafka/Apache Kafka is suitable for high-throughput event streaming and replay/history.
- AMQP/RabbitMQ is suitable for reliable queue-based delivery and routing.

## Outputs

- CSV summary: `Scenario1/data/protocol_comparison_summary.csv`
- Average latency chart: `Scenario1/plots/protocol_comparison_latency.png`
- Latency stability chart: `Scenario1/plots/protocol_comparison_latency_stability.png`
- Delivery success chart: `Scenario1/plots/protocol_comparison_delivery_success.png`
