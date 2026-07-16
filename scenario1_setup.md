# Scenario1 Setup

Scenario1 now uses the Scenario1-style alert/subscriber-controller workflow, but
keeps the Scenario1 model setup and Scenario1 traffic files.

```text
alert engine -> protocol broker -> subscriber-controller -> TraCI vehicle control
```

## Model

- Model artifacts stay in `models/`.
- `task5_model_metadata.json` reports `input_len = 20`, `pred_len = 30`,
  `dt_seconds = 0.1`.
- This means the LSTM uses 2.0 seconds of past trajectory and predicts
  3.0 seconds into the future.
- The LSTM prediction logic and LOW/HIGH arrival-time risk logic are unchanged.

## Vehicle Scenario

- Active SUMO config: `osm.sumocfg`.
- Active route file: `scenario1_50.routes.xml`.
- Vehicle count: 50.
- Vehicle IDs run from `targeted_vehicle_001` to `targeted_vehicle_050`.
- Target junction: `cluster_255722000_4115305935`.
- Simulation time: `0` to `300` seconds.
- Step length: `0.1`.
- The route generation logic follows the Scenario1 50-vehicle pattern: 15
  intentional conflict pairs plus 20 randomized background vehicles.

## Warning-Only

The warning-only scripts publish LOW/HIGH alerts but do not control vehicles:

```bash
./run_scenario1_mqtt.sh
./run_scenario1_kafka.sh
./run_scenario1_amqp.sh
```

## Subscriber-Control Protection

The subscriber-control scripts match the Scenario1 approach. The engine starts
SUMO in TraCI multi-client mode, publishes alerts through the selected protocol,
and a separate subscriber-controller receives HIGH alerts and applies TraCI
yield/release control.

```bash
./run_scenario1_mqtt_subscriber_control.sh
./run_scenario1_kafka_subscriber_control.sh
./run_scenario1_amqp_subscriber_control.sh
```

## Logs

Protocol sent/received logs:

```text
data/mqtt_sent_alert_log.csv   + data/mqtt_alert_log.csv
data/kafka_sent_alert_log.csv  + data/kafka_alert_log.csv
data/amqp_sent_alert_log.csv   + data/amqp_alert_log.csv
```

Subscriber-controller logs:

```text
data/scenario1_subscriber_controller_received_alert_log.csv
data/scenario1_subscriber_controller_protection_log.csv
data/scenario1_kafka_subscriber_controller_received_alert_log.csv
data/scenario1_kafka_subscriber_controller_protection_log.csv
data/scenario1_amqp_subscriber_controller_received_alert_log.csv
data/scenario1_amqp_subscriber_controller_protection_log.csv
```

## Protocol Comparison

Start the needed broker first:

```bash
docker compose -f docker-compose.mqtt.yml up -d
docker compose -f docker-compose.kafka.yml up -d
docker compose -f docker-compose.amqp.yml up -d
```

Then run the subscriber and engine for each protocol, or use the
subscriber-control launcher when testing protection. After collecting logs:

```bash
python scripts/compare_protocol_results.py
```

## Notes

- Scenario1 was used only as the reference source for scripts and logic.
- Scenario1 files were not modified.
- Scenario1 keeps its own model, scalers, metadata, network, route file, and
  SUMO config.
