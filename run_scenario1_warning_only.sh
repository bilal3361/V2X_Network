#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python}"
SUMO_CONFIG="${SCENARIO1_SUMO_CONFIG:-osm.sumocfg}"
REALTIME_FACTOR="${SCENARIO1_REALTIME_FACTOR:-1}"

exec "$PYTHON_BIN" scripts/mqtt_alert_engine.py \
  --sumo-binary sumo-gui \
  --sumo-config "$SUMO_CONFIG" \
  --vehicle-groups targeted \
  --min-risk LOW \
  --publish-predictions \
  --prediction-interval-steps 1 \
  --real-time \
  --realtime-factor "$REALTIME_FACTOR" \
  "$@"
