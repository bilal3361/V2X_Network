#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python}"
SUMO_CONFIG="${SCENARIO1_SUMO_CONFIG:-osm.sumocfg}"
SUMO_BINARY="${SCENARIO1_SUMO_BINARY:-sumo-gui}"
REALTIME_FACTOR="${SCENARIO1_REALTIME_FACTOR:-1}"
TRACI_PORT="${SCENARIO1_TRACI_PORT:-8873}"
export SCENARIO1_TRACI_PORT="$TRACI_PORT"
export PYTHONUNBUFFERED=1

ENGINE_PID=""
CONTROLLER_PID=""

cleanup() {
  if [[ -n "$CONTROLLER_PID" ]] && kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    kill "$CONTROLLER_PID" 2>/dev/null || true
  fi
  if [[ -n "$ENGINE_PID" ]] && kill -0 "$ENGINE_PID" 2>/dev/null; then
    kill "$ENGINE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PYTHON_BIN" scripts/amqp_alert_subscriber_traci_controller.py \
  --traci-port "$TRACI_PORT" \
  --traci-client-order 2 \
  "$@" &
CONTROLLER_PID=$!

sleep 1

"$PYTHON_BIN" scripts/amqp_alert_engine.py \
  --sumo-binary "$SUMO_BINARY" \
  --sumo-config "$SUMO_CONFIG" \
  --vehicle-groups targeted \
  --min-risk LOW \
  --publish-predictions \
  --prediction-interval-steps 1 \
  --real-time \
  --realtime-factor "$REALTIME_FACTOR" \
  --traci-num-clients 2 \
  --traci-client-order 1 \
  "$@" &
ENGINE_PID=$!

wait "$ENGINE_PID"
ENGINE_STATUS=$?
wait "$CONTROLLER_PID"
CONTROLLER_STATUS=$?

exit $(( ENGINE_STATUS != 0 ? ENGINE_STATUS : CONTROLLER_STATUS ))
