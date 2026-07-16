#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python}"

exec "$PYTHON_BIN" scripts/amqp_alert_subscriber.py "$@"
