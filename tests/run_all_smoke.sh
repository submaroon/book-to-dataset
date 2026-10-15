#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
python3 tools/mock_llm_server.py --port 8999 > tests/mock_all.log 2>&1 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null' EXIT
sleep 1
python3 tools/run_all.py --config tests/config.test.yaml "$@"
