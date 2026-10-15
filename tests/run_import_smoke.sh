#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
python3 tools/mock_llm_server.py --port 8999 > tests/mock_imp.log 2>&1 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null' EXIT
sleep 1
for step in "$@"; do
  echo ""; echo "############ $step ############"
  python3 "$step" --config tests/config.test.import.yaml || { echo "!!!! $step failed"; exit 1; }
done
