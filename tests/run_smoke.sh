#!/usr/bin/env bash
# Start the mock server, run the requested steps against it, stop it.
set -u
cd "$(dirname "$0")/.."
python3 tools/mock_llm_server.py --port 8999 > tests/mock.log 2>&1 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null' EXIT
for i in $(seq 1 40); do
  curl -s -o /dev/null http://127.0.0.1:8999/ 2>/dev/null && break
  python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(0.2)
sys.exit(0 if s.connect_ex(('127.0.0.1',8999))==0 else 1)" && break
  sleep 0.2
done
for step in "$@"; do
  echo ""
  echo "############ $step ############"
  python3 "$step" --config tests/config.test.yaml
  code=$?
  if [ $code -ne 0 ]; then echo "!!!! $step exited with $code"; exit $code; fi
done
