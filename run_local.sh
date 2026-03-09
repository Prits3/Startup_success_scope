#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8620}"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

pkill -f "streamlit run app.py" >/dev/null 2>&1 || true
if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
  lsof -ti tcp:"$PORT" | xargs kill -9 >/dev/null 2>&1 || true
fi

cd "$APP_DIR"
nohup streamlit run app.py \
  --server.address localhost \
  --browser.serverAddress localhost \
  --server.port "$PORT" \
  --server.headless true \
  --server.fileWatcherType none \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  > /tmp/venturescope.log 2>&1 &

echo "Starting VentureScope AI on http://localhost:${PORT} ..."
for _ in {1..20}; do
  if curl -sS "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if curl -sS "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1; then
  echo "Server healthy. Opening browser..."
  open -na "Safari" "http://localhost:${PORT}" >/dev/null 2>&1 || true
  open -na "Google Chrome" "http://localhost:${PORT}" >/dev/null 2>&1 || true
  echo "Done. URL: http://localhost:${PORT}"
else
  echo "Server did not become healthy. Check logs: tail -n 120 /tmp/venturescope.log"
  exit 1
fi
