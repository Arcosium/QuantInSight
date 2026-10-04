#!/bin/bash
# Arquant v1.0 — watchdog. Keeps the uvicorn server AND the Cloudflare tunnel alive.
# Run once in the background:  nohup /home/arcosium/projects/QuantInSight/supervise.sh > /tmp/arquant_supervise.log 2>&1 &
# (or add to crontab:  @reboot /home/arcosium/projects/QuantInSight/supervise.sh >> /tmp/arquant_supervise.log 2>&1 )
set -u
APP_DIR="/home/arcosium/projects/QuantInSight"
PORT=8500
APP_LOG="/tmp/arquant.log"
# 터널은 이제 systemd(cloudflared.service)가 관리한다 — 여기선 손대지 않는다.
PY="python3.12"

start_server() {
  echo "$(date -u +%FT%TZ) [supervise] starting uvicorn on :$PORT"
  ( cd "$APP_DIR" && find . -name __pycache__ -exec rm -rf {} + 2>/dev/null;
    nohup $PY -m uvicorn server.app:app --host 127.0.0.1 --port $PORT >> "$APP_LOG" 2>&1 & )
}

server_up() { curl -fsS -m4 "http://localhost:$PORT/health" >/dev/null 2>&1; }

while true; do
  server_up || { echo "$(date -u +%FT%TZ) [supervise] server DOWN → restart"; start_server; sleep 5; }
  sleep 20
done
