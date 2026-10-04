#!/bin/bash
# Arquant v1.0 - Start Script
# 사장 피드백 2026-05-16: 포트 8500 bind race(Errno 98) 하드닝 —
#   기존 인스턴스를 lsof + pkill 양쪽으로 확실히 정리하고, 포트가 '실제로'
#   비워질 때까지 대기한 뒤에만 새 uvicorn 을 띄운다. 마지막에 헬스 확인.

APP_DIR="/home/arcosium/projects/QuantInSight"
LOG_FILE="/tmp/arquant.log"
PORT=8500

port_busy() { lsof -ti:$PORT >/dev/null 2>&1; }

echo "🛑 Stopping existing services..."
# lsof 기반 + 프로세스명 기반 양쪽으로 — 다른 python 경로/좀비 인스턴스까지 정리
pkill -9 -f "uvicorn server.app" 2>/dev/null
lsof -ti:$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null

# 포트가 실제로 해제될 때까지 대기 (최대 ~15s). 안 풀리면 매 회차 재차 kill.
freed=0
for i in $(seq 1 30); do
  if ! port_busy; then freed=1; echo "   port $PORT free (after ${i} check(s))"; break; fi
  lsof -ti:$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null
  sleep 0.5
done
if [ "$freed" -ne 1 ]; then
  echo "❌ port $PORT 가 여전히 점유 중 — 기동 중단 (수동 확인 필요: lsof -i:$PORT)"
  exit 1
fi

echo "🧹 Cleaning up pycache..."
find "$APP_DIR" -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "🚀 Starting Arquant Web Server..."
cd "$APP_DIR" || exit 1
nohup python3.12 -m uvicorn server.app:app --host 127.0.0.1 --port $PORT > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# 터널은 systemd(cloudflared.service)가 관리한다 — 여기선 손대지 않는다.

# 기동 헬스 확인 (최대 ~30s) — 새 인스턴스가 실제로 bind/응답하는지 검증
ok=0
for i in $(seq 1 30); do
  if curl -fsS -m3 "http://localhost:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  sleep 1
done

echo "✅ All services started in background!"
echo "Server PID: $SERVER_PID"
if [ "$ok" -eq 1 ]; then
  echo "🟢 Health check OK (http://localhost:$PORT/health)"
else
  echo "⚠️  Health check 실패 — 로그 확인: tail -n 40 $LOG_FILE"
fi
echo "To view server logs: tail -f $LOG_FILE"
