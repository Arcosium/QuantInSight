"""모듈 C — 통계적 차익거래(잔차 평균회귀) 실행 전략. 롱온리.

2026-07-28 리드-랙 → stat-arb 재구성. 엔진(analytics.compute_scores)이 매 분 각 종목의
s-score 스냅샷을 보낸다. s = (X-m)/σ_eq, X=시장중립 잔차의 누적, OU 평균회귀 적합.

진입 (매수, 하나라도 미충족 시 스킵 — 엔진이 후보를 이미 걸러 보낸다):
  · s < -S_ENTRY (과매도, 팩터대비 저평가)  · 평균회귀 유의(ADF) + 반감기 빠름
  · 기대 반전폭 exp_ret ≥ 왕복 마찰(0.4%)   → 가장 과매도 순으로 MAX_POSITIONS 까지
청산 (하나라도 충족 시 즉시):
  · s > -S_EXIT (평균회귀 완료)  · 진입 후 반감기 경과(타임스톱)
  · s < -S_STOP (더 깊이 하락 = 낙하칼, 손절)  · 가격 손절/이익실현(안전판)

signals.jsonl 에 BUY/SELL 을 기록한다(follower=종목코드, exp_ret=엣지게이트용) — 웹 autofolio
루프가 이걸 소비해 타임폴리오 대회 계정에 실주문한다. 동시에 로컬 페이퍼 장부(trades.csv)도 갱신.
"""
import json
import logging
from datetime import timedelta

import config
from core import crawler
from utils import market_time as mt
from utils.notify import notify

log = logging.getLogger("lag.strategy")


def latest_prices(conn, codes):
    """종목별 최신 (ts, close). 스냅샷에 없는 보유종목 청산가 확보용."""
    if not codes:
        return {}
    rows = conn.execute(
        "SELECT code, ts, close FROM bars WHERE code IN (%s) ORDER BY code, ts DESC"
        % ",".join("?" * len(codes)), list(codes))
    out = {}
    for code, ts, close in rows:
        if code not in out:                       # DESC 라 첫 행이 최신
            out[code] = (ts, close)
    return out


class Position:
    __slots__ = ("code", "qty", "entry_price", "entry_ts", "half_life", "entry_s")

    def __init__(self, code, qty, entry_price, entry_ts, half_life, entry_s):
        self.code, self.qty = code, qty
        self.entry_price, self.entry_ts = entry_price, entry_ts
        self.half_life, self.entry_s = half_life, entry_s

    def to_dict(self):
        return {"code": self.code, "qty": self.qty, "entry_price": self.entry_price,
                "entry_ts": self.entry_ts.isoformat(),
                "half_life": self.half_life, "entry_s": self.entry_s}


class StatArbStrategy:
    def __init__(self, broker, conn=None):
        self.broker = broker
        self.conn = conn or crawler.open_db()
        self.snapshot = {}             # 엔진이 보낸 최신 스냅샷 (compute_scores 반환)
        self.scores = {}               # {code: {s,half_life,exp_ret,price,...}}
        self.positions = {}            # code -> Position
        self.entered_minute = {}       # code -> 마지막 진입 분 ts (동분 재진입 방지)
        self.short_open = set()        # (가상) 숏 신호 진행중 코드 — 에피소드당 SHORT_SIGNAL 1회

    # ── 스냅샷 갱신 (Engine 큐에서) ─────────────────────────────
    def update_snapshot(self, out):
        self.snapshot = out or {}
        self.scores = self.snapshot.get("scores", {}) or {}
        log.info("s-score 갱신: %d종목 · 적격 %d · 매수후보 %d",
                 self.snapshot.get("n_codes", 0), self.snapshot.get("n_eligible", 0),
                 self.snapshot.get("n_candidates", 0))

    # ── 진입 (엔진이 이미 sig_ok·s<-S_ENTRY·exp_ret≥마찰 로 거른 후보) ──
    def check_entries(self):
        room = config.MAX_POSITIONS - len(self.positions)
        if room <= 0:
            return
        for c in self.snapshot.get("candidates", []):
            if room <= 0:
                break
            code, sc = c["code"], self.scores.get(c["code"], c)
            if code in self.positions:
                continue
            ts = sc.get("ts")
            if self.entered_minute.get(code) == ts:
                continue
            self._enter(code, sc)
            room -= 1

    def _enter(self, code, sc):
        price = float(sc.get("price") or 0.0)
        if price <= 0:
            return
        qty = max(1, int(config.ORDER_NOTIONAL // price))
        fill = self.broker.buy_market(code, qty, price,
                                      reason=f"s={sc['s']:+.2f} hl={sc['half_life']:.0f}m exp={sc['exp_ret']*100:.2f}%")
        self.positions[code] = Position(code, qty, fill, mt.now_kst(),
                                        sc.get("half_life", 0.0), sc.get("s", 0.0))
        self.entered_minute[code] = sc.get("ts")
        self._persist()
        self._emit("BUY", code, fill, {"qty": qty, "s": sc.get("s"), "exp_ret": sc.get("exp_ret"),
                                       "half_life": sc.get("half_life")})
        notify(f"진입: {code} x{qty} @ {fill:.0f} (s={sc['s']:+.2f}, 반감기 {sc['half_life']:.0f}분, "
               f"기대반전 {sc['exp_ret']*100:.2f}%)", level=logging.INFO)

    # ── 청산 ─────────────────────────────────────────────────────
    def check_exits(self, px_map):
        now = mt.now_kst()
        for code in list(self.positions):
            pos = self.positions[code]
            sc = self.scores.get(code)
            px = float((sc or {}).get("price") or 0.0)
            if px <= 0:                            # 스냅샷에 없으면 최신 분봉가로
                pr = px_map.get(code)
                px = float(pr[1]) if pr and pr[1] else 0.0
            if px <= 0:
                continue
            pnl = px / pos.entry_price - 1.0
            s = float(sc["s"]) if sc and "s" in sc else None
            held_min = (now - pos.entry_ts).total_seconds() / 60.0
            reason = None
            if (s is not None and s > -config.STATARB_S_EXIT
                    and held_min >= config.STATARB_MIN_HOLD_MIN):
                # 잔차가 되돌아왔어도 최소 홀딩 전엔 청산 안 함 — 가격 무변 즉시청산(−0.1% churn) 방지.
                reason = f"REVERT s={s:+.2f} {pnl*100:+.2f}%"
            elif s is not None and s < -config.STATARB_S_STOP:
                reason = f"SSTOP s={s:+.2f} {pnl*100:+.2f}%"   # 낙하칼: 더 깊이 하락
            elif pnl <= config.STOP_LOSS:
                reason = f"SL {pnl*100:+.2f}%"
            elif pnl >= config.TAKE_PROFIT:
                reason = f"TP {pnl*100:+.2f}%"
            elif pos.half_life > 0 and held_min >= pos.half_life:
                reason = f"TIMEOUT hl={pos.half_life:.0f}m {pnl*100:+.2f}%"
            if reason:
                self._exit(pos, px, pnl, reason)

    def _exit(self, pos, price, pnl, reason):
        fill = self.broker.sell_market(pos.code, pos.qty, price, reason=reason)
        pnl = fill / pos.entry_price - 1.0
        del self.positions[pos.code]
        self._persist()
        self._emit("SELL", pos.code, fill, {"qty": pos.qty, "pnl": round(pnl, 4), "reason": reason})
        notify(f"청산: {pos.code} @ {fill:.0f} ({reason}) PnL {pnl*100:+.2f}%", level=logging.INFO)

    # ── signals.jsonl (웹 autofolio 루프가 소비 → 타임폴리오 실주문) ──
    def _emit(self, kind, code, price, extra):
        rec = {"ts": mt.now_kst().isoformat(timespec="seconds"), "kind": kind,
               "follower": code, "price": round(float(price), 1)}
        rec.update({k: v for k, v in extra.items() if v is not None})
        try:
            with open(config.SIGNALS_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            log.warning("signals.jsonl 기록 실패: %s %s", kind, code)

    def _persist(self):
        try:
            config.POSITIONS_JSON.write_text(json.dumps(
                [p.to_dict() for p in self.positions.values()], ensure_ascii=False))
        except OSError:
            pass

    # ── (가상) 숏 신호 — 롱+숏 대비 표시용(실거래 없음) ───────────
    def emit_shorts(self):
        """고평가(s>+진입) 종목에 SHORT_SIGNAL 을 에피소드당 1회 기록. 현물 롱온리라 실제 숏은
        없지만, 대시보드 '롱+숏' 모드가 이 신호로 '숏까지 했다면'의 가상 손익을 보여준다
        (backtest.resolve_signals 가 lag(=반감기)분 뒤 분봉으로 청산가 시뮬). s 가 청산문턱
        (+S_EXIT) 아래로 회귀하면 에피소드 종료로 보고 다시 열 수 있게 한다."""
        for c in self.snapshot.get("shorts", []):
            code = c["code"]
            if code in self.short_open or code in self.positions:
                continue
            self.short_open.add(code)
            self._emit("SHORT_SIGNAL", code, c["price"],
                       {"s": c["s"], "half_life": c["half_life"],
                        "lag": max(1, round(c.get("half_life") or 1))})
        for code in list(self.short_open):        # 고평가 해소(회귀) → 에피소드 종료
            sc = self.scores.get(code)
            if sc is None or sc.get("s", 0.0) <= config.STATARB_S_EXIT:
                self.short_open.discard(code)

    # ── 1틱 ──────────────────────────────────────────────────────
    def tick(self):
        held = list(self.positions)
        px_map = latest_prices(self.conn, held) if held else {}
        self.check_exits(px_map)       # 청산 우선
        self.check_entries()
        self.emit_shorts()             # 가상 숏(표시용)


def trading_loop(map_queue, stop_event, db_path=None):
    """Execution 프로세스 본체: 스냅샷 수신 + STRATEGY_TICK_SEC 마다 tick."""
    from utils.logging_setup import setup
    from core.executor import make_broker
    setup("trader")
    conn = crawler.open_db(db_path)
    strat = StatArbStrategy(make_broker(), conn)
    while not stop_event.is_set():
        try:
            while not map_queue.empty():
                strat.update_snapshot(map_queue.get_nowait())
            if mt.in_crawl_session():
                strat.tick()
        except Exception as e:          # fail-safe
            log.error("trading tick 예외: %s", e, exc_info=True)
        if stop_event.wait(config.STRATEGY_TICK_SEC):
            break
    conn.close()
