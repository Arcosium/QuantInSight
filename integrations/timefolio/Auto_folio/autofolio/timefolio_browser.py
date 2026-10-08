from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

from .errors import ContestClosedError   # 재노출 — 기존 import 경로 유지

KST = ZoneInfo("Asia/Seoul")

# 주문 제출 후 뜨는 사이트 오류 팝업. 예: "[TMS] 오류\n\n전량 청산 주문 작동 시 추가 청산 불가\n\n확인"
# (2026-07-13 실측). 이 팝업이 떠도 본문에 '체결/완료' 같은 단어가 항상 있어서 텍스트 휴리스틱이
# accepted 로 오판했다 — 성공 판정보다 먼저, 명시적으로 거부 처리해야 한다.
TMS_ERROR_RE = re.compile(r"\[TMS\]\s*오류\s*\n+\s*([^\n]{2,200})")

# 대회 섹터코드(섹터표/드롭다운 접미사) — ticker 섹터 파싱용.
SECTOR_CODES = {"En", "Ma", "In", "CD", "CS", "He", "Fi", "IT", "Co", "Ut", "Re"}
# 남은 섹터 편입 여유(추가 편입 가능 비중, %)가 이 값 이하이면 그 섹터 신규 매수를 막는다
# (사장 지시 2026-07-08: 섹터 10% 중 9%+ 차서 여유 1% 이하면 매수 금지 — 상한이 더 높은 섹터도 동일 룰).
MIN_SECTOR_ROOM_PCT = 1.0
# 주문 제출 후 체결/접수 확인 재시도 (2026-07-22: 1회 스크랩 레이스로 체결을 '미접수' 오판)
_CONFIRM_TRIES = 5
_CONFIRM_WAIT_MS = 1500


@dataclass
class TimefolioCredentials:
    username: str
    password: str


class TimefolioBrowser:
    """Playwright adapter for contest.timefolio.net."""

    def __init__(self, *, headless: bool = True, live_enabled: bool | None = None):
        self.headless = headless
        self.live_enabled = bool(live_enabled)
        self._pw = None
        self._browser = None
        self._context = None
        self.page: Page | None = None

    def __enter__(self) -> "TimefolioBrowser":
        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self.headless, args=["--no-sandbox"])
            self._context = self._browser.new_context(locale="ko-KR", timezone_id="Asia/Seoul")
            self.page = self._context.new_page()
            return self
        except Exception:
            # launch 단계에서 브라우저 실행 파일 누락 등이 나도 sync_playwright 드라이버를
            # 반드시 멈춘다. 이 정리가 없으면 같은 워커 스레드의 다음 재시도가
            # "Sync API inside asyncio loop"라는 엉뚱한 2차 오류로 바뀐다.
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        for closer in (self._context, self._browser):
            try:
                if closer:
                    closer.close()
            except Exception:  # noqa: BLE001 — 종료 실패가 호출자를 죽이면 안 된다
                pass
        try:
            if self._pw:
                self._pw.stop()
        except Exception:  # noqa: BLE001
            pass
        self._context = self._browser = self._pw = None
        self.page = None

    # ── 세션 재사용(지연 최소화) ────────────────────────────────────
    # 주문 1건마다 브라우저를 새로 띄우고 로그인하면 실측 8~12초가 그냥 날아간다.
    # 리드-랙 신호는 1~5분 보유 단타라 그 지연이 곧 손실이므로, 세션을 살려두고 재사용한다.
    # Playwright sync API 객체는 스레드 친화적이므로 항상 같은 스레드에서만 써야 한다.
    open = __enter__

    def close(self) -> None:
        self.__exit__(None, None, None)

    def alive(self) -> bool:
        return self.page is not None and not self.page.is_closed()

    def logged_in(self) -> bool:
        try:
            text = self._body_text(self._require_page())
        except Exception:  # noqa: BLE001
            return False
        return "logout" in text.lower() and "주문" in text

    def ensure_logged_in(self, creds: "TimefolioCredentials") -> bool:
        """세션이 살아 있으면 로그인 생략. 만료/로그아웃 시에만 재로그인."""
        if self.alive() and self.logged_in():
            return True
        return bool(self.login(creds).get("logged_in"))

    def refresh(self) -> None:
        """대시보드를 다시 읽어 보유/현재가/NAV 를 최신화 (로그인 유지)."""
        page = self._require_page()
        page.reload(wait_until="domcontentloaded", timeout=20000)
        try:
            page.wait_for_load_state("networkidle", timeout=6000)
        except PlaywrightTimeoutError:
            pass
        self._wait_nav(page)

    def _wait_nav(self, page: Page) -> None:
        """NAV와 보유표가 모두 렌더될 때까지 기다린다.

        NAV가 보유표보다 먼저 뜨는 SPA라 NAV만 보고 곧바로 긁으면 실제 보유 11종목을 빈 목록으로
        덮어쓴다. 정상적인 빈 보유도 구분할 수 있도록 보유표의 종목행 또는 Totals 행까지 확인한다.
        """
        try:
            page.wait_for_function(
                r"""() => {
                    const body = document.body.innerText || '';
                    const nav = body.match(/NAV\s*([0-9,]+(?:\.[0-9]+)?)/);
                    if (!nav || parseFloat(String(nav[1]).replace(/,/g, '')) < 100) return false;
                    const norm = s => String(s || '').replace(/\s+/g, ' ').trim();
                    const table = [...document.querySelectorAll('table')]
                        .find(t => /잔고\s*\/\s*비중\s*\/\s*평가액/.test(norm(t.innerText)));
                    if (!table) return false;
                    const rows = [...table.querySelectorAll('tr')].map(tr => norm(tr.innerText));
                    return rows.some(text => /Totals\s*:/.test(text));
                }""",
                timeout=12000)
            page.wait_for_timeout(400)
        except PlaywrightTimeoutError:
            pass

    def login(self, creds: TimefolioCredentials) -> dict[str, Any]:
        # 로그인 플레이키 하드닝: SPA 로그인 폼이 실제로 렌더될 때까지 명시 대기하고,
        # networkidle 조기 종료·콜드스타트 레이스로 실패하면 최대 3회 재시도한다.
        # (재시도는 로그인에 한정 — 주문 로직과 무관하므로 중복 주문 위험 없음.)
        page = self._require_page()
        last: dict[str, Any] = {"logged_in": False, "url": page.url}
        for attempt in range(3):
            try:
                page.goto("https://contest.timefolio.net/", wait_until="domcontentloaded", timeout=30000)
                try:
                    page.wait_for_load_state("networkidle", timeout=12000)
                except PlaywrightTimeoutError:
                    pass
                email = page.locator("#email, input[type=email]").first
                email.wait_for(state="visible", timeout=15000)  # 폼 렌더 대기 (핵심 수정)
                password = page.locator("#password, input[type=password]").first
                password.wait_for(state="visible", timeout=5000)
                email.fill(creds.username)
                password.fill(creds.password)
                page.get_by_role("button", name=re.compile("submit|로그인|login", re.I)).click(timeout=5000)
                # networkidle 만으론 인증 완료를 못 보장 — 로그인 성공 지표(logout+주문)까지 대기
                try:
                    page.wait_for_function(
                        "() => { const t = document.body.innerText || ''; "
                        "return /logout/i.test(t) && t.includes('주문'); }",
                        timeout=15000)
                except PlaywrightTimeoutError:
                    pass
                self._wait_nav(page)
                last = self.status()
                if last.get("logged_in"):
                    return last
            except PlaywrightTimeoutError as exc:
                last = {"logged_in": False, "url": page.url, "error": f"timeout: {exc}"}
            except Exception as exc:  # noqa: BLE001 — 다음 시도로 폴백
                last = {"logged_in": False, "url": page.url, "error": str(exc)}
            page.wait_for_timeout(1500)  # 재시도 전 짧은 백오프
        return last

    def status(self) -> dict[str, Any]:
        page = self._require_page()
        text = page.locator("body").inner_text(timeout=5000)
        return {
            "url": page.url,
            "title": page.title(),
            "logged_in": "logout" in text.lower() and "주문" in text,
            "has_order_screen": "신규 주문" in text or "주문" in text,
            "webdriver": page.evaluate("navigator.webdriver"),
            "summary": self.scrape_summary(),
        }

    def scrape_summary(self) -> dict[str, Any]:
        page = self._require_page()
        text = self._body_text(page)
        return page.evaluate(
            r"""(bodyText) => {
                const norm = s => String(s || '').replace(/\s+/g, ' ').trim();
                const tables = [...document.querySelectorAll('table')].map((table, ti) => ({
                  index: ti,
                  text: norm(table.innerText),
                  rows: [...table.querySelectorAll('tr')].map(tr => [...tr.children].map(td => norm(td.innerText || td.textContent)))
                })).filter(x => x.text);
                const allText = norm(bodyText || document.body.innerText || '');
                const toNum = value => {
                  const s = String(value || '').replace(/,/g, '').replace(/[^0-9.\-]/g, '');
                  const n = Number(s);
                  return Number.isFinite(n) ? n : 0;
                };
                const turnover = /금주\s*([0-9.]+)%/.exec(allText);
                const nav = /NAV\s*([0-9,]+(?:\.[0-9]+)?)/.exec(allText);
                const navNumber = nav ? toNum(nav[1]) : 0;
                const holdingsTable = tables.find(t => /잔고\s*\/\s*비중\s*\/\s*평가액/.test(t.text));
                const positions = [];
                if (holdingsTable) {
                  for (const row of holdingsTable.rows) {
                    const code = String(row[0] || '').replace(/^A/, '');
                    if (!/^[0-9A-Z]{6}$/.test(code)) continue;
                    positions.push({
                      ticker: code,
                      name: row[1] || code,
                      last_price: toNum(row[2]),
                      day_pct: toNum(row[3]),
                      qty: Math.trunc(toNum(row[4])),
                      weight_pct: toNum(row[5]),
                      value_krw: toNum(row[6]) * 10000,
                      avg_price: toNum(row[7]),
                      pnl_pct: toNum(row[8]),
                      pnl_krw: toNum(row[9]) * 10000,
                    });
                  }
                }
                const hasNoHoldingTotals = positions.length === 0 && /보유\s*잔고[\s\S]*Totals:\s*0\.0\s+0\s+0/.test(allText);
                return {
                  text: allText.slice(0, 5000),
                  nav_text: nav ? nav[1] : '',
                  total_eval: navNumber > 0 ? navNumber * 1000000 : null,
                  weekly_turnover_pct: turnover ? Number(turnover[1]) : null,
                  no_holdings_visible: hasNoHoldingTotals,
                  positions,
                  tables,
                };
            }""",
            text,
        )

    # ── 주문 상태 권위 판정/미체결 관리 (2026-07-13 매도 불능 사건 후 추가) ──────────

    def detect_and_close_tms_error(self, page: Page | None = None) -> str | None:
        """[TMS] 오류 팝업을 감지해 메시지를 돌려주고 '확인'으로 닫는다. 없으면 None."""
        page = page or self._require_page()
        visible_text = ""
        try:
            dialogs = page.locator('[role="dialog"]:visible')
            visible_text = "\n".join(dialogs.nth(i).inner_text(timeout=1000)
                                     for i in range(min(dialogs.count(), 4)))
        except Exception:
            pass
        m = TMS_ERROR_RE.search(visible_text) or TMS_ERROR_RE.search(self._body_text(page))
        if not m:
            return None
        msg = m.group(1).strip()
        try:
            self._click_first_visible(page.get_by_role("button", name=re.compile(r"^확인$")), timeout=1000)
            page.wait_for_timeout(300)
        except Exception:  # noqa: BLE001 — 못 닫아도 감지 결과는 유효
            pass
        return msg

    def ledger_orders(self) -> list[dict[str, Any]]:
        """'주문 내역' 원장(입력T|종목코드|종목명|매수도|비중|전량|지정가|주문T|취소T|체결)을 파싱한다.

        진단·감사용이다. **주문 흐름에서는 쓰지 않는다** — 이 표는 하단 탭(가이드라인 위반/섹터
        비중/주문 내역…) 중 하나라 열려면 탭을 클릭해야 하고, 그 클릭이 화면을 전환시켜 이어지는
        주문의 '신규 주문' 버튼을 못 찾게 만든다(2026-07-13 실측: 첫 종목 이후 전부 실패).
        주문 접수 판정은 탭을 건드리지 않는 list_working_orders()/보유 변화로 한다.
        """
        page = self._require_page()
        for attempt in range(2):
            rows = page.evaluate(r"""() => {
                const norm = s => String(s || '').replace(/\s+/g, ' ').trim();
                for (const t of document.querySelectorAll('table')) {
                    const head = norm((t.querySelector('tr') || {}).innerText || '');
                    if (!(/입력T/.test(head) && /매수도/.test(head))) continue;
                    return [...t.querySelectorAll('tr')].slice(1).map(tr => {
                        const c = [...tr.children].map(td => norm(td.innerText));
                        return c.length >= 8 ? {
                            entered_at: c[0], ticker: String(c[1] || '').replace(/^A/, ''),
                            name: c[2], side: c[3], weight: c[4], full: c[5],
                            price_type: c[6], ordered_at: c[7], canceled_at: c[8] || '',
                        } : null;
                    }).filter(Boolean);
                }
                return null;
            }""")
            if rows is not None:
                return rows
            if attempt == 0:
                try:
                    self._click_first_visible(page.get_by_text(re.compile(r"주문\s*내역")), timeout=1500)
                    page.wait_for_timeout(900)
                except Exception:  # noqa: BLE001
                    pass
        return []

    def list_working_orders(self) -> list[dict[str, Any]]:
        """주문표(코드|…|주문|미체결|타겟)에서 미체결(작동 중) 주문이 걸린 종목 목록.

        미체결 칼럼의 숫자 버튼(예: '-7.41')이 곧 '작동 중 주문 있음' 표식이자 관리 다이얼로그
        오프너다. 이런 주문이 살아 있으면 같은 종목 신규 청산이 TMS 오류로 전부 거부된다.
        """
        page = self._require_page()
        return page.evaluate(r"""() => {
            const norm = s => String(s || '').replace(/\s+/g, ' ').trim();
            const out = [];
            for (const t of document.querySelectorAll('table')) {
                // 주문표는 헤더가 2행(그룹행+칼럼행)이라 첫 tr 만 보면 '미체결/타겟'을 놓친다
                // — 헤더 2행을 합쳐 판별한다(잔고표/원장과 구분되는 칼럼 조합).
                const trs = t.querySelectorAll('tr');
                const head = norm([trs[0], trs[1]].map(r => (r ? r.innerText : '')).join(' '));
                if (!(/미체결/.test(head) && /타겟/.test(head))) continue;
                for (const tr of t.querySelectorAll('tr')) {
                    const cells = [...tr.children].map(td => norm(td.innerText));
                    const code = cells.find(c => /^A\d{6}$/.test(c));
                    if (!code) continue;
                    const btn = [...tr.querySelectorAll('button')]
                        .find(b => /^-?[0-9.]+$/.test(norm(b.innerText)));
                    if (!btn) continue;
                    out.push({ticker: code.slice(1), pending_weight: Number(norm(btn.innerText)) || 0});
                }
                break;
            }
            return out;
        }""")

    # 아직 살아 있는(정지시켜야 할) 주문 상태. 사이트 실측값:
    #   '작동'                    — 호가에 걸려 체결을 기다리는 중
    #   '과주문 후 대기 (11:13)'  — 사이트가 물량 과다로 보류하고 재시도를 예약한 상태
    # 둘 다 살아 있어서, 남겨두면 같은 종목의 신규 청산이 TMS 오류로 거부된다.
    LIVE_ORDER_STATE_RE = re.compile(r"작동|대기")

    def cancel_working_orders(self, ticker: str, *, min_age_min: float = 0.0) -> dict[str, Any]:
        """해당 종목의 살아 있는(미체결) 주문을 '선택 주문 정지'로 중지한다.

        플로우(2026-07-13 실측): 주문표 미체결 버튼 → 종목 주문 다이얼로그(행 예:
        '매도 전량청산@미정 (상대5호가) LMT | 11:08 | 과주문 후 대기 (11:13)') → 행(그리드 셀)
        클릭으로 선택 → 그때 활성화되는 '선택 주문 정지' → 닫기.
        min_age_min 이 주어지면 그보다 오래된(시간 칼럼 기준) 주문만 중지한다.
        """
        page = self._require_page()
        code = str(ticker or "").strip().zfill(6)
        opener = page.locator("table tr", has_text=f"A{code}").locator(
            "button", has_text=re.compile(r"^\s*-?[0-9.]+\s*$"))
        if not self._click_first_visible(opener, timeout=2500):
            return {"ok": False, "stopped": 0, "result": "미체결 버튼을 찾지 못했습니다"}
        page.wait_for_timeout(900)
        dialog = page.locator('[role="dialog"]').last
        try:
            dialog.wait_for(state="visible", timeout=5000)
        except PlaywrightTimeoutError:
            return {"ok": False, "stopped": 0, "result": "주문 관리 다이얼로그가 뜨지 않았습니다"}
        now = datetime.now(KST)
        selected = 0
        rows = dialog.locator("table tr")
        for i in range(min(rows.count(), 30)):
            tr = rows.nth(i)
            try:
                txt = tr.inner_text(timeout=1000)
            except Exception:  # noqa: BLE001
                continue
            if not self.LIVE_ORDER_STATE_RE.search(txt):
                continue
            if min_age_min > 0:
                m = re.search(r"\b(\d{1,2}):(\d{2})\b", txt)   # 첫 시각 = 주문 입력 시각
                if m:
                    ts = now.replace(hour=int(m.group(1)), minute=int(m.group(2)),
                                     second=0, microsecond=0)
                    if ts > now:                      # 자정 경계
                        ts -= timedelta(days=1)
                    if (now - ts) < timedelta(minutes=min_age_min):
                        continue
            # 행 선택 = 그리드 셀 클릭 (행에는 체크박스도 버튼도 없다). 선택돼야 '선택 주문 정지'가
            # 활성화된다(그 전엔 disabled).
            try:
                tr.locator("td").first.click(timeout=1500)
                selected += 1
                page.wait_for_timeout(250)
            except Exception:  # noqa: BLE001
                continue
        stopped = 0
        if selected:
            stop_btn = dialog.get_by_role("button", name=re.compile(r"선택\s*주문\s*정지")).first
            try:
                if stop_btn.is_enabled(timeout=1500):
                    stop_btn.click(timeout=2000)
                    page.wait_for_timeout(800)
                    self._confirm_dialog(page)
                    stopped = selected
            except Exception:  # noqa: BLE001
                pass
        self._close_ticket(page)
        page.wait_for_timeout(600)
        still = [w for w in self.list_working_orders() if w.get("ticker") == code]
        return {"ok": stopped > 0 and not still, "stopped": stopped,
                "still_working": bool(still),
                "result": f"살아있는 주문 {selected}건 선택, {stopped}건 정지 요청"}

    def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        """Place a Timefolio contest order using its visible weight-based ticket.

        주문 폼의 '비중%' 는 **목표 비중이 아니라 이번에 매매할 비중**이다 (사이트 '주문 내역' 실측,
        2026-07-09: `매도 | 비중 9 | 전량 T | 상대1호가` → 보유 9% 전량 청산). 따라서 매도 비중은
        현재 보유 비중을 그대로 넣으면 전량 청산이고, 0 을 넣으면 아무 것도 팔리지 않는다.
        """
        page = self._require_page()
        ticker = str(order.get("ticker") or "").strip().zfill(6)
        side = str(order.get("side") or "").lower()
        qty = int(order.get("qty") or 0)
        weight_pct = self._coerce_weight(order)
        # 상대호가 틱(1~10). 기본 1이지만, 급락 종목의 손절처럼 가격이 도망가는 상황에서는
        # 재시도마다 더 깊은 틱으로 쫓아가야 체결된다(2026-07-13: 상대1호가 매도가 1시간 미체결).
        opp_tick = max(1, min(10, int(order.get("opp_tick") or 1)))
        submit = bool(order.get("submit", True))
        # 매수·매도 모두 비중>0 이어야 한다 (0 = 매매 안 함).
        if not ticker.strip("0") or side not in ("buy", "sell") or qty <= 0 or weight_pct <= 0:
            return {"accepted": False, "filled": False, "result": "invalid Timefolio order payload"}
        if submit and not self.live_enabled:
            return {
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "weight_pct": weight_pct,
                "accepted": False,
                "filled": False,
                "result": "dry-run: Timefolio live order disabled",
            }

        before_summary = self.scrape_summary()
        before_text = self._body_text(page)
        # 섹터 여유 게이트(매수 한정): 주문 폼을 열기 전 섹터표에서 '추가 편입 가능 비중'을 읽어둔다.
        # (사이트 권위값 그대로 — 매도 미체결은 선반영하지 않음.) 실패 시 {}=게이트 미적용(fail-open).
        sector_rooms = self._scrape_sector_rooms(page) if side == "buy" else {}
        self._open_order_ticket(page)
        selected = self._select_ticker(page, ticker)
        sector_clamped = None
        if side == "buy":
            sector = (selected or {}).get("sector")
            room = sector_rooms.get(sector) if sector else None
            if not sector or room is None:
                self._close_ticket(page)
                msg = "섹터 한도를 확인하지 못해 매수 제출을 중단했습니다"
                return {
                    "ticker": ticker, "side": side, "qty": qty, "weight_pct": weight_pct,
                    "accepted": False, "filled": False, "pending": False,
                    "rejected_reason": "sector_check_unavailable", "result": msg,
                    "summary": before_summary,
                }
            if room is not None and room <= MIN_SECTOR_ROOM_PCT:
                self._close_ticket(page)
                msg = (f"섹터 편입 여유 부족: {sector} 추가편입가능 {room:.1f}% ≤ {MIN_SECTOR_ROOM_PCT:.0f}% — 매수 스킵")
                return {
                    "ticker": ticker, "side": side, "qty": qty, "weight_pct": weight_pct,
                    "accepted": False, "filled": False, "pending": False,
                    "rejected_reason": "sector_full", "sector": sector, "sector_room_pct": room,
                    "result": msg, "summary": before_summary,
                }
            # 사장 지시 2026-07-29: 여유가 남아 있어도 **주문 비중이 그 여유보다 크면 한도를 넘는다**.
            # 종전엔 여유>1% 이면 전량 통과라 (여유 3% · 주문 5%) 가 그대로 나가 섹터 한도를 깼다.
            # 사이트 권위값(추가 편입 가능 비중)까지로 이번 주문 비중을 깎는다 — 남는 몫은 다음 사이클.
            if room is not None and weight_pct > room:
                sector_clamped = {"from": weight_pct, "to": round(room, 2), "sector": sector}
                weight_pct = max(0.01, round(room, 2))
        self._choose_side(page, side)
        self._fill_weight(page, weight_pct)
        self._choose_default_price_type(page, opp_tick=opp_tick)
        if not submit:
            return {
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "weight_pct": weight_pct,
                "accepted": False,
                "filled": False,
                "prepared": True,
                "result": "prepared Timefolio order form without submit",
                "summary": self.scrape_summary(),
            }

        self._submit_order(page)
        page.wait_for_timeout(1800)
        try:
            page.wait_for_load_state("networkidle", timeout=7000)
        except PlaywrightTimeoutError:
            pass
        # 1) 오류 팝업([TMS] 오류 등)은 무조건 거부 — 본문 성공단어 휴리스틱보다 먼저 본다.
        #    (2026-07-13: 이 팝업을 accepted 로 오판해 매도 불능이 하루 종일 은폐됐다.)
        tms = self.detect_and_close_tms_error(page)
        if tms:
            summary = self.scrape_summary()
            self._ensure_no_dialog(page)
            return {
                "ticker": ticker, "side": side, "qty": qty, "weight_pct": weight_pct,
                "opp_tick": opp_tick, "accepted": False, "filled": False, "pending": False,
                "rejected_reason": "tms_error", "result": f"[TMS] 오류: {tms}",
                "summary": summary,
            }
        after_text = self._body_text(page)
        summary = self.scrape_summary()
        self._ensure_no_dialog(page)   # 세션 재사용 — 다음 주문을 위해 티켓을 남기지 않는다
        # 2) 권위 판정 — 사이트 상태로만 본다(본문 단어 휴리스틱은 신뢰하지 않는다).
        #    ① 보유 수량이 바뀌었으면 체결. ② 주문표 '미체결' 칼럼에 이 종목의 작동 주문이
        #    떴으면 접수(=미체결). 둘 다 아니면 접수 안 된 것으로 본다 — 다음 사이클이 재시도한다.
        #    (본문 정규식은 가이드라인 위반표의 '초과%' 같은 단어에 걸려 오탐이 났고, 원장 탭
        #     클릭은 화면을 전환시켜 후속 주문을 깨뜨렸다 — 2026-07-13 실측.)
        # 사장 지시 2026-07-22: 제출 직후 1회 스크랩은 레이스다 — 사이트가 보유/미체결표에
        # 반영하기 전에 읽어 '미접수'로 오판했고(7/22 09:05 매도 3건이 실제로는 전부 체결),
        # 다음 사이클이 같은 매도를 재제출해 252990 을 이중 매도했다. 체결/접수 흔적이
        # 잡힐 때까지 최대 _CONFIRM_TRIES 회 재확인한 뒤에만 '미접수'로 판정한다.
        filled = self._position_changed(before_summary, summary, ticker=ticker, side=side)
        def _working_now() -> bool:
            try:
                return any(w.get("ticker") == ticker for w in self.list_working_orders())
            except Exception:  # noqa: BLE001
                return False
        working = _working_now()
        for _ in range(_CONFIRM_TRIES):
            if filled or working:
                break
            page.wait_for_timeout(_CONFIRM_WAIT_MS)
            summary = self.scrape_summary()
            filled = self._position_changed(before_summary, summary, ticker=ticker, side=side)
            working = _working_now()
        accepted = bool(filled or working)
        # 사장 지시 2026-07-21: 여기서 절대 페이지 본문(after_text)을 result 로 흘리지 않는다 —
        # 과거 after_text[-900:] 가 실패 로그를 '보유 잔고/섹터 비중/가이드라인…' 표 텍스트로 깨뜨렸다.
        # 사이트 상태(체결/미체결/미접수)만 짧은 한글 상태로 반환한다.
        if filled:
            _msg = f"{ticker} {side} 체결"
        elif working:
            _msg = f"{ticker} {side} 접수(미체결) — 상대호가 대기"
        else:
            _msg = f"{ticker} {side} 미접수 — 사이트가 주문을 받지 않음(다음 사이클 재시도)"
        if sector_clamped:
            _msg += (f" · 섹터 한도 클램프 {sector_clamped['from']:.2f}%→{sector_clamped['to']:.2f}% "
                     f"({sector_clamped['sector']} 추가편입가능)")
        return {
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "weight_pct": weight_pct,
            "opp_tick": opp_tick,
            "accepted": accepted,
            "working_order": working,
            "filled": bool(filled),
            "pending": bool(working and not filled),
            "sector_clamped": sector_clamped,
            "result": _msg,
            "summary": summary,
        }

    def _coerce_weight(self, order: dict[str, Any]) -> float:
        for key in ("weight_pct", "order_weight_pct"):
            try:
                value = float(order.get(key) or 0.0)
                if value > 0:
                    return max(0.01, min(100.0, value))
            except (TypeError, ValueError):
                pass
        amount = float(order.get("amount") or 0.0)
        total = float(order.get("total_eval") or 0.0)
        if amount > 0 and total > 0:
            return max(0.01, min(100.0, amount / total * 100.0))
        return 0.0

    def _position_changed(self, before: dict[str, Any], after: dict[str, Any], *, ticker: str, side: str) -> bool:
        def qty(summary: dict[str, Any]) -> int:
            for pos in summary.get("positions") or []:
                if str(pos.get("ticker") or "").zfill(6) == ticker:
                    return int(pos.get("qty") or 0)
            return 0

        before_qty = qty(before or {})
        after_qty = qty(after or {})
        if side == "buy":
            return after_qty > before_qty
        return after_qty < before_qty

    def _open_order_ticket(self, page: Page) -> None:
        # 세션을 재사용하면 직전 주문의 다이얼로그·오류 팝업이 남아 있을 수 있다 — 먼저 닫고 연다.
        self._ensure_no_dialog(page)
        button = page.get_by_role("button", name=re.compile(r"신규\s*주문"))
        if self._click_first_visible(button, timeout=2500):
            page.wait_for_timeout(700)
            page.locator('[role="dialog"]').last.wait_for(state="visible", timeout=5000)
            return
        # 버튼이 안 보이는 건 대개 화면이 다른 패널/팝업 상태로 어긋난 것이다. 새로고침 한 번으로
        # 복구하고 재시도한다 — 예전엔 여기서 곧장 예외를 던져 남은 청산·매수를 통째로 잃었다.
        self.refresh()
        self._ensure_no_dialog(page)
        if self._click_first_visible(page.get_by_role("button", name=re.compile(r"신규\s*주문")), timeout=3000):
            page.wait_for_timeout(700)
            page.locator('[role="dialog"]').last.wait_for(state="visible", timeout=5000)
            return
        # 새로고침 뒤에도 '신규 주문' 이 없으면 주문 창 자체가 닫힌 것이다 = 대회 종료.
        # 2026-09-01 부터 이 상태가 이어졌고(사장 확인: 대회 종료), 그때까지는 매 사이클
        # ERROR + 트레이스백으로 1,600건 넘게 쌓였다. 오류가 아니라 상태로 올린다.
        raise ContestClosedError(
            "타임폴리오 대회 종료 — 신규 주문 창이 열리지 않습니다(주문 불가).")

    def _ensure_no_dialog(self, page: Page) -> None:
        try:
            for _ in range(3):
                if page.locator('[role="dialog"]').count() == 0:
                    return
                self._close_ticket(page)
                page.wait_for_timeout(250)
        except Exception:  # noqa: BLE001 — 정리 실패는 치명적이지 않다
            pass

    def _select_ticker(self, page: Page, ticker: str) -> dict[str, Any]:
        dialog = page.locator('[role="dialog"]').last
        field = dialog.locator('input[placeholder="종목 선택"], input[placeholder*="종목"], input[type="text"]').nth(1)
        try:
            field = dialog.locator('input[placeholder="종목 선택"]').first
            field.wait_for(state="visible", timeout=2500)
        except Exception:
            field = dialog.locator('input[type="text"]').nth(1)
        field.fill(ticker)
        page.wait_for_timeout(900)
        # 후보 li 텍스트(예: "[A007660] 이수페타시스IT")에서 섹터코드를 파싱해둔다 — 매수 섹터 게이트용.
        cand_text = ""
        # React 포털의 자동완성 목록은 dialog 바깥에 붙지만, 이전 검색의 숨은 li도 DOM에 남는다.
        # 보이는 후보만 대상으로 하고 정확한 종목코드를 확인해 엉뚱한 종목 선택을 막는다.
        code_re = re.compile(rf"(?:^|\[)A?{re.escape(ticker)}(?:\]|\s|$)")
        candidates = page.locator("li:visible").filter(has_text=code_re)
        try:
            for i in range(min(candidates.count(), 6)):
                t = (candidates.nth(i).inner_text(timeout=800) or "").strip()
                if code_re.search(t):
                    cand_text = t
                    break
        except Exception:
            pass
        if self._click_first_visible(candidates, timeout=1200):
            page.wait_for_timeout(700)
        else:
            page.keyboard.press("ArrowDown")
            page.keyboard.press("Enter")
            page.wait_for_timeout(700)
        text = dialog.inner_text(timeout=3000)
        if ticker not in text and f"A{ticker}" not in text:
            raise RuntimeError(f"타임폴리오 종목 선택 실패: {ticker}")
        if "종목 미선택" in text:
            raise RuntimeError(f"타임폴리오 종목 후보를 선택하지 못했습니다: {ticker}")
        return {"sector": self._parse_sector_code(cand_text), "candidate_text": cand_text}

    def _choose_side(self, page: Page, side: str) -> None:
        dialog = page.locator('[role="dialog"]').last
        value = "true" if side == "buy" else "false"
        radio = dialog.locator(f'input[name="매수도"][value="{value}"]').first
        try:
            # Bootstrap의 btn-check 입력은 숨겨져 있어 input.check(force=True)가 클릭 뒤
            # 프런트 상태에 의해 원복될 수 있다. 사용자가 누르는 가시 label을 우선 클릭한다.
            radio_id = radio.get_attribute("id")
            label = dialog.locator(f'label[for="{radio_id}"]').first if radio_id else None
            if label is not None and label.count():
                try:
                    label.click(timeout=2500)
                except Exception:
                    label.click(force=True, timeout=2500)
            else:
                radio.check(force=True, timeout=2500)
            page.wait_for_timeout(300)
        except Exception as exc:
            raise RuntimeError(f"타임폴리오 {'매수' if side == 'buy' else '매도'} 선택 실패: {exc}") from exc
        checked = radio.evaluate("el => !!el.checked")
        opposite = dialog.locator(
            f'input[name="매수도"][value="{"false" if side == "buy" else "true"}"]').first
        opposite_checked = opposite.evaluate("el => !!el.checked") if opposite.count() else False
        if not checked or opposite_checked:
            raise RuntimeError(f"타임폴리오 {'매수' if side == 'buy' else '매도'} 상태 검증 실패 — 제출 중단")
        text = dialog.inner_text(timeout=3000)
        label = "매수" if side == "buy" else "매도"
        if label not in text:
            raise RuntimeError(f"타임폴리오 {label} 버튼을 찾지 못했습니다.")

    def _fill_weight(self, page: Page, weight_pct: float) -> None:
        dialog = page.locator('[role="dialog"]').last
        number_inputs = dialog.locator('input[type="number"]')
        if number_inputs.count() < 1:
            raise RuntimeError("타임폴리오 주문 비중 입력칸을 찾지 못했습니다.")
        value = f"{weight_pct:.2f}".rstrip("0").rstrip(".")
        number_inputs.nth(0).fill(value)
        got = number_inputs.nth(0).evaluate("el => el.value")
        try:
            ok = abs(float(got) - float(weight_pct)) <= 0.011
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise RuntimeError(f"주문 비중 입력 실패(입력 {value}, 실제 {got!r}) — 제출 중단")

    def _choose_default_price_type(self, page: Page, *, opp_tick: int = 1) -> None:
        """상대호가(Opp) + 틱 + 즉시실행을 세팅하고 **폼 상태를 읽어 검증**한다.

        예전엔 전부 try/except-pass 라 라디오 클릭이 조용히 실패하면 엉뚱한 가격유형으로 제출됐고,
        그 주문이 미체결로 살아남아 같은 종목 후속 청산까지 TMS 오류로 막았다(2026-07-13 사건).
        설정이 확인되지 않으면 제출 전에 예외로 끊는다 — 잘못된 주문보다 안 낸 주문이 낫다
        (사이클이 다음 루프에서 재시도한다).
        """
        dialog = page.locator('[role="dialog"]').last
        tick = max(1, min(10, int(opp_tick or 1)))
        try:
            dialog.locator('input[name="prcTy"][value="Opp"]').first.check(force=True, timeout=2000)
            page.wait_for_timeout(200)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"상대호가 선택 실패: {exc}") from exc
        if not dialog.locator('input[name="prcTy"][value="Opp"]').first.evaluate("el => !!el.checked"):
            raise RuntimeError("상대호가 라디오가 선택되지 않았습니다 — 제출 중단")
        nums = dialog.locator('input[type="number"]')
        if nums.count() < 2:
            raise RuntimeError("상대호가 틱 입력칸을 찾지 못했습니다 — 제출 중단")
        nums.nth(1).fill(str(tick))
        got = nums.nth(1).evaluate("el => el.value")
        if str(got).strip() != str(tick):
            raise RuntimeError(f"상대호가 틱 입력 실패(입력 {tick}, 실제 {got!r}) — 제출 중단")
        try:
            dialog.locator('input[name="isSlice"][value="false"]').first.check(force=True, timeout=1200)
        except Exception:
            pass  # 즉시 실행은 폼 기본값이 이미 true — 실패해도 기본값이 안전하다

    def _parse_sector_code(self, text: str) -> str | None:
        """드롭다운/표의 종목 라벨에서 섹터코드(뒤 2글자)를 추출. 예: "[A007660] 이수페타시스IT" -> "IT"."""
        s = re.sub(r"^\[A?\d{6}\]\s*", "", str(text or "")).strip()
        m = re.search(r"([A-Z][A-Za-z])\s*$", s)
        code = m.group(1) if m else s[-2:]
        return code if code in SECTOR_CODES else None

    def _scrape_sector_rooms(self, page: Page) -> dict[str, float]:
        """섹터표(섹터 비중 탭)의 '추가 편입 가능 비중'을 {섹터코드: 여유%} 로 읽는다. 실패 시 {}."""
        try:
            for pat in ("섹터 비중", "섹터비중"):
                if self._click_first_visible(page.get_by_text(re.compile(pat)), timeout=1000):
                    page.wait_for_timeout(700)
                    break
            rows = page.evaluate(r"""() => {
                const norm = s => String(s || '').replace(/\s+/g, ' ').trim();
                for (const t of document.querySelectorAll('table')) {
                    const head = norm([...t.querySelectorAll('tr')].slice(0, 3).map(r => r.innerText).join(' '));
                    if (/섹터코드/.test(head) && /추가\s*편입\s*가능/.test(head)) {
                        return [...t.querySelectorAll('tr')].map(tr =>
                            [...tr.children].map(td => norm(td.innerText || td.textContent)));
                    }
                }
                return [];
            }""")
            out: dict[str, float] = {}
            for r in rows:
                if len(r) >= 8 and str(r[0]) in SECTOR_CODES:
                    try:
                        out[str(r[0])] = float(str(r[7]).replace(",", ""))
                    except (TypeError, ValueError):
                        continue
            return out
        except Exception:
            return {}

    def _close_ticket(self, page: Page) -> None:
        try:
            if self._click_first_visible(page.get_by_role("button", name=re.compile(r"닫기|Close")), timeout=1200):
                page.wait_for_timeout(300)
                return
        except Exception:
            pass
        try:  # [TMS] 오류류 팝업은 '확인' 버튼뿐이다 — 남겨두면 다음 사이클의 신규 주문 버튼을 가린다
            if self._click_first_visible(page.get_by_role("button", name=re.compile(r"^확인$")), timeout=800):
                page.wait_for_timeout(300)
                return
        except Exception:
            pass
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(200)
        except Exception:
            pass

    def _submit_order(self, page: Page) -> None:
        dialog = page.locator('[role="dialog"]').last
        submit = dialog.get_by_role("button", name=re.compile(r"주문\s*제출"))
        if not self._click_first_visible(submit, timeout=2500):
            raise RuntimeError("타임폴리오 주문 제출 버튼을 찾지 못했습니다.")
        self._confirm_dialog(page)

    def _confirm_dialog(self, page: Page) -> None:
        """확인 팝업을 딱 한 번 누른다.

        예전엔 (확인|예|동의|주문) 을 전부 순회하며 매번 클릭해서, '확인'을 누른 뒤 **'신규 주문'
        버튼까지 눌러** 빈 주문 티켓을 띄워놓고 나왔다. 그 뜬 티켓이 다음 주문의 '신규 주문' 클릭을
        가로막아 "신규 주문 버튼을 찾지 못했습니다" 예외를 낳았다(2026-07-13). 첫 성공에서 멈추고,
        '주문' 같은 광범위 패턴은 쓰지 않는다."""
        for pattern in (r"^확인$", r"^예$", r"^동의$"):
            try:
                if self._click_first_visible(page.get_by_role("button", name=re.compile(pattern)), timeout=700):
                    page.wait_for_timeout(500)
                    return
            except Exception:
                continue

    def _click_first_visible(self, locator, *, timeout: int = 1200) -> bool:
        try:
            count = min(locator.count(), 12)
        except Exception:
            return False
        for idx in range(count):
            item = locator.nth(idx)
            try:
                if item.is_visible(timeout=timeout) and item.is_enabled(timeout=timeout):
                    item.click(timeout=timeout)
                    return True
            except Exception:
                continue
        return False

    def _body_text(self, page: Page) -> str:
        try:
            return page.locator("body").inner_text(timeout=5000)
        except Exception:
            return ""

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("TimefolioBrowser must be used as a context manager")
        return self.page
