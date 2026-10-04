package com.arquant.mobile.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.arquant.mobile.data.ArQuantRepository
import com.arquant.mobile.network.*
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

data class DashState(
    val status: StatusResponse = StatusResponse(),
    val balance: BalanceResponse = BalanceResponse(),
    val news: List<NewsArticle> = emptyList(),
    val agents: List<AgentInfo> = emptyList(),
    val equity: EquityResponse = EquityResponse(),
    val trades: List<TradeEvent> = emptyList(),
    val strategy: StrategyResponse = StrategyResponse(),
    val logMessages: List<LogEntry> = emptyList(),
    val equityView: String = "realtime",
    val isLoading: Boolean = true,
    val toastMessage: String? = null,
    val sidebarOpen: Boolean = false,
)

data class LogEntry(
    val type: String,      // system, agent, ceo
    val text: String,
    val agentName: String? = null,
    val agentColor: Long? = null,
    val ts: Long = System.currentTimeMillis(),
    // ts 는 표시용 시각(같은 ms 충돌 가능). id 는 LazyColumn 키 등 정체성 식별 전용.
    val id: Long = nextLogId(),
) {
    companion object {
        private val counter = java.util.concurrent.atomic.AtomicLong()
        fun nextLogId(): Long = counter.incrementAndGet()
    }
}

@HiltViewModel
class DashViewModel @Inject constructor(
    private val repo: ArQuantRepository,
    private val wsManager: WsManager,
) : ViewModel() {

    private val _state = MutableStateFlow(DashState())
    val state: StateFlow<DashState> = _state.asStateFlow()

    // Agent colors matching web dashboard
    private val agentColors = mapOf(
        "주식운용실장" to 0xFF6366F1,        // 운용전략실장 → 주식운용실장 (사장 지시 2026-06-09)
        "글로벌리서치팀장" to 0xFF06B6D4,    // (구 전략리서치팀장)
        "계량분석팀장" to 0xFF10B981,
        "마켓센티먼트팀장" to 0xFFF59E0B,    // (구 뉴스분석팀장)
        "프롭트레이딩팀장" to 0xFFEF4444,    // (구 트레이딩팀장)
        "리스크관리실장" to 0xFF8B5CF6,
        "사후관리실장" to 0xFFF43F5E,
        "운용지원실장" to 0xFF14B8A6,
        "포트폴리오기획팀장" to 0xFFEC4899,  // 진입 thesis·보유계획
        "채권운용실장" to 0xFF3B82F6,        // 채권 ETF 슬리브
        "원자재운용실장" to 0xFFD97706,      // 원자재 ETF 슬리브 (사장 지시 2026-06-09)
    )

    init {
        // Connect WebSocket
        wsManager.connect()

        // Initial data load
        viewModelScope.launch { loadAll() }

        // Replay persisted log
        viewModelScope.launch { replayLog() }

        // Periodic status refresh (5s)
        viewModelScope.launch {
            while (true) {
                delay(5000)
                refreshStatus()
            }
        }

        // Periodic balance (10 min)
        viewModelScope.launch {
            while (true) {
                delay(600_000)
                val s = _state.value.status
                if (s.session != "OFF_HOURS") refreshBalance()
            }
        }

        // Periodic equity/trades (30s)
        viewModelScope.launch {
            while (true) {
                delay(30_000)
                refreshEquity()
                refreshTrades()
            }
        }

        // Periodic strategy (60s)
        viewModelScope.launch {
            while (true) {
                delay(60_000)
                refreshStrategy()
            }
        }

        // Observe WebSocket events
        viewModelScope.launch {
            wsManager.events.collect { ev -> handleEvent(ev) }
        }
    }

    private suspend fun loadAll() {
        try {
            val status = repo.status()
            val balance = repo.balance()
            val news = repo.news()
            val agents = repo.agents()
            val equity = repo.equity(_state.value.equityView)
            val trades = repo.trades()
            val strategy = repo.strategy()
            _state.update {
                it.copy(
                    status = status,
                    balance = balance,
                    news = news.articles,
                    agents = agents.agents,
                    equity = equity,
                    trades = trades.trades,
                    strategy = strategy,
                    isLoading = false,
                )
            }
        } catch (e: Exception) {
            // HTTP 302는 followRedirects(false) + 토큰 누락 시 CF Access 로그인 리다이렉트의 시그니처.
            // 'unexpected JSON token' 류의 모호한 에러 대신 원인을 직접 표시한다.
            val raw = e.message.orEmpty()
            val friendly = when {
                "302" in raw || "cloudflareaccess" in raw.lowercase() ->
                    "Cloudflare Access 인증 실패 — local.properties 의 cf.access.client.id/secret 확인"
                else -> "서버 연결 실패: $raw"
            }
            _state.update { it.copy(isLoading = false, toastMessage = friendly) }
        }
    }

    private suspend fun replayLog() {
        try {
            val events = repo.events()
            val entries = events.events.mapNotNull { ev -> eventToLog(ev) }
            if (entries.isNotEmpty()) {
                _state.update {
                    it.copy(logMessages = (listOf(LogEntry("system", "↩︎ 누적 로그 ${entries.size}건 복원")) + entries + it.logMessages).takeLast(300))
                }
            }
        } catch (_: Exception) {}
    }

    private fun handleEvent(ev: EventItem) {
        val log = eventToLog(ev) ?: return
        _state.update {
            val msgs = (it.logMessages + log).takeLast(300)
            it.copy(logMessages = msgs)
        }
        viewModelScope.launch { refreshStatus() }
    }

    // 사장 지시 2026-06-11: 메시지가 이미 이모지로 시작하면 타입 아이콘을 안 붙임(이중 아이콘 방지).
    // 백엔드가 "⚠️ ...","✅ ...","⛔ ..." 처럼 이모지를 넣어 보내는데 여기서 또 prepend 하던 구조 → 한 줄에 2개.
    private fun ic(icon: String, m: String): String {
        val t = m.trimStart()
        val emojiLed = t.isNotEmpty() && Character.getType(t.codePointAt(0)) == Character.OTHER_SYMBOL.toInt()
        return if (emojiLed) m else "$icon $m"
    }

    private fun eventToLog(ev: EventItem): LogEntry? {
        // 사장 피드백 2026-05-16: WS/replay 로 들어오는 메시지(특히 agent_msg)가
        // cleanLog 를 안 거쳐 **굵게**·## 헤더가 그대로 노출되던 버그 — 여기서 일괄 정리.
        // (웹은 addLog 가 모든 메시지를 _cleanLog 처리하지만 모바일은 이 경로가 우회됐음)
        val m = cleanLog(ev.message)
        return when (ev.type) {
            "status" -> LogEntry("system", "[${ev.state}] $m")
            "news" -> LogEntry("system", ic("📰", m))
            "trigger" -> LogEntry("system", ic("🔔", m))
            "agent_msg" -> LogEntry("agent", m, ev.agent, agentColors[ev.agent])
            "cycle_complete" -> LogEntry("system", "🏁 사이클 완료${if (ev.tradesTotal != null) " · 누적 실매매 ${ev.tradesTotal}건" else ""}")
            "execution_ready" -> LogEntry("system", ic("✅", m))
            "execution_skipped" -> LogEntry("system", ic("❌", m))
            "trade_executed" -> LogEntry("system", ic("💸", m) + (if (ev.tradesTotal != null) " (누적 ${ev.tradesTotal}건)" else ""))
            "trade_failed" -> LogEntry("system", ic("⚠️", m))
            "error" -> LogEntry("system", ic("❌", m))
            else -> null
        }
    }

    fun toggleSidebar() {
        _state.update { it.copy(sidebarOpen = !it.sidebarOpen) }
    }

    fun closeSidebar() {
        _state.update { it.copy(sidebarOpen = false) }
    }

    fun startMonitor() {
        viewModelScope.launch {
            try {
                val r = repo.start()
                addLog("system", r.message)
                refreshStatus()
            } catch (e: Exception) {
                addLog("system", "❌ 시작 실패: ${e.message}")
            }
        }
    }

    fun stopMonitor() {
        viewModelScope.launch {
            try {
                val r = repo.stop()
                addLog("system", r.message)
                refreshStatus()
            } catch (e: Exception) {
                addLog("system", "❌ 중지 실패: ${e.message}")
            }
        }
    }

    fun sendCeo(message: String) {
        if (message.isBlank()) return
        addLog("ceo", "🔴 [사장] $message")
        viewModelScope.launch {
            try {
                val r = repo.ceo(message)
                addLog("agent", r.response, "주식운용실장", agentColors["주식운용실장"])
            } catch (e: Exception) {
                addLog("system", "❌ ${e.message}")
            }
        }
    }

    fun clearLog() {
        viewModelScope.launch {
            try {
                repo.clearEvents()
                _state.update { it.copy(logMessages = listOf(LogEntry("system", "로그 초기화됨"))) }
            } catch (_: Exception) {}
        }
    }

    fun clearTrades() {
        viewModelScope.launch {
            try {
                val r = repo.clearTrades()
                addLog("system", r.message)
                refreshTrades()
            } catch (_: Exception) {}
        }
    }

    fun setEquityView(view: String) {
        _state.update { it.copy(equityView = view) }
        viewModelScope.launch { refreshEquity() }
    }

    fun setStrategy(name: String) {
        viewModelScope.launch {
            try {
                repo.setStrategy(name)
                refreshStrategy()
                refreshStatus()
                addLog("system", "⚙️ 전략 변경 → $name")
            } catch (e: Exception) {
                addLog("system", "❌ 전략 변경 실패: ${e.message}")
            }
        }
    }

    fun refreshBalance() {
        viewModelScope.launch {
            try {
                _state.update { it.copy(balance = repo.balance()) }
            } catch (_: Exception) {}
        }
    }

    fun consumeToast() {
        _state.update { it.copy(toastMessage = null) }
    }

    private suspend fun refreshStatus() {
        try { _state.update { it.copy(status = repo.status()) } } catch (_: Exception) {}
    }

    private suspend fun refreshEquity() {
        try { _state.update { it.copy(equity = repo.equity(it.equityView)) } } catch (_: Exception) {}
    }

    private suspend fun refreshTrades() {
        try { _state.update { it.copy(trades = repo.trades().trades) } } catch (_: Exception) {}
    }

    private suspend fun refreshStrategy() {
        try { _state.update { it.copy(strategy = repo.strategy()) } } catch (_: Exception) {}
    }

    private fun addLog(type: String, text: String, agentName: String? = null, color: Long? = null) {
        val entry = LogEntry(type, cleanLog(text), agentName, color)
        _state.update { it.copy(logMessages = (it.logMessages + entry).takeLast(300)) }
    }

    // 사장 피드백 2026-05-15 모바일 #3: 에이전트 통신 로그에서 **, ##, ---, |, ``` 등 마크다운 잔여물 정리.
    // 사람이 읽기 좋은 일반 텍스트 형태로 변환.
    private fun cleanLog(t: String): String =
        t.replace(Regex("```[^\n]*"), "")              // 코드 블록 펜스 제거
            .replace(Regex("\\*\\*+"), "")              // **bold** → 보이지 않게 제거
            .replace(Regex("__+"), "")                  // __italic__ → 평문
            .replace(Regex("(?m)^#{1,6}\\s*"), "")     // 줄머리 # 헤더 → 평문
            .replace(Regex("#{2,}\\s?"), "")            // 인라인 ##/### 잔여물 제거 (단일 #는 OPS#17 등 보존)
            .replace(Regex("(?m)^---+\\s*$"), "")      // --- 구분선 → 빈 줄
            .replace(Regex("(?m)^\\|.*\\|\\s*$"), "")  // 표 행 자체 제거 (UI 깨짐 방지)
            .replace(Regex("`([^`]+)`"), "$1")          // 인라인 코드 → 평문
            .replace(Regex("[ \\t]{3,}"), "  ")

    override fun onCleared() {
        super.onCleared()
        // 사장 지시 2026-05-21: WS 연결 해제는 WsRelayService(onDestroy)가 담당한다.
        // 여기서 disconnect 하면 Activity 재생성/뷰모델 정리 시 백그라운드 연결이 끊겨
        // 알림이 멈추므로, 연결 수명은 foreground service 가 단독 소유한다.
    }
}
