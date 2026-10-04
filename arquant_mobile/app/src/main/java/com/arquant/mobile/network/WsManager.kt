package com.arquant.mobile.network

import android.content.Context
import android.util.Log
import com.arquant.mobile.BuildConfig
import com.arquant.mobile.notification.TradeNotifier
import com.arquant.mobile.security.TokenManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.serialization.json.Json
import okhttp3.*
import javax.inject.Inject
import javax.inject.Singleton

/**
 * OkHttp WebSocket wrapper that auto-reconnects, emits parsed EventItems,
 * and fires push notifications for trade events.
 */
@Singleton
class WsManager @Inject constructor(
    private val client: OkHttpClient,
    private val json: Json,
    private val tokenManager: TokenManager,
    @ApplicationContext private val appContext: Context,
) {
    companion object {
        private const val TAG = "WsManager"
        private const val RECONNECT_MS = 3000L
    }

    private val _events = MutableSharedFlow<EventItem>(extraBufferCapacity = 256)
    val events = _events.asSharedFlow()

    private var ws: WebSocket? = null
    private var scope: CoroutineScope? = null
    private var running = false

    fun connect() {
        if (running) return
        running = true
        scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
        doConnect()
    }

    fun disconnect() {
        running = false
        ws?.close(1000, "bye")
        ws = null
        scope?.cancel()
        scope = null
    }

    private fun doConnect() {
        if (!running) return
        // Cloudflare validates the Access cookie before forwarding the signed JWT.
        val base = BuildConfig.ARQUANT_WS_URL
        val token = tokenManager.get()
        // 사장 지시 2026-05-21: client=mobile 로 표시 → 서버가 이 연결에 한해 프로필 알림설정으로
        // 4종 푸시(체결신청·체결완료·사이클완료·장마감)를 게이트한다. (웹 WebView 연결은 전체 수신)
        var url = base + (if (base.contains("?")) "&" else "?") + "client=mobile"
        val req = Request.Builder().url(url).header("Cookie", "CF_Authorization=$token").build()
        ws = client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "WS connected")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val ev = json.decodeFromString<EventItem>(text)
                    _events.tryEmit(ev)
                    // 사장 지시 2026-05-21: 모바일 알림 4종 + 실패.
                    //   order_submitted(체결 신청) · trade_executed/failed(체결 완료/실패)
                    //   · cycle_complete(사이클 완료) · market_close(장 마감)
                    // (서버가 client=mobile 연결엔 프로필 설정으로 켜진 종류만 보내므로 1차 필터는 서버가 수행)
                    if (ev.type in setOf("order_submitted", "trade_executed", "trade_failed",
                                         "cycle_complete", "market_close")) {
                        TradeNotifier.maybeNotify(appContext, ev)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Parse error: ${e.message}")
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "WS failure: ${t.message}")
                scheduleReconnect()
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "WS closed: $code")
                scheduleReconnect()
            }
        })
    }

    private fun scheduleReconnect() {
        if (!running) return
        scope?.launch {
            delay(RECONNECT_MS)
            doConnect()
        }
    }
}
