package com.arquant.mobile.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.arquant.mobile.MainActivity
import com.arquant.mobile.network.WsManager
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

/**
 * Foreground service that keeps the WebSocket alive when the app is backgrounded
 * (사장 지시 2026-05-21: 모바일 백그라운드 지속 알림이 앱 재빌드의 핵심 목적).
 *
 * - 인증 후 MainActivity 가 startForegroundService 로 기동한다.
 * - onCreate/onStartCommand 에서 [WsManager.connect] 를 호출해 서비스 수명 동안 연결을 유지한다
 *   (WsManager.connect 는 idempotent — DashViewModel 과 중복 호출돼도 안전).
 * - persistent notification 으로 OS 가 프로세스를 회수하지 않게 해 백그라운드에서도 알림이 도착한다.
 * - 로그아웃 시 MainActivity 가 stopService → onDestroy 에서 연결을 끊는다.
 */
@AndroidEntryPoint
class WsRelayService : Service() {

    @Inject lateinit var wsManager: WsManager

    companion object {
        private const val CHANNEL_ID = "arquant_ws_relay"
        private const val NOTIF_ID = 9001
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createChannel()
        val pi = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            },
            PendingIntent.FLAG_IMMUTABLE,
        )
        val notif = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle("QuantInSight")
            .setContentText("실시간 모니터링 중")
            .setOngoing(true)
            .setContentIntent(pi)
            .build()
        startForeground(NOTIF_ID, notif)
        wsManager.connect()   // 서비스 수명 동안 WS 연결 유지 (백그라운드 알림)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        wsManager.connect()   // 재기동 시에도 연결 보장 (idempotent)
        return START_STICKY    // 프로세스가 회수돼도 OS 가 서비스를 재시작 시도
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        // 버그수정 2026-05-29(사장 제보): 최근앱에서 앱을 스와이프하면 기본 동작으로 task 의
        // foreground service 가 함께 종료돼 알림이 끊겼다. 포그라운드 알림이 아직 살아있는 이
        // 시점엔 self-restart 가 허용되므로 서비스를 즉시 재기동해 백그라운드 알림을 유지한다.
        // (로그아웃 종료는 stopService→onDestroy 경로이므로 여기로 오지 않는다.)
        try {
            ContextCompat.startForegroundService(
                applicationContext, Intent(applicationContext, WsRelayService::class.java))
        } catch (_: Exception) {
        }
        super.onTaskRemoved(rootIntent)
    }

    override fun onDestroy() {
        wsManager.disconnect()  // 로그아웃 등으로 서비스가 멈출 때만 연결 해제
        super.onDestroy()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                nm.createNotificationChannel(
                    NotificationChannel(CHANNEL_ID, "QuantInSight 실시간", NotificationManager.IMPORTANCE_LOW).apply {
                        description = "WebSocket 실시간 연결 유지"
                    }
                )
            }
        }
    }
}
