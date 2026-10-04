package com.arquant.mobile.widget

import android.content.Context
import android.util.Log
import com.arquant.mobile.BuildConfig
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/** Read-only widgets use the Access cookie synchronized by the dashboard. */
internal object WidgetHttp {
    private const val TAG = "WidgetHttp"

    private val client by lazy {
        OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(15, TimeUnit.SECONDS)
            .followRedirects(false)
            .build()
    }

    private fun sessionToken(context: Context): String =
        context.getSharedPreferences("arquant_auth", Context.MODE_PRIVATE)
            .getString("cf_access_token", "")
            .orEmpty()

    private fun base(): String =
        BuildConfig.ARQUANT_BASE_URL.let { if (it.endsWith("/")) it.dropLast(1) else it }

    /**
     * GET /api/balance → { buying_power: {...}, holdings: [...] }
     * 반환: 전체 JSON 또는 실패(미로그인 401 포함) 시 null.
     */
    fun fetchBalance(context: Context): JSONObject? {
        val url = "${base()}/api/balance"
        return runCatching {
            val token = sessionToken(context)
            val req = Request.Builder()
                .url(url)
                .header("User-Agent", "ArQuant-Android-Widget/1.0")
                .apply { if (token.isNotBlank()) header("Cookie", "CF_Authorization=$token") }
                .build()
            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    Log.w(TAG, "fetchBalance non-2xx: ${resp.code}")
                    return@use null
                }
                val body = resp.body?.string() ?: return@use null
                JSONObject(body)
            }
        }.onFailure { Log.e(TAG, "fetchBalance failed", it) }
            .getOrNull()
    }
}
