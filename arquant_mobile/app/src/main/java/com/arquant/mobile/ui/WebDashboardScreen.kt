package com.arquant.mobile.ui

import android.annotation.SuppressLint
import android.content.Intent
import android.net.Uri
import android.webkit.CookieManager
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView

/** Cloudflare owns authentication; the application never asks for a password. */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun WebDashboardScreen(baseUrl: String, onTokenSynced: (String) -> Unit, onLoggedOut: () -> Unit) {
    var browser by remember { mutableStateOf<WebView?>(null) }
    var canGoBack by remember { mutableStateOf(false) }
    val homeHost = Uri.parse(baseUrl).host
    val sync by rememberUpdatedState(onTokenSynced)
    val logout by rememberUpdatedState(onLoggedOut)
    BackHandler(canGoBack) { browser?.goBack() }
    AndroidView(modifier = Modifier.fillMaxSize().systemBarsPadding(), factory = { context ->
        WebView(context).apply {
            browser = this
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = false
            settings.allowContentAccess = false
            settings.mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW
            CookieManager.getInstance().setAcceptCookie(true)
            webViewClient = object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                    val uri = request.url
                    val host = uri.host.orEmpty()
                    if (uri.scheme == "https" && (host == homeHost || host.endsWith(".cloudflareaccess.com"))) {
                        if (uri.path.orEmpty().contains("/cdn-cgi/access/logout")) logout()
                        return false
                    }
                    if (uri.scheme in listOf("https", "http", "mailto")) {
                        runCatching { context.startActivity(Intent(Intent.ACTION_VIEW, uri)) }
                    }
                    return true
                }
                override fun onPageFinished(view: WebView, url: String) {
                    canGoBack = view.canGoBack()
                    if (Uri.parse(url).host != homeHost) return
                    val cookie = CookieManager.getInstance().getCookie(baseUrl).orEmpty()
                        .split(";").map { it.trim() }
                        .firstOrNull { it.startsWith("CF_Authorization=") }
                        ?.substringAfter("=").orEmpty()
                    if (cookie.isNotBlank()) sync(cookie)
                    CookieManager.getInstance().flush()
                }
            }
            loadUrl(baseUrl)
        }
    })
    DisposableEffect(Unit) {
        onDispose { browser?.destroy(); browser = null }
    }
}
