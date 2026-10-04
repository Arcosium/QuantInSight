package com.arquant.mobile.network

import com.arquant.mobile.security.TokenManager
import okhttp3.Interceptor
import okhttp3.Response
import javax.inject.Inject
import javax.inject.Singleton

/** Forward the Access application cookie to the protected host only. */
@Singleton
class AppAuthInterceptor @Inject constructor(
    private val tokenManager: TokenManager,
) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val req = chain.request()
        val token = tokenManager.get()
        if (token.isBlank() || req.url.host != "quantinsight.ai-ve.uk") return chain.proceed(req)
        return chain.proceed(
            req.newBuilder().header("Cookie", "CF_Authorization=$token").build()
        )
    }
}
