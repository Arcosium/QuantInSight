package com.arquant.mobile.security

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/** Access application cookie synchronized from the WebView. */
@Singleton
class TokenManager @Inject constructor(
    @ApplicationContext context: Context,
) {
    private val prefs = context.getSharedPreferences("arquant_auth", Context.MODE_PRIVATE)

    init { prefs.edit().remove("session_token").apply() }

    fun get(): String = prefs.getString(KEY, "").orEmpty()

    fun save(token: String) {
        prefs.edit().putString(KEY, token).apply()
    }

    fun clear() {
        prefs.edit().remove(KEY).apply()
    }

    fun isLoggedIn(): Boolean = get().isNotBlank()

    private companion object {
        const val KEY = "cf_access_token"
    }
}
