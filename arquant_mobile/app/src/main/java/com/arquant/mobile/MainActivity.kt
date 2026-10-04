package com.arquant.mobile

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.arquant.mobile.security.TokenManager
import com.arquant.mobile.service.WsRelayService
import com.arquant.mobile.ui.WebDashboardScreen
import com.arquant.mobile.ui.theme.ArQuantTheme
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    @Inject lateinit var tokenManager: TokenManager
    private val notifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(this,
                Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        setContent {
            ArQuantTheme {
                WebDashboardScreen(
                    baseUrl = BuildConfig.ARQUANT_BASE_URL,
                    onTokenSynced = { token ->
                        tokenManager.save(token)
                        ContextCompat.startForegroundService(this, Intent(this, WsRelayService::class.java))
                    },
                    onLoggedOut = {
                        tokenManager.clear()
                        stopService(Intent(this, WsRelayService::class.java))
                    },
                )
            }
        }
    }
}
