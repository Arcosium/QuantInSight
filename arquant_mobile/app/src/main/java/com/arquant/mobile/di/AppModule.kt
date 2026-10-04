package com.arquant.mobile.di

import com.arquant.mobile.BuildConfig
import com.arquant.mobile.network.AppAuthInterceptor
import com.arquant.mobile.network.ArQuantApi
import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
        explicitNulls = false
    }

    @Provides
    @Singleton
    fun provideOkHttp(authInterceptor: AppAuthInterceptor): OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(15, TimeUnit.SECONDS)
        // Cloudflare Access application cookie is scoped to the protected host.
        .addInterceptor(authInterceptor)
        // Access redirects expired sessions to sign-in; API clients must not follow it.
        // 보수적으로 자동 리다이렉트는 차단 유지.
        .followRedirects(false)
        .addInterceptor(HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        })
        .build()

    @Provides
    @Singleton
    fun provideRetrofit(client: OkHttpClient): Retrofit = Retrofit.Builder()
        .baseUrl(BuildConfig.ARQUANT_BASE_URL + "/")
        .client(client)
        .addConverterFactory(json.asConverterFactory("application/json".toMediaType()))
        .build()

    @Provides
    @Singleton
    fun provideApi(retrofit: Retrofit): ArQuantApi = retrofit.create(ArQuantApi::class.java)

    @Provides
    @Singleton
    fun provideJson(): Json = json
}
