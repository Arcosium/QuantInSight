#!/usr/bin/env bash
# QuantInSight Android APK 빌드 — 한 방 자동화 (GB10 네이티브, sudo/docker 불필요)
#
# 사용법:
#   ./build_apk.sh                # debug APK (기본)
#   ./build_apk.sh release        # release APK (서명 필요)
#
# 방식: aarch64 호스트에서 JDK17·Gradle 은 네이티브로 돌리고, x86_64 전용인
# aapt2 만 qemu-user-static 으로 에뮬레이션한다(~/android-build/native/, poyong 패턴).
# (구 docker/QEMU 이미지 방식은 Oracle 호스트 전용이라 GB10 이관 때 폐기.)
#
# 산출물: /home/arcosium/projects/QuantInSight/QuantInSight.apk
set -euo pipefail

HERE="/home/arcosium/projects/QuantInSight"
MOBILE_DIR="${HERE}/arquant_mobile"
NATIVE="${HOME}/android-build/native"   # jdk17 / sdk / aapt2-wrap (README 참고)
VARIANT="${1:-debug}"
GRADLE_TASK="assemble$(echo "${VARIANT:0:1}" | tr '[:lower:]' '[:upper:]')${VARIANT:1}"
APK_OUT="${HERE}/QuantInSight.apk"

log(){ printf '\033[1;36m[quantinsight-apk]\033[0m %s\n' "$*"; }
die(){ printf '\033[1;31m[quantinsight-apk] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "$MOBILE_DIR" ] || die "모바일 디렉토리 없음: $MOBILE_DIR"
[ -x "${NATIVE}/aapt2-wrap/aapt2" ] || die "네이티브 빌드 툴체인 없음: ${NATIVE} (jdk17/sdk/aapt2-wrap)"

export JAVA_HOME="${NATIVE}/jdk17"
export ANDROID_HOME="${NATIVE}/sdk"
cd "$MOBILE_DIR"
echo "sdk.dir=${ANDROID_HOME}" > local.properties

log "Gradle 실행: $GRADLE_TASK (JDK17 네이티브 + qemu aapt2)"
chmod +x ./gradlew
# AGP 는 오버라이드 경로의 파일명이 정확히 'aapt2' 여야 받아들인다 (aapt2-wrap/aapt2)
./gradlew --no-daemon "$GRADLE_TASK" \
  -Pandroid.aapt2FromMavenOverride="${NATIVE}/aapt2-wrap/aapt2"

APK_SRC="app/build/outputs/apk/${VARIANT}/app-${VARIANT}.apk"
[ -f "$APK_SRC" ] || die "APK 미생성: $APK_SRC"
cp -f "$APK_SRC" "$APK_OUT"
ls -lh "$APK_OUT"
log "완료. 폰에 사이드로드: QuantInSight.apk"
