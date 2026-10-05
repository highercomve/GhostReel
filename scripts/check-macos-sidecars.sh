#!/usr/bin/env bash
# Fail packaging if a staged sidecar needs Homebrew/build-machine libraries or lacks preview features.
set -euo pipefail
cd "$(dirname "$0")/.."
triple="${1:?Usage: scripts/check-macos-sidecars.sh <rust triple>}"
case "$triple" in
  aarch64-apple-darwin) arch=arm64 ;;
  x86_64-apple-darwin) arch=x86_64 ;;
  *) echo "Unsupported macOS target: $triple" >&2; exit 1 ;;
esac
for name in ffmpeg ffprobe ghostreel-asr ghostreel-llm; do
  binary="src-tauri/binaries/$name-$triple"
  lipo "$binary" -verify_arch "$arch"
  deps=$(otool -L "$binary")
  # All native and codec libraries must be static; only OS libraries/frameworks may remain.
  non_system=$(printf '%s\n' "$deps" | grep '^[[:space:]]' | grep -Ev '^[[:space:]]+(/usr/lib/|/System/Library/)' || true)
  if [ -n "$non_system" ]; then
    echo "Non-system dynamic dependency in $binary:" >&2
    echo "$non_system" >&2
    exit 1
  fi
done
ffmpeg="src-tauri/binaries/ffmpeg-$triple"
filters=$("$ffmpeg" -hide_banner -filters 2>&1)
for filter in drawtext subtitles; do
  printf '%s\n' "$filters" | grep -E "[[:space:]]${filter}[[:space:]]" >/dev/null
done
# These are the software encoders used by preview rendering and embedded playback proxies.
"$ffmpeg" -hide_banner -loglevel error -f lavfi -i color=s=320x240:d=0.1 \
  -c:v libx264 -pix_fmt yuv420p -f null -
"src-tauri/binaries/ffprobe-$triple" -version
