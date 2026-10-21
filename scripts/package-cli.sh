#!/usr/bin/env bash
# Package the standalone GhostReel CLI distribution tarball for Linux x64.
set -euo pipefail

cd "$(dirname "$0")/.."

# Linux x64 only
os="$(uname -s)"
arch="$(uname -m)"
if [ "$os" != "Linux" ] || [ "$arch" != "x86_64" ]; then
  echo "Error: scripts/package-cli.sh is Linux x64 only (detected $os $arch)." >&2
  exit 1
fi

TRIPLE="x86_64-unknown-linux-gnu"
BIN_DIR="src-tauri/binaries"

# Verify ffmpeg and required helpers are staged
if [ ! -f "$BIN_DIR/ffmpeg-$TRIPLE" ] || \
   [ ! -f "$BIN_DIR/ffprobe-$TRIPLE" ] || \
   [ ! -f "$BIN_DIR/ghostreel-asr-$TRIPLE" ] || \
   [ ! -f "$BIN_DIR/ghostreel-llm-$TRIPLE" ]; then
  echo "run node scripts/fetch-sidecars.mjs && node scripts/stage-helpers.mjs first" >&2
  exit 1
fi

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
GHOSTREEL_BIN="$TARGET_DIR/release/ghostreel"

if [ ! -f "$GHOSTREEL_BIN" ]; then
  echo "ghostreel CLI binary not found at $GHOSTREEL_BIN, building..."
  cargo build --release -p ghostreel-cli
fi

DIST_DIR="$TARGET_DIR/dist"
STAGE_DIR="$DIST_DIR/ghostreel-cli-linux-x64"

echo "Staging CLI distribution in $STAGE_DIR..."
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# Copy CLI binary and staged helpers/ffmpeg with triple suffix stripped
cp "$GHOSTREEL_BIN" "$STAGE_DIR/ghostreel"
cp "$BIN_DIR/ghostreel-asr-$TRIPLE" "$STAGE_DIR/ghostreel-asr"
cp "$BIN_DIR/ghostreel-llm-$TRIPLE" "$STAGE_DIR/ghostreel-llm"
cp "$BIN_DIR/ffmpeg-$TRIPLE" "$STAGE_DIR/ffmpeg"
cp "$BIN_DIR/ffprobe-$TRIPLE" "$STAGE_DIR/ffprobe"

chmod 755 "$STAGE_DIR"/*

# Copy lib/ from src-tauri/lib (excluding .keep)
mkdir -p "$STAGE_DIR/lib"
if [ -d "src-tauri/lib" ]; then
  shopt -s nullglob
  for f in src-tauri/lib/*; do
    if [ "$(basename "$f")" != ".keep" ]; then
      cp -a "$f" "$STAGE_DIR/lib/"
    fi
  done
  shopt -u nullglob
fi

# Copy ffmpeg-LICENSE.txt if present
if [ -f "$BIN_DIR/ffmpeg-LICENSE.txt" ]; then
  cp "$BIN_DIR/ffmpeg-LICENSE.txt" "$STAGE_DIR/"
fi

# Write README.txt
cat << 'EOF' > "$STAGE_DIR/README.txt"
GhostReel CLI
=============

Quick start:
  ./ghostreel doctor
  ./ghostreel --help

GPU Acceleration:
  - CUDA builds need an NVIDIA driver 570+; CUDA runtime libraries are bundled.
  - Vulkan builds need a Vulkan 1.2-capable AMD, Intel or NVIDIA GPU driver.
  - Bundled CUDA runtime libraries in lib/ are discovered via RUNPATH ($ORIGIN/lib).
  - Install the matching GPU driver for the backend shipped in this bundle.
EOF

# Package tarball
TARBALL="$DIST_DIR/ghostreel-cli-linux-x64.tar.gz"
echo "Creating $TARBALL..."
tar -C "$DIST_DIR" -czf "$TARBALL" ghostreel-cli-linux-x64

size="$(du -h "$TARBALL" | cut -f1)"
echo "Packaged $TARBALL ($size)"
