#!/usr/bin/env bash
# Fetch the ONNX Runtime dylib that Intel Macs load at run time (faces, speakers).
#
# pyke ships no prebuilt onnxruntime for x86_64-apple-darwin, so ghostreel-core uses ort's
# load-dynamic feature there (crates/ghostreel-core/Cargo.toml) and its src/onnx.rs looks for
# libonnxruntime.dylib in the app's Contents/Frameworks, next to the executable, or at
# ORT_DYLIB_PATH. 1.23.2 is Microsoft's last release with an osx-x86_64 build; ort is compiled
# against API 23 on that target to match it.
#
# Writes:
#   src-tauri/binaries/onnxruntime-x86_64-apple-darwin/libonnxruntime.dylib  (bundled by stage-helpers.mjs)
#   <target>/release/libonnxruntime.dylib                                      (for the CLI built there)
# and prints the staged path. Other platforms link onnxruntime statically: nothing to do.
#
#   scripts/fetch-onnxruntime.sh [--force]
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=1.23.2
ARCHIVE="onnxruntime-osx-x86_64-$VERSION.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$ARCHIVE"
# GitHub's published asset digest for $ARCHIVE.
SHA256=d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != x86_64 ]; then
  echo "fetch-onnxruntime: only Intel macOS loads onnxruntime dynamically; nothing to do on $(uname -s) $(uname -m)."
  exit 0
fi

dest_dir=src-tauri/binaries/onnxruntime-x86_64-apple-darwin
dest="$dest_dir/libonnxruntime.dylib"
stamp="$dest_dir/VERSION"
if [ "${1:-}" != --force ] && [ -f "$dest" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$VERSION" ]; then
  echo "onnxruntime $VERSION already staged at $dest"
else
  tmp=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/onnxruntime.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT
  curl --fail --location --retry 3 --silent --show-error "$URL" -o "$tmp/$ARCHIVE"
  actual=$(shasum -a 256 "$tmp/$ARCHIVE" | cut -d' ' -f1)
  if [ "$actual" != "$SHA256" ]; then
    echo "fetch-onnxruntime: checksum mismatch for $ARCHIVE: expected $SHA256, got $actual" >&2
    exit 1
  fi
  tar -xzf "$tmp/$ARCHIVE" -C "$tmp"
  lib="$tmp/onnxruntime-osx-x86_64-$VERSION/lib/libonnxruntime.$VERSION.dylib"
  lipo "$lib" -verify_arch x86_64
  mkdir -p "$dest_dir"
  # A real file, not the archive's symlink: Tauri copies it into Contents/Frameworks and signs it.
  cp "$lib" "$dest"
  chmod 755 "$dest"
  cp "$tmp/onnxruntime-osx-x86_64-$VERSION/LICENSE" "$dest_dir/LICENSE"
  cp "$tmp/onnxruntime-osx-x86_64-$VERSION/ThirdPartyNotices.txt" "$dest_dir/ThirdPartyNotices.txt"
  printf '%s\n' "$VERSION" > "$stamp"
  echo "Staged onnxruntime $VERSION at $dest"
fi

# The CLI looks next to its own binary.
profile_dir="${CARGO_TARGET_DIR:-target}/release"
mkdir -p "$profile_dir"
cp "$dest" "$profile_dir/libonnxruntime.dylib"
echo "Copied it next to the CLI: $profile_dir/libonnxruntime.dylib"
