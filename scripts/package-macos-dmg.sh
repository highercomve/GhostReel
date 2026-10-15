#!/usr/bin/env bash
# Build directly from a staging folder: no Finder automation or mounted writable
# image for Spotlight/Finder to hold open (create-dmg can fail with Resource busy).
set -euo pipefail

[ "$(uname -s)" = Darwin ] || { echo 'DMG packaging requires macOS.' >&2; exit 1; }
: "${APPLE_SIGNING_IDENTITY:?A signing identity is required}"
arch="${1:?Usage: scripts/package-macos-dmg.sh <aarch64|x64>}"
case "$arch" in aarch64|x64) ;; *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; esac

app=target/release/bundle/macos/GhostReel.app
test -d "$app"
codesign --verify --deep --strict "$app"
version=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")
out="$(pwd)/target/release/bundle/dmg/GhostReel_${version}_${arch}.dmg"
mkdir -p "$(dirname "$out")"
stage=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/ghostreel-dmg.XXXXXX")
trap 'rm -rf "$stage"' EXIT
ditto "$app" "$stage/GhostReel.app"
ln -s /Applications "$stage/Applications"

hdiutil create -volname GhostReel -srcfolder "$stage" -fs HFS+ \
  -format UDZO -ov "$out"
hdiutil verify "$out"
codesign --force --sign "$APPLE_SIGNING_IDENTITY" --timestamp=none "$out"
codesign --verify --strict --verbose=2 "$out"
echo "Created signed installer: $out"
