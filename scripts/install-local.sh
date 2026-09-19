#!/usr/bin/env bash
#
# GhostReel — local build & install (Linux)
#
# Builds a production binary with the Tauri CLI (embedded frontend, no dev server) and installs
# it for the current user, with a desktop entry + icon so it shows in your app launcher. This is
# the "from source" path; for prebuilt packages use the GitHub Releases (.deb / .AppImage) the
# release workflow produces.
#
# IMPORTANT: builds via `tauri build`, NOT `cargo build` — a bare cargo build leaves the app
# pointing at the Vite dev server (a blank "Could not connect to localhost" window).
#
# Usage:
#   ./scripts/install-local.sh                 # build + install app and CLI to ~/.local
#   ./scripts/install-local.sh --prefix ~/.local
#   ./scripts/install-local.sh --helpers       # also build+install ghostreel-asr / ghostreel-llm
#   ./scripts/install-local.sh --no-build      # install what is already built
#   ./scripts/install-local.sh --no-desktop    # binaries only, skip .desktop + icon
#   ./scripts/install-local.sh --no-cli        # skip the `ghostreel` CLI
#   ./scripts/install-local.sh -h
#
# Installs `ghostreel-app` (the desktop app) and `ghostreel` (the CLI: index, search, script,
# doctor, and the MCP server). The app is deliberately NOT called `ghostreel` — that name belongs
# to the CLI, and installing the app under it would overwrite the tool the app itself documents.
#
# With --helpers it also installs `ghostreel-asr` and `ghostreel-llm`, the whisper.cpp and
# llama.cpp helper processes used by the standalone profile (models in-process, no servers).
# They are off by default because they are a long CUDA build, and a machine using the shared
# servers (highllama, GhostPen STT) never runs them.
#
set -euo pipefail

# ---- locate the repo root (this script lives in scripts/) ----------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- logging -------------------------------------------------------------------------
info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die() {
    printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2
    exit 1
}

# ---- options -------------------------------------------------------------------------
PREFIX="${PREFIX:-$HOME/.local}"
DO_BUILD=1
DO_DESKTOP=1
DO_CLI=1
DO_HELPERS=0

while [ $# -gt 0 ]; do
    case "$1" in
    --prefix)
        PREFIX="${2:?--prefix needs a directory}"
        shift 2
        ;;
    --prefix=*)
        PREFIX="${1#*=}"
        shift
        ;;
    --no-build)
        DO_BUILD=0
        shift
        ;;
    --no-desktop)
        DO_DESKTOP=0
        shift
        ;;
    --no-cli)
        DO_CLI=0
        shift
        ;;
    --helpers)
        DO_HELPERS=1
        shift
        ;;
    -h | --help)
        # Print the leading comment block (everything after the shebang up to the first code line).
        awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
        exit 0
        ;;
    *) die "Unknown option: $1 (try -h)" ;;
    esac
done

BIN_DIR="$PREFIX/bin"
APP_DIR="$PREFIX/share/applications"
ICON_DIR="$PREFIX/share/icons/hicolor/128x128/apps"
# One workspace, one target directory at the repo root — not src-tauri/target.
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
# `tauri build` names the executable after productName ("GhostReel"); a plain cargo build leaves
# it as the crate's bin name. Accept whichever is there, newest first.
APP_SRC="$TARGET/release/GhostReel"
[ -x "$APP_SRC" ] || APP_SRC="$TARGET/release/ghostreel-app"
CLI_SRC="$TARGET/release/ghostreel"

# ---- build ---------------------------------------------------------------------------
if [ "$DO_BUILD" -eq 1 ]; then
    command -v cargo >/dev/null 2>&1 || die "cargo not found. Install Rust (https://rustup.rs) and re-run."
    command -v npm >/dev/null 2>&1 || die "npm not found. Install Node.js and re-run."

    cd "$ROOT"
    # A renamed checkout leaves build-script caches pointing at the old path, and tauri-build
    # then fails on a file under a directory that is gone. Clear those first; it is a no-op
    # when nothing moved.
    "$ROOT/scripts/clean-stale-target.sh" >/dev/null || true

    if [ ! -d node_modules ]; then
        info "Installing frontend dependencies (npm ci)…"
        npm ci || npm install
    fi

    info "Building the app (tauri build --no-bundle)…"
    # --no-bundle: a local install needs the executable, not deb/AppImage.
    npm run tauri -- build --no-bundle

    if [ "$DO_CLI" -eq 1 ]; then
        info "Building the CLI (ghostreel)…"
        cargo build --release -p ghostreel-cli
    fi

    if [ "$DO_HELPERS" -eq 1 ]; then
        info "Building the local model helpers (CUDA if the toolkit is present)…"
        # build-helpers.sh picks the backend and caps parallelism: a full-parallel CUDA build of
        # llama.cpp + whisper.cpp exhausts RAM (plan §11).
        "$ROOT/scripts/build-helpers.sh" asr llm
    fi
fi

if [ ! -x "$APP_SRC" ]; then
    APP_SRC="$TARGET/release/GhostReel"
    [ -x "$APP_SRC" ] || APP_SRC="$TARGET/release/ghostreel-app"
fi
[ -x "$APP_SRC" ] || die "App not found in $TARGET/release (GhostReel or ghostreel-app) — run without --no-build first."

# ---- stop any running instance so we can replace the binary --------------------------
if pgrep -x ghostreel-app >/dev/null 2>&1; then
    info "Stopping the running GhostReel instance…"
    pkill -x ghostreel-app 2>/dev/null || true
    sleep 1
fi

# ---- install binaries ----------------------------------------------------------------
info "Installing app    → $BIN_DIR/ghostreel-app"
install -Dm755 "$APP_SRC" "$BIN_DIR/ghostreel-app"

if [ "$DO_CLI" -eq 1 ]; then
    if [ -x "$CLI_SRC" ]; then
        info "Installing CLI    → $BIN_DIR/ghostreel"
        install -Dm755 "$CLI_SRC" "$BIN_DIR/ghostreel"
    else
        warn "ghostreel not built ($CLI_SRC) — skipping. Run without --no-build to build it."
    fi
fi

# The helpers are found next to whichever binary is running (doctor::locate), so they go in the
# same directory as the app and the CLI rather than anywhere clever.
if [ "$DO_HELPERS" -eq 1 ]; then
    for helper in ghostreel-asr ghostreel-llm; do
        if [ -x "$TARGET/release/$helper" ]; then
            info "Installing helper → $BIN_DIR/$helper"
            install -Dm755 "$TARGET/release/$helper" "$BIN_DIR/$helper"
        else
            warn "$helper not built — skipping."
        fi
    done
fi

# ---- desktop entry + icon ------------------------------------------------------------
if [ "$DO_DESKTOP" -eq 1 ]; then
    info "Installing desktop entry + icon"
    install -Dm644 "$ROOT/src-tauri/icons/128x128.png" "$ICON_DIR/ghostreel.png"
    install -d "$APP_DIR"
    cat >"$APP_DIR/ghostreel.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=GhostReel
Comment=Search inside your videos, locally
Exec=$BIN_DIR/ghostreel-app
Icon=ghostreel
Terminal=false
Categories=AudioVideo;Video;AudioVideoEditing;
StartupNotify=true
DESKTOP
    command -v update-desktop-database >/dev/null 2>&1 &&
        update-desktop-database "$APP_DIR" >/dev/null 2>&1 || true
fi

# ---- post-install notes --------------------------------------------------------------
info "Installed: $("$BIN_DIR/ghostreel" --version 2>/dev/null || echo ghostreel-app)"

case ":$PATH:" in
*":$BIN_DIR:"*) ;;
*) warn "$BIN_DIR is not on your PATH. Add it, e.g.:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

cat <<NOTES

GhostReel installed. Next steps:
  - Check the setup:   ghostreel doctor        (ffmpeg, GPU, model servers, database)
  - Start the app:     ghostreel-app &         (or find GhostReel in your launcher)
  - From a terminal:   ghostreel project add "My project" && ghostreel folder add ...
                       ghostreel index --watch
                       ghostreel search "what someone said"
                       ghostreel script chat --project "My project" "a 40 second teaser about ..."
  - As an MCP server:  ghostreel mcp           (stdio; drive it from another machine's agent)

  Models: either the shared servers (vision :8089, embeddings :8091, STT :8771) or, with
  --helpers, everything in-process. \`ghostreel doctor\` says which it found; \`ghostreel config\`
  and the app's Models page switch between them.
NOTES
