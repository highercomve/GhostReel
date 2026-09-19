#!/usr/bin/env bash
#
# Remove build-script caches that still point at a directory this repo no longer lives in.
#
# Cargo bakes absolute paths into the output of every build script it runs, and nothing in cargo
# notices when the checkout is renamed or moved. Most crates survive it; the ones that read files
# back out of a recorded path do not. After this repo moved from HighVid/ to ghostreel/,
# `tauri build` failed with:
#
#   failed to read plugin permissions: failed to read file
#   '/home/sergiom/Code/HighVid/target/release/build/tauri-<hash>/out/permissions/...'
#
# which says nothing about the real cause. llama-cpp-sys-2 and whisper-rs-sys fail the same way.
#
# This finds the build directories whose recorded paths have stopped existing and deletes just
# those, so their build scripts re-run against the current location. A build directory that points
# somewhere else that *does* exist is left alone: that is a shared CARGO_TARGET_DIR, not a stale
# cache.
#
# Usage:
#   ./scripts/clean-stale-target.sh            # clean
#   ./scripts/clean-stale-target.sh --check    # report only, exit 1 if anything is stale
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

[ -d "$TARGET" ] || exit 0

stale=()
for out in "$TARGET"/*/build/*/output; do
    [ -f "$out" ] || continue
    # Every absolute path this build script recorded that lives inside some target directory.
    while read -r path; do
        [ -n "$path" ] || continue
        case "$path" in "$TARGET"/*) continue ;; esac # ours
        [ -e "$path" ] && continue                    # elsewhere, but real: a shared target dir
        stale+=("$(dirname "$out")")
        break
    done < <(grep -ohE "/[^ \"'=]*/target/(debug|release)(/[^ \"':]*)?" "$out" 2>/dev/null | sort -u)
done

if [ "${#stale[@]}" -eq 0 ]; then
    echo "target/ is consistent with $ROOT"
    exit 0
fi

# Same crate can appear once per recorded path.
readarray -t stale < <(printf '%s\n' "${stale[@]}" | sort -u)

echo "${#stale[@]} build directories still point at a path that no longer exists:"
printf '%s\n' "${stale[@]}" | sed "s|^$TARGET/|  |" | sed 's/-[0-9a-f]\{16\}$//' | sort -u | head -12
[ "${#stale[@]}" -gt 12 ] && echo "  …"

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "run scripts/clean-stale-target.sh to remove them" >&2
    exit 1
fi

for dir in "${stale[@]}"; do
    rm -rf "$dir"
done
echo "removed. Their build scripts will re-run on the next build."
