#!/usr/bin/env bash
# Recreate the "Podcast clips test" project used to develop and compare clip generation:
# the 70-min episode "Agentes de IA en producción" (O Corres O Te Encaramas), its glossary,
# and the 8 hand-picked baseline clips. Uses the normal data folder (~/.ghostreel, or
# GHOSTREEL_HOME) and the dev CLI build. Indexing itself is started separately (see the end).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
G="${GHOSTREEL_BIN:-$repo/target/debug/ghostreel}"
P="${PROJECT:-Podcast clips test}"
MEDIA="${MEDIA:-$HOME/Videos/GhostReel}"
URL="https://www.youtube.com/watch?v=JcGjXwc57gE"
FILE="$MEDIA/agentes-de-ia-en-produccion.mp4"

mkdir -p "$MEDIA"
if [ ! -f "$FILE" ]; then
  yt-dlp --remote-components ejs:github -f "bv*[height<=1080]+ba/b[height<=1080]" \
    --merge-output-format mp4 -o "$MEDIA/agentes-de-ia-en-produccion.%(ext)s" "$URL"
fi

# 1920x1080 @ 30 like the source; no camera-shake stage (a static podcast setup).
"$G" project list | grep -q "^$P " || "$G" project create "$P" --fps 30 --width 1920 --height 1080 \
  --stages probe,transcribe,frames,describe,embed,faces,speakers
"$G" folder add --project "$P" "$MEDIA" || true

# Glossary: term + how Whisper mishears it.
python3 -I - "$here/baseline/glossary.json" <<'PY' | while IFS=$'\t' read -r term heard; do
import json, sys
for t in json.load(open(sys.argv[1])):
    print(t["term"] + "\t" + "|".join(t["heard_as"]))
PY
  args=(); IFS='|' read -ra hs <<< "$heard"; for h in "${hs[@]}"; do [ -n "$h" ] && args+=(--heard-as "$h"); done
  "$G" project glossary "$P" --add "$term" "${args[@]}" >/dev/null
done
echo "glossary: $("$G" project glossary "$P" | grep -c . ) lines"
echo
echo "Next: $G index -p \"$P\"   (then: $G speakers -p \"$P\"), then $here/import-baseline.sh"
