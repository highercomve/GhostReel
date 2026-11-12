#!/usr/bin/env bash
# Import the 8 hand-picked baseline clips (baseline/clips.json) into the podcast project, once
# its video is indexed. FORMAT=9:16 saves them vertical with follow-speaker framing + captions.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
G="${GHOSTREEL_BIN:-$repo/target/debug/ghostreel}"
P="${PROJECT:-Podcast clips test}"
FORMAT="${FORMAT:-original}"
DB="${GHOSTREEL_HOME:-$HOME/.ghostreel}/ghostreel.db"
vid="$(sqlite3 "$DB" "SELECT id FROM videos WHERE path LIKE '%agentes-de-ia-en-produccion%' ORDER BY id DESC LIMIT 1")"
[ -n "$vid" ] || { echo "video not indexed yet in $DB" >&2; exit 1; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
python3 -I - "$here/baseline/clips.json" "$vid" "$FORMAT" "$tmp" <<'PY'
import json, sys, os
clips, vid, fmt, out = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3], sys.argv[4]
for i, c in enumerate(clips, 1):
    s = {"title": f"Baseline {i}/8 · {c['title']}", "target_duration_s": round(c["out_s"] - c["in_s"], 1),
         "beats": [{"id": "b1", "purpose": c["why"], "narration": "", "on_screen_text": c["title"],
                    "clips": [{"video_id": vid, "in_s": c["in_s"], "out_s": c["out_s"], "audio": "source", "why": c["why"]}]}]}
    if fmt == "9:16":
        s.update(width=1080, height=1920, framing={"layout": "follow_speaker"}, captions={})
        s["title"] = s["title"].replace("Baseline", "Baseline 9:16", 1)
    json.dump(s, open(os.path.join(out, f"clip{i}.json"), "w"), ensure_ascii=False)
PY
for i in 1 2 3 4 5 6 7 8; do "$G" script import -p "$P" "$tmp/clip$i.json" 2>&1 | tail -1; done
