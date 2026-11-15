//! What every model is told about working with GhostReel: the one set of instructions an MCP
//! client reads at `initialize` and the script chat puts at the top of its system prompt. Whichever
//! model edits (a CLI agent over MCP, a server model, the local helper), it is told the same way to
//! work with the same tools; what differs between them is the transport, not the harness.

/// The harness instructions.
pub const INSTRUCTIONS: &str = "GhostReel turns indexed video into an edit. Each tool's description says what it \
does; these are the rules for using them to cut well.

HOW TO WORK
1. get_outline first: the videos, the people, every chapter with a summary. Decide the story there. Write \
the one argument the cut makes in one sentence; every clip moves it forward and the last one lands it. \
Strong lines that make a different point belong in a different cut.
2. Choose the spoken lines from moments: find_clip_candidates with min_s/max_s the length of one spoken beat \
(about 4-15 s for a 60 s cut) and the video_ids that carry the story. Code has already made each one whole \
sentences, and lines code has flagged as off the microphone are kept out; read_cut shows any it missed. You \
choose which ones say the story, in what order.
3. Read the pictures before you choose. A clip is a picture as well as words. get_manifest (on the video_ids \
you take moments from) shows, on one clock, who is on screen and how they are framed, what each shot looks \
like, and camera shake; and for b-roll, what each shot shows. Read the manifest of every video you take sound \
from before you save: save_script tells you when you did not. Then write each clip as its moment's \
{video_id, from: \"s412\", to: \"s418\"} with save_script (dry_run first). Cut points, breaths, holds and \
framing are placed by code on the measured speech and reported in `notes`, so do not nudge cuts by hand.
4. read_cut on the saved script, then read its words top to bottom as a viewer who never heard the \
episode. Fix every \"lo\" or \"eso\" that points at nothing, every clause cut in half, every clip that opens \
on a reply, and the story notes and issues it lists (apply_fix takes the typed fixes). Do not loop on in/out \
flags: they belong to code.
5. Render (preview_script, then export_script), then review_render: the words in the file, the silence \
at every cut, a contact sheet of frames. Fix what you hear or see, then review again.

EDITORIAL RULES
- A hook in the first 3 seconds: open on the most specific moment you have (a name, a story, something that \
happened), not the most general (\"I love the community\"). One idea per beat; end on a complete thought.
- Open on a line that stands alone, not on a reply (\"Exacto…\") or a connective that leans on something cut.
- Show the face when someone speaks, but do not hold one talking head for long: cut to b-roll of what they \
talk about while their voice carries on. A beat's `bed` lays the speech under muted b-roll (a J-cut or L-cut); \
the viewer sees who speaks first, so such a beat opens on the speaker (code puts their face before the b-roll). \
B-roll plays under a voice, never on its own sound.
- Use people's names when they are known (get_people). Say who speaks, and put the right face on screen \
for the voice.
- No screen share in a 9:16 cut unless the screen is the point. When a line is said over a share, carry it \
under the speaker's face with a bed, or frame their camera tile (apply_fix set_layout follow_speaker).
- Stay within the target length. Clip in_s/out_s must lie inside real footage.

COST
Text before pixels. The outline, manifest, transcript and read_cut are cheap. get_frames and renders cost \
tokens and time: use them only when the picture decides.";

#[cfg(test)]
mod tests {
    use super::INSTRUCTIONS;

    #[test]
    fn the_instructions_name_only_tools_that_exist() {
        let names = super::super::names();
        // Every `snake_case` word in the instructions that looks like a tool name is one.
        for word in INSTRUCTIONS.split(|c: char| !(c.is_ascii_lowercase() || c == '_')).filter(|w| w.contains('_')) {
            let known = names.contains(&word)
                || [
                    "dry_run",
                    "heard_as",
                    "in_s",
                    "out_s",
                    "start_s",
                    "end_s",
                    "script_id",
                    "video_id",
                    "fcp_xml",
                    "burn_captions",
                    "set_layout",
                    "video_ids",
                    "follow_speaker",
                    "min_s",
                    "max_s",
                ]
                .contains(&word);
            assert!(known, "the instructions mention {word}, which is not a tool");
        }
    }
}
