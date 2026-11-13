//! What every model is told about working with GhostReel: the one set of instructions an MCP
//! client reads at `initialize` and the script chat puts at the top of its system prompt. Whichever
//! model edits (a CLI agent over MCP, a server model, the local helper), it is told the same way to
//! work with the same tools; what differs between them is the transport, not the harness.

/// The harness instructions.
pub const INSTRUCTIONS: &str = "GhostReel indexes folders of video (speech transcripts, speech activity, chapters, \
keyframe descriptions, embeddings) and turns clips into an edit. Typical flow: create_project → add_folder \
→ get_glossary / set_glossary (terms Whisper should spell correctly) → index (repeat until it \
reports no pending work) → get_outline → read what matters → write the script with save_script → read_cut → \
preview_script / export_script.

A way to work that is cheap and fast, and still yours to change:
1. get_outline first. One call returns the videos, the people, every video's chapters (range, title, \
a summary of what is talked about) and the glossary. Decide the story there, before reading \
everything: the one argument the cut makes and the 2-4 chapters that carry it. Write the argument down in one \
sentence before you choose a clip: every clip should move it forward, and the last one should land it. Strong \
lines that make a different point are a different cut.
2. get_manifest for those chapters only (start_s/end_s of a chapter): every measured fact on one clock — \
each sentence with its speaker and the silence before and after it, each change of picture with the faces on \
screen (position, size, cut off), keyframe descriptions and on-screen text, camera shake. It is measurements, \
not verdicts: you decide what is usable. get_transcript is the words alone, 's412 [12:03.4 Freddy] words', \
when that is all you need. A sentence ends at a full stop, a change of speaker or a real pause, so its id is \
a whole thought you can cut on.
3. Write the clips by sentence id: {video_id, from: \"s412\", to: \"s418\"} (\"s412@3\" for the third word, \
where the video has word timings) instead of seconds, with save_script (dry_run first). Open a clip on a \
line that stands alone, not on a reply (\"Exacto…\") or a connective that leans on a line the viewer never \
hears. You choose the sentences; the exact cut points, breaths, holds and framing are placed by code on the \
measured speech and come back in `notes`, so do not add holds or nudge a cut to dodge the next speaker. \
in_s/out_s still work for pictures. A beat's `bed` lays an interview's audio under muted b-roll (a J-cut or \
L-cut); it also carries a line said over a screen share under the same speaker's face from a nearby stretch. \
Or keep the moment and frame the speaker's own camera tile over the share (apply_fix set_layout follow_speaker on \
that stretch, as the screen-share issue offers): their lips stay in sync, zoomed from a small tile.
4. read_cut once on the saved script_id (or a draft) to check it as text without rendering: the words heard, \
who speaks and who is on screen, a 'story' line when a clip starts or ends mid-sentence, opens on a reply \
or ends on an unanswered question, the total against the target, and issues with typed `fixes` (apply_fix). \
Then read its words top to bottom as a viewer who never heard the episode: every clip has to make sense \
without what you cut around it — a \"lo\" or \"eso\" with nothing it points to, a clause cut in half, a hole inside \
a sentence are what a viewer trips on. Fix those, the story notes and the issues. The in/out flags on each cut are code's job: save again rather than \
moving cuts by hand, and do not loop on them. captions: true lists the captions an mp4 burns in, with the \
src ranges a script's caption fixes take.
5. preview_script (light, 540p) only when the reading is clean; export_script with format mp4 for the \
finished full-size video (burn_captions: true to burn the speech captions in), or fcp_xml / otio for an editor.
6. review_render on what you rendered, before you call it done: the words heard in the file itself, the silence \
at every cut and a contact sheet of the frames — what a viewer gets. Fix what you hear or see, render, review again.

Editorial rules of thumb: a hook in the first 3 seconds; one idea per beat; end on a complete thought; \
show the face when someone speaks; carry the voice through a cutaway with a bed; no screen share in a \
9:16 cut unless the screen is the point.

Cost: text before pictures. get_outline, list_videos, get_transcript, get_manifest, search, get_video, get_scenes, \
get_steadiness and read_cut are cheap reads; get_frames (the picture itself, exact at the time you give) \
and previews cost tokens and time, so use them only when the picture decides.

The rest of the toolbox: list_videos gives the library at a glance (interviews vs b-roll, speech, camera), \
search finds moments by what is said or visible, get_video lists a clip's keyframe descriptions, \
get_steadiness tells you whether a stretch is shaky and where the nearest steady one is; get_faces finds \
face tracks (bounding boxes, framing, whether cut off, person); get_people lists who appears (name, role, \
screen and speaking time) and set_person / merge_people name them or join duplicates; get_speakers says \
who talks when; get_scenes says which stretches are a two-shot, a single face, a screen share or empty \
(a screen share of three seconds or more is always listed); get_glossary and set_glossary manage \
transcription vocabulary (heard_as are known mis-hearings). draft_script instead hands the whole edit to \
GhostReel's local chat model, which needs that model to fit on this machine. Clip in_s/out_s must lie \
inside real footage. save_script does the deterministic work itself (a vertical clip is trimmed off a \
screen share at sentence edges, its closing hold kept on a face, cuts placed on the sound) and says what \
it changed in `notes`; everything else comes back as issues, many with typed `fixes` you can apply with \
apply_fix. Issues are computed from the index as it is now (get_script returns them too), so they never \
go stale.

Short social clips: generate_clips runs the whole pipeline with GhostReel's own models; or \
orchestrate it yourself — find_clip_candidates (sentence-bounded windows with a visual label), \
read the transcript around the ones you like, check_clips, then save_clips to store them as a \
clip set; list_clip_sets shows the sets.

Tools that change things say so in their annotations (readOnlyHint false); delete_project and \
merge_people are marked destructive and cannot be undone.";

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
                    "follow_speaker",
                ]
                .contains(&word);
            assert!(known, "the instructions mention {word}, which is not a tool");
        }
    }
}
