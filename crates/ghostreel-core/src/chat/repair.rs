//! Turning a draft into a finished script, outside the turn that produced it.
//!
//! The passes themselves are `crate::finish`, shared with every other way a cut is saved; what is
//! the chat's own here is the review a turn runs on its editor's cut before saving it.

use serde_json::Value;

use crate::Error;
use crate::db::Db;
use crate::projects::Project;
use crate::script::{Issue, IssueSeverity, Script, validate};

/// Everything applied once narration exists, up to but not including saving: the finishing chain
/// every cut goes through ([`crate::finish::run`], a chat turn's profile), then the checks a turn
/// reports to the editor.
///
/// Returns `None` when there is nothing left to finish — every clip dropped — which is the turn's
/// "unable to assemble a script" path.
pub fn finish_script(
    db: &Db,
    project_id: i64,
    s: &mut Script,
    enforce_target: bool,
    cfg: &crate::config::ScriptConfig,
) -> Result<Option<Vec<Issue>>, Error> {
    if s.clip_count() == 0 || s.beats.is_empty() {
        return Ok(None);
    }
    let notes = crate::finish::run(db, project_id, s, cfg, &crate::finish::Profile::chat(enforce_target))?;
    // The passes can drop every clip (a foreign-video draft, pacing against a tight target), and
    // an empty cut must not be handed back to save. finish::run leaves no empty beats.
    if s.clip_count() == 0 || s.beats.is_empty() {
        return Ok(None);
    }
    let mut issues: Vec<Issue> = notes
        .into_iter()
        .map(|n| Issue {
            fixes: Vec::new(),
            severity: IssueSeverity::Info,
            beat_id: None,
            clip_index: None,
            message: n.message,
        })
        .collect();
    issues.extend(super::content_issues(db, s, cfg));
    issues.extend(validate(db, project_id, s)?);
    Ok(Some(issues))
}

/// How many times a turn hands its own cut back to the editor before it is saved.
pub const MAX_REVIEWS: usize = 2;

/// What a turn asked for, which every draft it reviews is held to: the project and its shape, the
/// footage the turn has seen, the requested length, and the cut being revised.
pub(crate) struct Rules<'a> {
    pub project: &'a Project,
    pub grounding: &'a super::Grounding,
    pub cfg: &'a crate::config::ScriptConfig,
    /// Squeeze the cut to its target (a first draft, or a length the user stated).
    pub enforce_target: bool,
    /// The length the user stated, which wins over whatever the editor wrote.
    pub requested_s: Option<f64>,
    /// The format the brief chose, when it chose one.
    pub requested_fmt: Option<crate::clips::ClipFormat>,
    /// The version this turn revises: its title, agents and shape are kept.
    pub baseline: Option<&'a Script>,
    /// How a finished cut is read back for the editor ([`read_back`]; a test can make it fail).
    pub read: Reader,
}

/// Reads a finished cut back: how many things in it still need fixing, and, with a `round` to
/// spend, the note that tells the editor what they are.
pub(crate) type Reader = fn(&Db, i64, &Script, &crate::config::ScriptConfig, Option<usize>) -> Result<Reading, Error>;

/// A finished cut read back ([`read_back`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Reading {
    /// Clips and beds with a story line, plus warnings and errors from the check.
    pub flags: usize,
    /// What the editor still has to fix, when there is something and a round to ask in.
    pub note: Option<String>,
}

/// How far over its target a draft may run, as written, before a revision that runs that long
/// counts as a different, longer draft rather than a fix (the finish would have to drop beats).
const LONGER_DRAFT: f64 = 1.25;

/// What an editor handed back in one round, in whatever form its backend delivers it.
// One answer per round: boxing the script would only add noise.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Answer {
    /// Text that should hold the script: a script JSON object, a `{"action":"final",…}` action, or
    /// either one wrapped in prose.
    Text(String),
    /// A script already parsed out of an action, not yet filled in or resolved.
    Script(Script),
    /// The version the editor saved itself (save_script, apply_fix).
    Saved(i64),
    /// An answer already made ready ([`normalise`]): what a backend that checks the draft before
    /// the review (a redraft) passes on.
    Ready(Ready),
    /// No answer: the backend failed or gave up. The cut under review stands.
    Nothing,
    /// The user stopped the turn.
    Cancelled,
}

/// An answer made ready for review: one draft, or the reason there is none.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Ready {
    Draft(Draft),
    /// No usable script in the answer, and why: the reason goes back to the editor.
    Unusable {
        why: String,
        text: String,
    },
    Reply(String),
    Nothing,
    Cancelled,
}

/// A draft as every backend's answer becomes one: filled from the project, its sentences resolved
/// to seconds, the requested length set.
pub(crate) struct Draft {
    pub script: Script,
    /// What resolving it changed (a clip moved to its sentences' video), for the editor.
    pub notes: Vec<String>,
    /// The version it came from, when the editor saved it itself.
    pub saved: Option<i64>,
}

/// The script the editor last gave, as the backend replays it with a review note.
pub(crate) enum Replay<'a> {
    Script(&'a Script),
    /// An answer that held no usable script, as it was written.
    Text(&'a str),
}

/// How a turn's backend talks to its editor during review. The review itself is the same for every
/// backend ([`deliver`]); only sending the note and getting the answer back differ.
pub(crate) trait Transport {
    /// Send `note` about the script the editor last gave (`replay`) and wait for its next answer.
    async fn ask(&mut self, note: &str, replay: Replay<'_>) -> Answer;

    /// What the backend adds to each draft before it is finished: a local model's narration for
    /// the silent beats. Nothing by default.
    async fn fill(&mut self, _db: &mut Db, _draft: &mut Script) -> Vec<Issue> {
        Vec::new()
    }
}

/// What a turn delivers once its review is over.
#[derive(Default)]
pub(crate) struct Delivered {
    /// The cut to save, finished exactly as it was reviewed. `None` when there is none.
    pub script: Option<Script>,
    /// What was changed or found on the way (resolving, the draft repairs, the finish, the checks).
    pub issues: Vec<Issue>,
    /// The version the delivered draft came from, when the editor saved it itself.
    pub saved: Option<i64>,
    /// The editor's answer in words, when it gave one instead of a script.
    pub reply: Option<String>,
    /// Why no script could be used, when none could.
    pub error: Option<String>,
    /// How many review notes went back to the editor.
    pub reviews: usize,
}

fn info(message: String) -> Issue {
    Issue { fixes: Vec::new(), severity: IssueSeverity::Info, beat_id: None, clip_index: None, message }
}

/// The script an answer holds, wherever it is in it: a script object, a final action, or either
/// one wrapped in prose. A reply is a reply; anything else is unusable, and the error says why.
fn script_in(text: &str) -> Result<Result<Script, String>, String> {
    let trimmed = text.trim();
    let mut candidates = vec![trimmed];
    if let (Some(a), Some(b)) = (trimmed.find('{'), trimmed.rfind('}'))
        && (a > 0 || b + 1 < trimmed.len())
        && a < b
    {
        candidates.push(&trimmed[a..=b]);
    }
    let mut why = None;
    for c in candidates {
        let Ok(v) = serde_json::from_str::<Value>(c) else { continue };
        if v.get("action").is_some() {
            match serde_json::from_value::<super::LocalAction>(v) {
                Ok(super::LocalAction::Final { script }) => return Ok(Ok(script)),
                Ok(super::LocalAction::Reply { text }) => return Ok(Err(text)),
                Ok(super::LocalAction::Tool { tool, .. }) => {
                    why.get_or_insert(format!("you asked for the tool {tool} instead of answering with the script"));
                }
                Err(e) => {
                    why.get_or_insert(format!("the script does not match the schema: {e}"));
                }
            }
            continue;
        }
        match serde_json::from_value::<Script>(v) {
            Ok(script) => return Ok(Ok(script)),
            Err(e) => {
                why.get_or_insert(format!("the script does not match the schema: {e}"));
            }
        }
    }
    Err(why.unwrap_or_else(|| {
        let shown: String = trimmed.chars().take(200).collect();
        format!("the answer holds no script JSON: {shown}")
    }))
}

/// Whether an answer holds a script anywhere in it ([`script_in`]), as opposed to words.
pub(crate) fn holds_script(text: &str) -> bool {
    matches!(script_in(text), Ok(Ok(_)))
}

/// Make an answer ready for review, the same way whichever backend delivered it.
pub(crate) fn normalise(db: &Db, rules: &Rules<'_>, answer: Answer) -> Ready {
    let ready = |mut script: Script, saved: Option<i64>, text: String| {
        script.fill_from_project(rules.project);
        match crate::sentences::resolve_script_noting(db, &mut script) {
            Ok((_, notes)) => {
                if rules.requested_s.is_some() {
                    script.target_duration_s = rules.requested_s;
                }
                Ready::Draft(Draft { script, notes, saved })
            }
            Err(e) => Ready::Unusable { why: e.to_string(), text },
        }
    };
    match answer {
        Answer::Text(text) => match script_in(&text) {
            Ok(Ok(script)) => ready(script, None, text),
            Ok(Err(reply)) => Ready::Reply(reply),
            Err(why) => Ready::Unusable { why, text },
        },
        Answer::Script(script) => {
            let text = serde_json::to_string(&script).unwrap_or_default();
            ready(script, None, text)
        }
        Answer::Saved(id) => match crate::script::load(db, id) {
            Ok(stored) => {
                let text = serde_json::to_string(&stored.script).unwrap_or_default();
                ready(stored.script, Some(id), text)
            }
            Err(e) => Ready::Unusable {
                why: format!("the version you saved (#{id}) could not be read: {e}"),
                text: String::new(),
            },
        },
        Answer::Ready(ready) => ready,
        Answer::Nothing => Ready::Nothing,
        Answer::Cancelled => Ready::Cancelled,
    }
}

/// A draft finished exactly as it will be saved, what was done to it, and what the editor still
/// has to fix in it (`None` when it reads clean, or when no rounds are left to ask).
pub struct Reviewed {
    pub finished: Script,
    pub issues: Vec<Issue>,
    pub note: Option<String>,
    /// How many things the reading still flags (`None` when it could not be read back).
    pub flags: Option<usize>,
}

/// Finish `draft` the way it will be saved — the turn's shape, the draft repairs, the finishing
/// chain, the revised cut's title — then, with a `round` to spend, read it back for the editor.
/// `None` when nothing is left once it is finished (every clip dropped). The same review whichever
/// model wrote the cut: code finishes it, code reads it, the editor decides what to change.
///
/// A failure to read it back does not lose the cut: it is logged, and the cut stands as finished.
pub(crate) fn review(
    db: &Db,
    rules: &Rules<'_>,
    draft: &Script,
    round: Option<usize>,
) -> Result<Option<Reviewed>, Error> {
    review_from(db, rules, draft, round, false)
}

/// [`review`], knowing whether the cut is a version the editor saved itself (`saved`). A saved
/// version was finished when it was saved: the draft-only squeeze toward the target, which trims
/// pictures in proportion so a model hears about a cut far too long, is not run on it.
pub(crate) fn review_from(
    db: &Db,
    rules: &Rules<'_>,
    draft: &Script,
    round: Option<usize>,
    saved: bool,
) -> Result<Option<Reviewed>, Error> {
    let project_id = rules.project.id;
    let mut s = draft.clone();
    if let Some(shape) = super::turn_shape(rules.project, rules.requested_fmt, rules.baseline, &s) {
        super::apply_shape(db, &mut s, shape, rules.cfg);
    }
    let squeeze = rules.enforce_target && !saved;
    let mut issues = super::enforce_grounding_and_pacing(db, project_id, &mut s, rules.grounding, squeeze, rules.cfg);
    let Some(more) = finish_script(db, project_id, &mut s, rules.enforce_target, rules.cfg)? else {
        return Ok(None);
    };
    issues.extend(more);
    // A revision keeps its cut's title, so the AI cannot split one cut into two by renaming it, and
    // the cut's agent choice, which is the cut's and not the model's to drop.
    if let Some(base) = rules.baseline {
        s.title = base.title.clone();
        if s.agents.is_none() {
            s.agents = base.agents.clone();
        }
    }
    // Read even with no round left: the last revision is compared with the cut it revised.
    let (note, flags) = match (rules.read)(db, project_id, &s, rules.cfg, round) {
        Ok(reading) => (reading.note, Some(reading.flags)),
        Err(e) => {
            tracing::warn!("review: reading the cut back failed, keeping it as finished: {e}");
            crate::llmlog::push("error", format!("Review read-back: {e}"));
            (None, None)
        }
    };
    Ok(Some(Reviewed { finished: s, issues, note, flags }))
}

/// A draft finished exactly as a chat turn's review finishes it before saving — the draft rules
/// that need the turn (grounding, with everything in `grounding` legal), the shape the project
/// gives it and the finishing chain — without reading it back. `None` when nothing is left. For
/// comparing a turn's cut with what another editor's save makes of the same draft.
pub fn finish_as_reviewed(
    db: &Db,
    project: &Project,
    draft: &Script,
    grounding: &super::Grounding,
    enforce_target: bool,
    cfg: &crate::config::ScriptConfig,
) -> Result<Option<Script>, Error> {
    let rules = Rules {
        project,
        grounding,
        cfg,
        enforce_target,
        requested_s: None,
        requested_fmt: None,
        baseline: None,
        read: read_back,
    };
    Ok(review(db, &rules, draft, None)?.map(|r| r.finished))
}

/// The editor's finished cut read back (`read_cut`): how many things still need fixing — a clip
/// that starts or ends inside a thought, opens on a reply, plays someone off the microphone, or a
/// warning from the check — and, with a `round` to spend, the note for the editor. No note when it
/// reads clean.
pub(crate) fn read_back(
    db: &Db,
    project_id: i64,
    finished: &Script,
    cfg: &crate::config::ScriptConfig,
    round: Option<usize>,
) -> Result<Reading, Error> {
    let opts = crate::script_check::CheckOptions { pictures: true, ..Default::default() };
    let issues = crate::script_check::check(db, project_id, finished, cfg, &opts)?;
    let reading = crate::readcut::read_cut(db, project_id, finished, cfg, issues)?;
    let story = reading
        .beats
        .iter()
        .map(|b| {
            b.clips.iter().filter(|c| !c.story.is_empty()).count()
                + usize::from(b.bed.as_ref().is_some_and(|bed| !bed.story.is_empty()))
        })
        .sum::<usize>();
    let warned =
        reading.issues.iter().filter(|i| matches!(i.severity, IssueSeverity::Warning | IssueSeverity::Error)).count();
    let flags = story + warned;
    let Some(round) = round.filter(|_| flags > 0) else {
        return Ok(Reading { flags, note: None });
    };
    let note = Some(format!(
        "REVIEW {round} of {MAX_REVIEWS}: your cut, finished the way it will be saved and read back.\n{}\n\
         Read it as a viewer who never heard the interviews. Fix every 'story' line and every warning: choose another \
         moment or other sentences, reorder, or drop a clip. Then answer with the complete corrected final script. \
         If a flag cannot be fixed with this footage, answer with the same script.",
        reading.render()
    ));
    Ok(Reading { flags, note })
}

/// A cut as the review compares it: what its reading flags (`None` when it could not be read), how
/// long it runs as written, and how long it plays once finished.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Standing {
    pub flags: Option<usize>,
    pub draft_s: f64,
    pub finished_s: f64,
}

/// How much nearer its target a revision with as many flags must land to replace the cut (s).
const NEARER_S: f64 = 2.0;

/// Why a revision is not used in place of the cut it revised (`None` when it is better).
///
/// A review exists to fix what it flagged, so a revision has to read cleaner than the cut it was
/// asked to fix, or, flagging as much, land clearly nearer the target: otherwise it is no
/// improvement, and the reviewed cut — which may be the version the editor saved, watched and
/// exported — stands. Neither is a draft that runs far past the target where the reviewed one sat
/// near it: that is an older or different draft pasted back, which only reads short because the
/// finish dropped beats from it. A reviewed cut half its target long is not one to protect that way.
fn worse_revision(reviewed: Standing, revision: Standing, target_s: Option<f64>) -> Option<String> {
    let before = reviewed.flags?;
    let (draft_before, draft_after) = (reviewed.draft_s, revision.draft_s);
    let target = target_s.filter(|t| *t > 0.0);
    if let Some(t) = target
        && draft_after > t * LONGER_DRAFT
        && draft_before <= t * LONGER_DRAFT
        && draft_before >= t / LONGER_DRAFT
    {
        return Some(format!(
            "it is a different, longer draft ({draft_after:.0} s as written for a {t:.0} s target), not a fix \
             of the clips the review flagged"
        ));
    }
    let miss = |s: &Standing| target.map_or(0.0, |t| (s.finished_s - t).abs());
    match revision.flags {
        None => Some("it could not be read back".into()),
        Some(after) if after > before => {
            Some(format!("it reads worse than the cut it revised ({after} flag(s) against {before})"))
        }
        Some(after) if after == before && miss(&revision) > miss(&reviewed) - NEARER_S => Some(format!(
            "it reads no cleaner than the cut it revised ({after} flag(s) against {before}) and lands no nearer \
             its target"
        )),
        _ => None,
    }
}

/// The review every turn runs on what its editor delivered, whichever backend it is and however
/// the answer came (a script in its final answer, one wrapped in prose, the draft a turn forced at
/// the end, or the version the editor saved itself): made ready ([`normalise`]), finished and read
/// back ([`review`]), and what still needs fixing — or why the script could not be used — sent
/// back through `transport`, at most [`MAX_REVIEWS`] times. What comes out is the last usable cut,
/// finished exactly as it was reviewed, so the version saved is the version read.
///
/// A revision that is unusable, or no answer at all, leaves the usable cut standing; so does a
/// failure to read a cut back, or to finish a revision of it.
pub(crate) async fn deliver<T: Transport>(
    db: &mut Db,
    rules: &Rules<'_>,
    transport: &mut T,
    first: Answer,
) -> Result<Delivered, Error> {
    let mut out = Delivered::default();
    let mut ready = normalise(db, rules, first);
    // The last usable draft as the editor wrote it (to replay), and its finished cut.
    let mut current: Option<Script> = None;
    // The cut under review: its flags, its length as written, and the note sent about it.
    let mut standing: Option<(Standing, String)> = None;
    let mut unusable = String::new();
    let mut why: Option<String> = None;
    loop {
        let note = match ready {
            Ready::Draft(Draft { mut script, notes, saved }) => {
                let mut issues: Vec<Issue> = notes.into_iter().map(info).collect();
                issues.extend(transport.fill(db, &mut script).await);
                let round = (out.reviews < MAX_REVIEWS).then_some(out.reviews + 1);
                let draft_s = script.length_s();
                match review_from(db, rules, &script, round, saved.is_some()) {
                    Ok(Some(reviewed)) => {
                        // A revision that is no better leaves the reviewed cut standing; the editor
                        // hears why while it still has rounds, and is asked again for a fix.
                        let this =
                            Standing { flags: reviewed.flags, draft_s, finished_s: reviewed.finished.length_s() };
                        if let Some((before, last_note)) = &standing
                            && let Some(reason) = worse_revision(*before, this, reviewed.finished.target_duration_s)
                        {
                            crate::llmlog::push(
                                "info",
                                format!("Review: kept the reviewed cut: the revision {reason}"),
                            );
                            if out.reviews >= MAX_REVIEWS {
                                break;
                            }
                            let body = last_note.split_once('\n').map_or(last_note.as_str(), |(_, rest)| rest);
                            let note = format!(
                                "REVIEW {} of {MAX_REVIEWS}: your revision was not used: {reason}. The cut below \
                                 stands. Change only the clips its story lines and warnings name, and keep the \
                                 rest as it is.\n{body}",
                                out.reviews + 1
                            );
                            out.reviews += 1;
                            let replay = Replay::Script(current.as_ref().expect("a standing cut has its draft"));
                            let answer = transport.ask(&note, replay).await;
                            ready = normalise(db, rules, answer);
                            continue;
                        }
                        issues.extend(reviewed.issues);
                        out.script = Some(reviewed.finished);
                        out.issues = issues;
                        out.saved = saved;
                        why = None;
                        current = Some(script);
                        match reviewed.note {
                            Some(note) => {
                                standing = Some((this, note.clone()));
                                note
                            }
                            None => break,
                        }
                    }
                    // Nothing left once finished: an earlier cut stands, if there is one.
                    Ok(None) => break,
                    Err(e) if out.script.is_some() => {
                        tracing::warn!("review: finishing a revision failed, keeping the reviewed cut: {e}");
                        crate::llmlog::push("error", format!("Review: {e}"));
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
            Ready::Unusable { why: reason, text } => {
                // A revision that breaks a usable cut is ignored: the usable one stands.
                if current.is_some() {
                    break;
                }
                why = Some(reason.clone());
                if out.reviews >= MAX_REVIEWS {
                    break;
                }
                unusable = text;
                format!(
                    "REVIEW {} of {MAX_REVIEWS}: your script could not be used: {reason}\n\
                     Fix that and return the complete Script JSON.",
                    out.reviews + 1
                )
            }
            Ready::Reply(text) => {
                if out.script.is_none() {
                    out.reply = Some(text);
                }
                break;
            }
            Ready::Nothing => break,
            Ready::Cancelled => return Err(Error::Invalid(super::CANCELLED.into())),
        };
        out.reviews += 1;
        let replay = match &current {
            Some(s) => Replay::Script(s),
            None => Replay::Text(&unusable),
        };
        let answer = transport.ask(&note, replay).await;
        ready = normalise(db, rules, answer);
    }
    if out.script.is_none() {
        out.error = why;
    }
    Ok(out)
}

/// Save a delivered cut as this session's version. The version the editor saved itself is kept as
/// it is when the review changed nothing in it; otherwise the finished cut is a new version.
pub(crate) fn save(db: &Db, project_id: i64, session_id: i64, s: &Script, saved: Option<i64>) -> Result<i64, Error> {
    if let Some(id) = saved
        && crate::script::load(db, id).is_ok_and(|stored| stored.script == *s)
    {
        db.conn.execute("UPDATE scripts SET session_id = ?1 WHERE id = ?2", rusqlite::params![session_id, id])?;
        return Ok(id);
    }
    crate::script::save_version(db, project_id, s, Some(session_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    /// One interview: two answers with the interviewer's question (off-mic) between them.
    fn project_with_interview() -> (Db, i64) {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("P")).unwrap();
        let tmp = std::env::temp_dir().join("ghostreel-review-fixture");
        std::fs::create_dir_all(&tmp).unwrap();
        let folder = db.add_folder(project.id, &tmp, true).unwrap();
        db.conn
            .execute("INSERT INTO videos(id, content_hash, size, duration_s, category) VALUES (7, 'h', 1, 60.0, 'interview')", [])
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen) VALUES (7, ?1, '/m/kara.mp4', 1, 0, 0)",
                [folder.id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO transcript_segments(video_id, start_s, end_s, text, off_mic) VALUES
                 (7, 1.0, 6.0, 'Every event we have come to, we walked away with a new relationship.', 0),
                 (7, 6.5, 8.5, 'What made you want to sponsor the bar?', 1),
                 (7, 9.0, 14.0, 'We love this hotel and the people who come here every month.', 0)",
                [],
            )
            .unwrap();
        crate::sentence_split::mirror_segments(&db, 7);
        (db, project.id)
    }

    fn cut(in_s: f64, out_s: f64) -> Script {
        serde_json::from_value(serde_json::json!({
            "title": "t",
            "beats": [{ "id": "b1", "purpose": "p", "narration": "",
                        "clips": [{ "video_id": 7, "in_s": in_s, "out_s": out_s, "audio": "source" }] }]
        }))
        .unwrap()
    }

    fn rules<'a>(
        project: &'a Project,
        g: &'a super::super::Grounding,
        cfg: &'a crate::config::ScriptConfig,
    ) -> Rules<'a> {
        Rules {
            project,
            grounding: g,
            cfg,
            enforce_target: false,
            requested_s: None,
            requested_fmt: None,
            baseline: None,
            read: read_back,
        }
    }

    #[test]
    fn a_cut_that_still_needs_fixing_goes_back_to_the_editor_and_a_clean_one_does_not() {
        let (db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let r = rules(&project, &g, &cfg);
        // The question in the middle: code cannot take it out without a jump cut.
        let reviewed = review(&db, &r, &cut(1.0, 14.0), Some(1)).unwrap().unwrap();
        let note = reviewed.note.expect("the off-mic question goes back to the editor");
        assert!(note.starts_with("REVIEW 1 of 2"), "{note}");
        assert!(note.contains("off-mic"), "{note}");
        // One whole answer: nothing to fix, so the turn saves it.
        assert_eq!(review(&db, &r, &cut(1.0, 6.0), Some(1)).unwrap().unwrap().note, None);
        // No round left to spend: finished, not read.
        assert_eq!(review(&db, &r, &cut(1.0, 14.0), None).unwrap().unwrap().note, None);
    }

    /// The finishing passes can drop every clip — a draft of footage outside the project, pacing
    /// against a tight target — and an empty cut must not come out as something to save. Before,
    /// the emptiness check ran only before the passes: the empty cut came back as `Some`, was
    /// read back and saved. Beats left with no clips are dropped too, the way every pass treats
    /// them, so the caller's earlier usable draft stays.
    #[test]
    fn a_cut_the_finish_empties_returns_none() {
        let (db, project_id) = project_with_interview();
        let cfg = crate::config::ScriptConfig::default();
        let beat = |id: &str, video_id: i64| {
            serde_json::json!({
                "id": id, "purpose": "p", "narration": "",
                "clips": [{ "video_id": video_id, "in_s": 1.0, "out_s": 6.0, "audio": "mute" }]
            })
        };
        // Every clip foreign: nothing left once the draft pass drops them.
        let mut empty: Script = serde_json::from_value(serde_json::json!({
            "title": "t", "beats": [beat("b1", 999)]
        }))
        .unwrap();
        assert_eq!(finish_script(&db, project_id, &mut empty, false, &cfg).unwrap(), None);
        assert!(empty.beats.is_empty(), "the emptied beat is gone too");
        // One foreign beat beside a usable one: the emptiness keeps the cut, but the beat left
        // with no clips does not survive as a husk.
        let mut mixed: Script = serde_json::from_value(serde_json::json!({
            "title": "t", "beats": [beat("b1", 999), beat("b2", 7)]
        }))
        .unwrap();
        let issues = finish_script(&db, project_id, &mut mixed, false, &cfg).unwrap().expect("the cut stands");
        assert_eq!(mixed.beats.len(), 1, "the beat with no clips is dropped");
        assert_eq!(mixed.beats[0].id, "b2");
        assert!(!issues.is_empty());
    }

    /// A version the editor saved itself is a composed edit: two shots of one take side by side,
    /// one louder or on another track, or one heard and the next not, are choices, not one clip
    /// split while looking. The review's draft clean-up joined them, keeping only the first's sound
    /// and promoting the silent half to sound.
    #[tokio::test]
    async fn a_saved_cut_keeps_adjacent_clips_that_play_differently() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        // Beds are not what this is about: with one inferred, every clip would go silent under it.
        let cfg = crate::config::ScriptConfig { infer_audio_beds: false, ..Default::default() };
        // The editor looked at all of it.
        let mut g = super::super::Grounding::default();
        g.add(7, 0.0, 60.0);
        let mut stored: Script = serde_json::from_value(serde_json::json!({
            "title": "t",
            "beats": [
                { "id": "b0", "purpose": "p", "narration": "", "clips": [
                    { "video_id": 7, "in_s": 20.0, "out_s": 24.0, "audio": "mute", "gain_db": -6.0 },
                    { "video_id": 7, "in_s": 24.0, "out_s": 28.0, "audio": "mute", "tracks": [1] } ] },
                { "id": "b1", "purpose": "p", "narration": "", "clips": [
                    { "video_id": 7, "in_s": 9.0, "out_s": 14.0, "audio": "source", "exact": true },
                    { "video_id": 7, "in_s": 14.0, "out_s": 17.0, "audio": "mute" } ] }
            ]
        }))
        .unwrap();
        stored.fill_from_project(&project);
        let id = crate::script::save_version(&db, project_id, &stored, None).unwrap();
        let r = rules(&project, &g, &cfg);
        let mut editor = Scripted::new(Vec::new());
        let out = deliver(&mut db, &r, &mut editor, Answer::Saved(id)).await.unwrap();
        let finished = out.script.expect("a cut");
        assert_eq!(finished, crate::script::load(&db, id).unwrap().script, "{:?}", out.issues);
        assert_eq!(out.saved, Some(id));
    }

    /// A backend as the review sees it: each note is written down, and the next answer is the next
    /// one in line (none once they run out, as a backend that failed).
    struct Scripted {
        answers: std::collections::VecDeque<Answer>,
        notes: Vec<String>,
        replays: Vec<String>,
    }

    impl Scripted {
        fn new(answers: Vec<Answer>) -> Self {
            Self { answers: answers.into(), notes: Vec::new(), replays: Vec::new() }
        }
    }

    impl Transport for Scripted {
        async fn ask(&mut self, note: &str, replay: Replay<'_>) -> Answer {
            self.notes.push(note.to_string());
            self.replays.push(match replay {
                Replay::Script(s) => serde_json::to_string(s).unwrap(),
                Replay::Text(t) => t.to_string(),
            });
            self.answers.pop_front().unwrap_or(Answer::Nothing)
        }
    }

    fn json_of(s: &Script) -> String {
        serde_json::to_string(s).unwrap()
    }

    fn action_of(s: &Script) -> String {
        serde_json::to_string(&super::super::LocalAction::Final { script: s.clone() }).unwrap()
    }

    /// The spoken clip of a one-beat cut (the finish may add a held closing picture after it).
    fn the_clip(s: &Script) -> (f64, f64) {
        let spoken: Vec<_> =
            s.beats.iter().flat_map(|b| &b.clips).filter(|c| c.audio == crate::script::Audio::Source).collect();
        assert_eq!(spoken.len(), 1, "{s:?}");
        (spoken[0].in_s, spoken[0].out_s)
    }

    /// Every way an editor's cut reaches the turn — a server's script JSON, a CLI's or local
    /// model's final action, the same wrapped in prose, an action already parsed, the version the
    /// editor saved itself — is reviewed the same way: one note, one revision, the revision saved.
    /// Before, a CLI's prose answer, a forced final draft and a tool-saved script skipped review.
    #[tokio::test]
    async fn every_form_an_answer_comes_in_is_reviewed_the_same_way() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let bad = cut(1.0, 14.0);
        let clean = cut(1.0, 6.0);
        let saved = crate::script::save_version(&db, project_id, &bad, None).unwrap();
        let forms: Vec<(&str, Answer)> = vec![
            ("server script JSON", Answer::Text(json_of(&bad))),
            ("final action", Answer::Text(action_of(&bad))),
            ("action in prose", Answer::Text(format!("Here is the cut:\n{}\nHope it works.", action_of(&bad)))),
            ("script in prose", Answer::Text(format!("Done. {} — that's it.", json_of(&bad)))),
            ("parsed action", Answer::Script(bad.clone())),
            ("saved by the editor", Answer::Saved(saved)),
        ];
        let mut finished = Vec::new();
        for (form, first) in forms {
            let r = rules(&project, &g, &cfg);
            let mut editor = Scripted::new(vec![Answer::Text(action_of(&clean))]);
            let out = deliver(&mut db, &r, &mut editor, first).await.unwrap();
            assert_eq!(editor.notes.len(), 1, "{form}: one note, then the clean revision reads clean");
            assert!(editor.notes[0].starts_with("REVIEW 1 of 2") && editor.notes[0].contains("off-mic"), "{form}");
            assert_eq!(out.reviews, 1, "{form}");
            let (a, b) = the_clip(out.script.as_ref().expect(form));
            assert!(a >= 0.4 && b <= 7.0, "{form}: the revision is what is saved, not the first cut: {a}-{b}");
            assert_eq!(out.saved, None, "{form}: the revision is a new version");
            assert!(out.error.is_none() && out.reply.is_none(), "{form}");
            finished.push(out.script.unwrap());
        }
        assert!(finished.windows(2).all(|w| w[0] == w[1]), "every form finishes the same");
    }

    /// A script that cannot be used goes back with the reason on every backend; before, the local
    /// and CLI loops ended the turn on it (`resolve_script(...)?`) where the server fed it back.
    #[tokio::test]
    async fn an_unusable_answer_goes_back_with_the_reason() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let clean = cut(1.0, 6.0);
        let bad_id = r#"{"action":"final","script":{"title":"t","beats":[{"id":"b1","purpose":"p","narration":"",
            "clips":[{"video_id":7,"from":"s99999","to":"s99999","audio":"source"}]}]}}"#;
        for (form, first) in [
            ("unknown sentence", Answer::Text(bad_id.to_string())),
            ("no script at all", Answer::Text("I saved the cut, it looks great.".into())),
            ("a tool call", Answer::Text(r#"{"action":"tool","tool":"get_outline","args":{}}"#.into())),
        ] {
            let r = rules(&project, &g, &cfg);
            let mut editor = Scripted::new(vec![Answer::Text(json_of(&clean))]);
            let out = deliver(&mut db, &r, &mut editor, first).await.unwrap();
            assert_eq!(editor.notes.len(), 1, "{form}");
            assert!(editor.notes[0].contains("could not be used"), "{form}: {}", editor.notes[0]);
            assert!(!editor.replays[0].is_empty() || form == "no script at all", "{form}: the answer is replayed");
            assert!(out.script.is_some(), "{form}: the fixed script is delivered");
            assert!(out.error.is_none(), "{form}");
        }
        // Still unusable after every round: nothing to save, and the turn says why.
        let r = rules(&project, &g, &cfg);
        let mut editor = Scripted::new(vec![Answer::Text(bad_id.into()), Answer::Text(bad_id.into())]);
        let out = deliver(&mut db, &r, &mut editor, Answer::Text(bad_id.into())).await.unwrap();
        assert_eq!(editor.notes.len(), MAX_REVIEWS);
        assert!(out.script.is_none());
        assert!(out.error.as_deref().is_some_and(|e| e.contains("s99999")), "{:?}", out.error);
    }

    /// What is saved is what was read: the delivered script is the review's own finished cut, not a
    /// second finish of the draft, and saving it stores exactly that.
    #[tokio::test]
    async fn the_saved_cut_is_the_one_that_was_reviewed() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let mut draft = cut(1.0, 14.0);
        draft.fill_from_project(&project);
        let r = rules(&project, &g, &cfg);
        let reviewed = review(&db, &r, &draft, Some(1)).unwrap().unwrap();
        assert!(reviewed.note.is_some());
        // Out of rounds (the editor gives no answer): the reviewed cut is delivered as reviewed.
        let mut editor = Scripted::new(Vec::new());
        let out = deliver(&mut db, &r, &mut editor, Answer::Script(draft.clone())).await.unwrap();
        let delivered = out.script.expect("a cut");
        assert_eq!(delivered, reviewed.finished, "saved = reviewed");
        let session = super::super::create_session(&db, project_id, "s").unwrap();
        let id = save(&db, project_id, session, &delivered, None).unwrap();
        assert_eq!(crate::script::load(&db, id).unwrap().script, delivered);
    }

    /// The version an editor saved itself, read clean, is kept as it is instead of saved again;
    /// one the review changed is a new version.
    #[tokio::test]
    async fn a_saved_version_the_review_leaves_alone_is_kept() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let r = rules(&project, &g, &cfg);
        // What save_script stores is already finished; finishing it for the chat changes nothing.
        let mut clean = cut(1.0, 6.0);
        clean.fill_from_project(&project);
        let clean = review(&db, &r, &clean, None).unwrap().unwrap().finished;
        let id = crate::script::save_version(&db, project_id, &clean, None).unwrap();
        let mut editor = Scripted::new(Vec::new());
        let out = deliver(&mut db, &r, &mut editor, Answer::Saved(id)).await.unwrap();
        assert!(editor.notes.is_empty());
        assert_eq!(out.saved, Some(id));
        let session = super::super::create_session(&db, project_id, "s").unwrap();
        assert_eq!(save(&db, project_id, session, out.script.as_ref().unwrap(), out.saved).unwrap(), id);
    }

    /// An error while reading a cut back is not worth the cut: it is logged and the cut is saved as
    /// finished, the way a failed network round keeps it (before, `review(...)?` ended the turn).
    #[tokio::test]
    async fn a_failure_to_read_the_cut_back_keeps_the_cut() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let mut r = rules(&project, &g, &cfg);
        r.read = |_, _, _, _, _| Err(Error::Invalid("database is locked".into()));
        let mut editor = Scripted::new(Vec::new());
        let out = deliver(&mut db, &r, &mut editor, Answer::Script(cut(1.0, 14.0))).await.unwrap();
        assert!(editor.notes.is_empty(), "nothing to send");
        assert!(out.script.is_some(), "the cut stands");
    }

    /// A revision that breaks a usable cut, an answer in words, or no answer at all leaves the
    /// reviewed cut standing; so do revisions that read no cleaner, and the rounds end at the limit.
    #[tokio::test]
    async fn a_usable_cut_is_never_lost_to_a_worse_answer() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let bad = cut(1.0, 14.0);
        for revision in [
            Answer::Text("sorry".into()),
            Answer::Text(r#"{"action":"reply","text":"cannot fix that"}"#.into()),
            Answer::Nothing,
        ] {
            let r = rules(&project, &g, &cfg);
            let mut editor = Scripted::new(vec![revision]);
            let out = deliver(&mut db, &r, &mut editor, Answer::Script(bad.clone())).await.unwrap();
            assert_eq!(editor.notes.len(), 1);
            let (a, b) = the_clip(out.script.as_ref().unwrap());
            assert!(a < 2.0 && b > 13.0, "the reviewed cut stands: {a}-{b}");
            assert!(out.reply.is_none() && out.error.is_none());
        }
        let r = rules(&project, &g, &cfg);
        let mut editor = Scripted::new(vec![Answer::Script(bad.clone()), Answer::Script(cut(1.0, 13.9))]);
        let out = deliver(&mut db, &r, &mut editor, Answer::Script(bad.clone())).await.unwrap();
        assert_eq!(editor.notes.len(), MAX_REVIEWS, "bounded");
        assert!(editor.notes[1].starts_with("REVIEW 2 of 2"), "{}", editor.notes[1]);
        assert_eq!(out.reviews, MAX_REVIEWS);
        assert!(editor.notes[1].contains("was not used"), "{}", editor.notes[1]);
        assert!(the_clip(out.script.as_ref().unwrap()).1 > 13.0, "the reviewed cut, finished");
    }

    /// The 9B's turn: the editor saved a cut, watched and exported it; the review flagged one clip,
    /// and the answer was an older draft pasted back, flagged as much. The saved version stands,
    /// and the editor is told its revision was not used.
    #[tokio::test]
    async fn a_revision_no_cleaner_than_the_saved_cut_leaves_the_saved_cut() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let mut saved = cut(1.0, 14.0);
        saved.fill_from_project(&project);
        let id = crate::script::save_version(&db, project_id, &saved, None).unwrap();
        let r = rules(&project, &g, &cfg);
        let mut editor = Scripted::new(vec![Answer::Script(cut(0.5, 14.0))]);
        let out = deliver(&mut db, &r, &mut editor, Answer::Saved(id)).await.unwrap();
        assert_eq!(editor.notes.len(), 2, "{:?}", editor.notes);
        assert!(editor.notes[1].starts_with("REVIEW 2 of 2: your revision was not used"), "{}", editor.notes[1]);
        assert!(editor.notes[1].contains("off-mic"), "the flags are repeated: {}", editor.notes[1]);
        assert_eq!(out.saved, Some(id), "the saved version is what the turn delivers");
        let stored = crate::script::load(&db, id).unwrap().script;
        let as_saved = review_from(&db, &r, &stored, None, true).unwrap().unwrap().finished;
        assert_eq!(out.script.as_ref(), Some(&as_saved), "not the pasted draft");
        // A revision that fixes the flag is still taken.
        let mut editor = Scripted::new(vec![Answer::Script(cut(1.0, 6.0))]);
        let out = deliver(&mut db, &r, &mut editor, Answer::Saved(id)).await.unwrap();
        assert_eq!(editor.notes.len(), 1);
        assert_eq!(out.saved, None);
        assert!(the_clip(out.script.as_ref().unwrap()).1 <= 7.0);
    }

    #[test]
    fn a_longer_draft_or_one_no_cleaner_is_not_a_fix() {
        let cut = |flags: Option<usize>, draft_s: f64, finished_s: f64| Standing { flags, draft_s, finished_s };
        // 54.7 s against 60 s, one clip flagged; the answer: the 146 s first draft, fitted down.
        let longer = worse_revision(cut(Some(1), 54.7, 54.7), cut(Some(0), 146.0, 63.0), Some(60.0));
        assert!(longer.as_deref().is_some_and(|r| r.contains("longer draft")), "{longer:?}");
        // Same flags, landing no nearer: no better.
        assert!(worse_revision(cut(Some(1), 54.7, 54.7), cut(Some(1), 55.0, 55.0), Some(60.0)).is_some());
        // More flags: worse.
        assert!(worse_revision(cut(Some(1), 54.7, 54.7), cut(Some(2), 58.0, 58.0), Some(60.0)).is_some());
        // Unreadable: no better.
        assert!(worse_revision(cut(Some(1), 54.7, 54.7), cut(None, 55.0, 55.0), Some(60.0)).is_some());
        // Fewer flags, within the target: a fix.
        assert_eq!(worse_revision(cut(Some(2), 54.7, 54.7), cut(Some(1), 58.0, 58.0), Some(60.0)), None);
        // The reviewed draft was already long: a long revision is judged on its flags alone.
        assert_eq!(worse_revision(cut(Some(2), 140.0, 70.0), cut(Some(0), 146.0, 62.0), Some(60.0)), None);
        // The reviewed cut could not be read: nothing to compare with.
        assert_eq!(worse_revision(cut(None, 54.7, 54.7), cut(Some(3), 146.0, 60.0), Some(60.0)), None);
        // The 9B's run 33: the reviewed cut was 29.7 s of a 60 s target (one flag, the length); the
        // revision ran 122 s as written and finished near 60 s with the same one flag. Half the
        // target is not a cut to protect from a longer draft, and the revision lands nearer.
        assert_eq!(worse_revision(cut(Some(1), 29.7, 29.7), cut(Some(1), 122.0, 61.0), Some(60.0)), None);
    }

    /// A first answer in words is the turn's reply, with nothing drafted.
    #[tokio::test]
    async fn an_answer_in_words_is_a_reply() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let r = rules(&project, &g, &cfg);
        let mut editor = Scripted::new(Vec::new());
        let first = Answer::Text(r#"{"action":"reply","text":"Which interview?"}"#.into());
        let out = deliver(&mut db, &r, &mut editor, first).await.unwrap();
        assert_eq!(out.reply.as_deref(), Some("Which interview?"));
        assert!(out.script.is_none() && editor.notes.is_empty());
    }

    /// The requested length and shape are the draft's before it is finished and read, so the review
    /// reads the cut that is saved (the local and CLI loops used to set the length after review,
    /// and every backend applied the shape after it).
    #[tokio::test]
    async fn the_requested_length_and_shape_are_set_before_the_review_reads_the_cut() {
        let (mut db, project_id) = project_with_interview();
        let project = db.project(project_id).unwrap();
        let cfg = crate::config::ScriptConfig::default();
        let g = super::super::Grounding::default();
        let mut r = rules(&project, &g, &cfg);
        r.requested_s = Some(5.0);
        r.requested_fmt = Some(crate::clips::ClipFormat::parse("9:16").unwrap());
        static SEEN: std::sync::Mutex<Vec<(Option<f64>, Option<i64>)>> = std::sync::Mutex::new(Vec::new());
        r.read = |db, pid, s, cfg, round| {
            SEEN.lock().unwrap().push((s.target_duration_s, s.width));
            read_back(db, pid, s, cfg, round)
        };
        let mut editor = Scripted::new(Vec::new());
        let out = deliver(&mut db, &r, &mut editor, Answer::Text(json_of(&cut(1.0, 6.0)))).await.unwrap();
        let s = out.script.unwrap();
        let (w, h) = crate::clips::ClipFormat::parse("9:16").unwrap().size().unwrap();
        assert_eq!((s.target_duration_s, s.width, s.height), (Some(5.0), Some(w), Some(h)));
        assert_eq!(SEEN.lock().unwrap().as_slice(), &[(Some(5.0), Some(w))], "the read saw the saved shape");
    }
}
