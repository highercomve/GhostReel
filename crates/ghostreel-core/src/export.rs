//! Script timeline export: OpenTimelineIO and Final Cut Pro 7 XML (plan §4a, D14).

use std::path::{Path, PathBuf};
use std::str::FromStr;

use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::db::Db;
use crate::otio;
use crate::script;

/// Supported timeline export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Otio,
    FcpXml,
}

impl ExportFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Otio => "otio",
            Self::FcpXml => "fcp_xml",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Otio => "otio",
            Self::FcpXml => "xml",
        }
    }
}

impl FromStr for ExportFormat {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "otio" | ".otio" | "otio_json" => Ok(Self::Otio),
            "fcp_xml" | "fcpxml" | "fcp" | "xml" | ".xml" => Ok(Self::FcpXml),
            other => Err(Error::Invalid(format!("unknown export format '{other}' (expected 'otio' or 'fcp_xml')"))),
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExportResult {
    pub export_id: i64,
    pub path: PathBuf,
    pub format: ExportFormat,
}

/// Export a script to an OpenTimelineIO or Final Cut Pro 7 XML timeline file.
pub fn export_script(db: &Db, script_id: i64, format: ExportFormat, out_path: &Path) -> Result<ExportResult, Error> {
    let stored = script::load(db, script_id)?;
    let timeline = otio::build_timeline(db, stored.project_id, &stored.script)?;

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::Io(parent.to_path_buf(), e))?;
    }

    match format {
        ExportFormat::Otio => {
            let json_str = serde_json::to_string_pretty(&timeline)
                .map_err(|e| Error::Export(format!("failed to serialize timeline JSON: {e}")))?;
            std::fs::write(out_path, json_str).map_err(|e| Error::Io(out_path.to_path_buf(), e))?;
        }
        ExportFormat::FcpXml => {
            let xml = crate::fcpxml::from_otio(&timeline)?;
            std::fs::write(out_path, xml).map_err(|e| Error::Io(out_path.to_path_buf(), e))?;
        }
    }

    let now = crate::projects::now();
    db.conn.execute(
        "INSERT INTO exports(script_id, format, path, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![stored.id, format.as_str(), out_path.to_string_lossy(), now],
    )?;
    let export_id = db.conn.last_insert_rowid();

    Ok(ExportResult { export_id, path: out_path.to_path_buf(), format })
}

/// Read an exported timeline back and report what it contains: how many clips landed on which
/// track, how long it runs, and — the failure worth catching before an editor opens it — any media
/// the timeline points at that is not on disk.
pub fn validate_export(path: &Path) -> Result<serde_json::Value, Error> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
    let is_xml = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("xml"))
        || text.trim_start().starts_with("<?xml")
        || text.trim_start().starts_with("<xmeml");

    let summary = if is_xml {
        crate::fcpxml::summarize_fcp_xml(&text)?
    } else {
        let doc: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Export(format!("{}: not a timeline we can read: {e}", path.display())))?;
        crate::fcpxml::summarize_otio(&doc)?
    };

    serde_json::to_value(summary).map_err(|e| Error::Export(format!("failed to serialize validation: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projects::NewProject;

    #[test]
    fn format_parsing_and_extension() {
        assert_eq!(ExportFormat::from_str("otio").unwrap(), ExportFormat::Otio);
        assert_eq!(ExportFormat::from_str(".otio").unwrap(), ExportFormat::Otio);
        assert_eq!(ExportFormat::from_str("otio_json").unwrap(), ExportFormat::Otio);
        assert_eq!(ExportFormat::from_str("fcp_xml").unwrap(), ExportFormat::FcpXml);
        assert_eq!(ExportFormat::from_str("xml").unwrap(), ExportFormat::FcpXml);
        assert_eq!(ExportFormat::from_str(".xml").unwrap(), ExportFormat::FcpXml);
        assert!(ExportFormat::from_str("unknown").is_err());

        assert_eq!(ExportFormat::Otio.as_str(), "otio");
        assert_eq!(ExportFormat::FcpXml.as_str(), "fcp_xml");
        assert_eq!(ExportFormat::Otio.extension(), "otio");
        assert_eq!(ExportFormat::FcpXml.extension(), "xml");
    }

    #[test]
    fn export_fcp_xml_end_to_end() {
        let mut db = Db::open_in_memory().unwrap();
        let project = db.create_project(&NewProject::named("TestExport")).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let folder = db.add_folder(project.id, temp.path(), true).unwrap();

        let vid_path = temp.path().join("v1.mp4");
        std::fs::write(&vid_path, b"test").unwrap();

        db.conn
            .execute(
                "INSERT INTO videos(id, content_hash, size, duration_s, fps, has_audio)
                 VALUES (1, 'hash1', 100, 10.0, 25.0, 1)",
                [],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO video_files(video_id, folder_id, path, size, mtime, last_seen)
                 VALUES (1, ?1, ?2, 100, 0, 0)",
                params![folder.id, vid_path.to_str().unwrap()],
            )
            .unwrap();

        let script = script::Script {
            title: "Integration Test".into(),
            target_duration_s: Some(5.0),
            fps: Some(script::Fps::new(25, 1)),
            width: Some(1920),
            height: Some(1080),
            beats: vec![script::Beat {
                id: "b1".into(),
                purpose: "intro".into(),
                narration: Some("Hello".into()),
                on_screen_text: Some("Title".into()),
                clips: vec![script::ScriptClip {
                    video_id: 1,
                    in_s: 0.0,
                    out_s: 4.0,
                    audio: script::Audio::Source,
                    why: None,
                }],
                notes: None,
                bed: None,
            }],
        };

        let script_id = script::save_version(&db, project.id, &script, None).unwrap();

        let out_xml = temp.path().join("out.xml");
        let res = export_script(&db, script_id, ExportFormat::FcpXml, &out_xml).unwrap();
        assert_eq!(res.format, ExportFormat::FcpXml);
        assert!(out_xml.is_file());

        let validation = validate_export(&out_xml).unwrap();
        assert_eq!(validation["clips"].as_i64(), Some(2)); // 1 V1 + 1 A1
        assert_eq!(validation["duration_s"].as_f64(), Some(4.0));
    }
}
