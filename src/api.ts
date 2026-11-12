import { convertFileSrc, invoke } from "@tauri-apps/api/core";

/** True in the desktop app: Tauri v2 injects its internals on `window`; a plain browser has none. */
export const isDesktop = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

const errorOf = (body: unknown): string | null =>
  typeof body === "object" && body !== null && typeof (body as { error?: unknown }).error === "string"
    ? (body as { error: string }).error
    : null;

/**
 * One backend command, whichever host this is running in: a Tauri command, or `POST /api/call`
 * when the frontend is served over HTTP. A failed command rejects with its own message either
 * way, so callers can keep doing `String(e)`.
 */
export async function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (isDesktop) return invoke<T>(cmd, args);
  const res = await fetch("/api/call", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ cmd, args: args ?? {} }),
  });
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    throw `${cmd}: the server answered ${res.status}, not JSON`;
  }
  if (!res.ok) throw errorOf(body) ?? `${cmd} failed (HTTP ${res.status})`;
  return body as T;
}

const mediaPath = (path: string) => `/media?path=${encodeURIComponent(path)}`;

export const fileUrl = (path: string) => (isDesktop ? convertFileSrc(path) : mediaPath(path));

// Mirrors ghostreel-core's doctor::Report (serde field names).
export type Target = "server" | "local" | "unavailable";
export type Backend = "auto" | "local" | "server" | "cli";

export interface Probe {
  url: string;
  reachable: boolean;
  capable: boolean;
  model: string | null;
  detail: string;
}

export interface Resolution {
  backend: Backend;
  target: Target;
  probe: Probe | null;
  reason: string;
}

export interface Tool {
  name: string;
  path: string | null;
  version: string | null;
}

export interface Gpu {
  name: string;
  vram_total_mib: number;
  vram_used_mib: number;
  driver: string;
}

export interface ModelFile {
  role: string;
  pattern: string;
  found: string | null;
}

export interface Report {
  version: string;
  config_file: string;
  config_error: string | null;
  data_dir: string;
  db: {
    path: string;
    ok: boolean;
    schema_version: number | null;
    sqlite_vec: string | null;
    error: string | null;
  };
  ffmpeg: Tool;
  ffprobe: Tool;
  gpu: Gpu[];
  vision: Resolution;
  embeddings: Resolution;
  stt: Resolution;
  models: ModelFile[];
}

export interface DoctorView {
  report: Report;
  blockers: string[];
}

export const doctor = () => call<DoctorView>("doctor");

// ---- projects & library ------------------------------------------------------------------

export interface PipelineConfig {
  probe: boolean;
  transcribe: boolean;
  frames: boolean;
  describe: boolean;
  embed: boolean;
  steadiness: boolean;
  faces: boolean;
  speakers?: boolean;
}

export interface GlossaryTerm {
  term: string;
  heard_as: string[];
  note?: string;
}

export interface Glossary {
  terms: GlossaryTerm[];
}

export interface Project {
  id: number;
  name: string;
  description: string;
  fps_num: number;
  fps_den: number;
  width: number;
  height: number;
  created_at: number;
  pipeline: PipelineConfig;
  glossary?: Glossary;
}

export interface StageCounts {
  stage: string;
  pending: number;
  running: number;
  done: number;
  failed: number;
  skipped: number;
}

export interface Status {
  folders: number;
  videos: number;
  total_size: number;
  total_duration_s: number;
  vfr_videos: number;
  stages: StageCounts[];
  steadiness_measured?: number;
  faces_measured?: number;
}

export interface ProjectSummary {
  project: Project;
  status: Status;
}

export interface FolderView {
  id: number;
  path: string;
  recursive: boolean;
  enabled: boolean;
  available: boolean;
}

export interface VideoRow {
  id: number;
  folder_ids: number[];
  path: string;
  copies: number;
  size: number;
  duration_s: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  vfr: boolean;
  vcodec: string | null;
  has_audio: boolean | null;
  status: string;
  error: string | null;
  language: string | null;
  segments: number;
  transcribe: string | null;
  frames: number;
  /** Seconds the camera measured shaky; 0 = none, unmeasured, or the check is off. */
  shaky_s: number;
  /** False until the camera has been measured; "steady" means nothing before that. */
  steadiness_measured: boolean;
  /** Whether faces were tracked for this video. */
  faces_measured: boolean;
  /** static, tripod, stabilised, handheld, or unknown. */
  camera: string;
  /** Category: interview, broll, screencast, drone, audio, other, or custom. */
  category: string;
  /** Confidence of auto-classification (0.0 to 1.0). */
  category_confidence: number;
  /** Source: 'auto' or 'manual'. */
  category_source: string;
  /** Secondary category tag (e.g. 'music'). */
  category_tag?: string | null;
}

export interface SidecarMove {
  from: string;
  to: string;
}

export interface FileMoveItem {
  video_id: number;
  file_name: string;
  current_path: string;
  target_path: string;
  category: string;
  category_folder: string;
  watched_folder_path: string;
  already_organized: boolean;
  conflict: string | null;
  sidecars: SidecarMove[];
}

export interface OrganizePlan {
  project_id: number;
  items: FileMoveItem[];
  total_files: number;
  to_move_count: number;
  already_organized_count: number;
  conflict_count: number;
}

export interface OrganizeReport {
  project_id: number;
  moved_videos: number;
  moved_sidecars: number;
  skipped: number;
  errors: string[];
}

export const CATEGORIES = [
  { id: "interview", label: "Interview", folder: "Interviews", icon: "🎙️" },
  { id: "broll", label: "B-Roll", folder: "B-Roll", icon: "🎬" },
  { id: "screencast", label: "Screen Recording", folder: "Screen Recordings", icon: "💻" },
  { id: "drone", label: "Drones & Aerials", folder: "Drones & Aerials", icon: "🚁" },
  { id: "audio", label: "Music & Audio", folder: "Music & Audio", icon: "🎵" },
  { id: "other", label: "Other", folder: "Other", icon: "📁" },
] as const;

/** One measured stretch of a video: `jerk` is camera shake as % of frame width per frame. */
export interface MotionWindow {
  start_s: number;
  end_s: number;
  jerk: number;
  /** How fast the camera moves on purpose, % of frame width per frame. */
  motion: number;
  /** Movement within a second that is undone, % of frame width per second. */
  sway: number;
}
export interface SteadinessView {
  windows: MotionWindow[];
  max_shake: number;
  /** The line this clip is judged against (floor, or a multiple of its own level). */
  limit: number;
  max_sway: number;
  camera: string;
}

export interface FrameRow {
  id: number;
  t_s: number;
  path: string;
  description: string | null;
  visible_text: string | null;
}

export const videoFrames = (videoId: number) => call<FrameRow[]>("video_frames", { videoId });
export const videoSteadiness = (videoId: number) => call<SteadinessView>("video_steadiness", { videoId });

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  off_mic?: boolean;
}

/** The models a coding-agent CLI will accept; empty when it cannot say. */
export const cliModels = (tool: string) => call<string[]>("cli_models", { tool });

export const videoTranscript = (videoId: number) => call<TranscriptSegment[]>("video_transcript", { videoId });

export interface ProjectView {
  project: Project;
  folders: FolderView[];
  status: Status;
  videos: VideoRow[];
  excluded: ExcludedVideo[];
  face_models_available?: boolean;
  /** Videos with speech whose voices are not analysed yet. */
  speakers_pending?: number;
  /** People found in the project so far. */
  people_count?: number;
  /** Each pipeline stage's status (index::stage_statuses), from the pipeline's own stage gate. */
  stages?: StageStatus[];
}

/** Where a stage stands for a project (`index::StageStatus`). Running is the queue's to say. */
export interface StageStatus {
  stage: string;
  label: string;
  state: "off" | "unavailable" | "pending" | "failed" | "done";
  done: number;
  total: number;
  /** Why it is off, unavailable, pending (e.g. "postponed: …") or failed. */
  reason: string | null;
}

export type ExclusionRole = "removed" | "reference";
export interface ExcludedVideo {
  video_id: number;
  role: ExclusionRole;
  path: string;
}
export const excludeVideo = (projectId: number, videoId: number, role: ExclusionRole) =>
  call<void>("exclude_video", { projectId, videoId, role });
export const includeVideo = (projectId: number, videoId: number) =>
  call<void>("include_video", { projectId, videoId });

export type IndexEvent =
  | { event: "scan_folder"; path: string }
  | { event: "folder_missing"; path: string }
  | { event: "scanned"; new: number; changed: number; unchanged: number; removed: number }
  | { event: "job_started"; video_id: number; stage: string; path: string }
  | { event: "job_done"; video_id: number; stage: string }
  | { event: "job_failed"; video_id: number; stage: string; error: string }
  | { event: "stage_backend"; stage: string; backend: string }
  | { event: "stage_unavailable"; stage: string; reason: string }
  | { event: "downloading_model"; file: string }
  | ({ event: "progress" } & Progress);

export interface Progress {
  phase: string;
  /** The phase's human name, from the core (`runs::phase_label`). */
  label: string;
  phase_done: number;
  phase_total: number;
  fraction: number;
  eta_secs: number | null;
  elapsed_secs: number;
  current: string | null;
  /** No measurable end — the bar animates rather than claiming a percentage. */
  indeterminate?: boolean;
}


/** "a few seconds", "42 s", "about 2 min", "about 1 h 20 min" — same wording as the CLI. */
export const etaText = (secs: number) => {
  const s = Math.max(0, Math.round(secs));
  if (s < 10) return "a few seconds";
  if (s < 60) return `${s} s`;
  if (s < 3600) return `about ${Math.ceil(s / 60)} min`;
  return `about ${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")} min`;
};

/** `partial`: finished, but some work could not be done now (a stage postponed, videos that
 *  failed); the reason is in `error`. */
export type TaskState = "queued" | "running" | "done" | "partial" | "failed" | "cancelled";

// Mirrors ghostreel-core's runs.rs: every long job, from the app's queue or the CLI.
export type RunState = TaskState;
/** `warning`: the step finished but kept warnings (its events say which). */
export type StepState = "running" | "done" | "warning" | "failed" | "stopped";

/** Why an agent's call failed (`agents::AgentError`): a kind to act on and show as a chip, a
 *  short line in the tool's own words and the raw tail for details. */
export type AgentErrorKind =
  | { kind: "invalid_model"; model: string }
  | { kind: "quota_exhausted"; resets_in_s?: number }
  | { kind: "busy"; retry_after_s?: number }
  | { kind: "auth" }
  | { kind: "not_installed" }
  | { kind: "timeout"; after_s: number }
  | { kind: "bad_answer" }
  | { kind: "unreachable" }
  | { kind: "http"; status: number }
  | { kind: "exit"; code?: number }
  | { kind: "no_agent_left"; role: string; agents: AgentError[] }
  | { kind: "other" };

export type AgentError = AgentErrorKind & {
  agent: string;
  message: string;
  details?: string;
};

export type EventLevel = "info" | "warning" | "error";

/** Something a step keeps beside its last note (a warning, a rerouted agent, the failure). */
export interface StepEvent {
  level: EventLevel;
  message: string;
  error?: AgentError | null;
  at: number;
}

export interface RunStep {
  phase: string;
  label: string;
  note: string;
  state: StepState;
  started_at: number;
  finished_at: number | null;
  events?: StepEvent[];
}

/** Something a run made or worked on (`runs::RunLink`); src/runs/links.tsx knows how to open each kind. */
export type RunLink =
  | { kind: "clip_set"; id: string }
  | { kind: "script"; id: number }
  | { kind: "file"; path: string }
  | { kind: "video"; id: number }
  | { kind: "project"; id: number };

/** Where a link's target stands (`runs::LinkState`): there, in the trash (restorable), or gone. */
export type LinkState = { state: "live" } | { state: "trashed"; trash_id: string } | { state: "gone" };

/** For each link, where what it points at stands now (`runs::RunLink::state`). */
export const runLinksState = (links: RunLink[]) => call<LinkState[]>("run_links_state", { links });

/** One delete in the trash (`trash::TrashItem`), restorable until `purge_at`. */
export interface TrashItem {
  id: string;
  project_id: number;
  kind: "cut" | "script_version" | "clip_set";
  title: string;
  /** What was deleted, in words: "Clip set “Launch” (8 clips)". */
  label: string;
  scripts: number;
  clips: number;
  version?: number;
  deleted_at: number;
  purge_at: number;
  link: RunLink;
}

export const listTrash = (projectId: number) => call<TrashItem[]>("list_trash", { projectId });
export const restoreTrash = (trashId: string) => call<TrashItem>("restore_trash", { trashId });
/** Restore what a run link points at from the trash. */
export const restoreRunLink = (link: RunLink) => call<TrashItem>("restore_run_link", { link });

/** A saved run (`runs::Run`). */
export interface Run {
  id: string;
  project_id: number | null;
  kind: string;
  label: string;
  state: RunState;
  steps: RunStep[];
  links: RunLink[];
  summary: string | null;
  error: string | null;
  source: "app" | "cli" | string;
  created_at: number;
  finished_at: number | null;
}

/** Saved runs, newest first: one project's, or every project's. */
export const listRuns = (projectId?: number | null, limit?: number) =>
  call<Run[]>("list_runs", { projectId: projectId ?? null, limit: limit ?? null });

export type TaskKind =
  | { type: "index"; project_id: number }
  | { type: "chat"; project_id: number; session_id: number }
  | { type: "render_preview"; script_id: number; burn_titles: boolean; burn_narration: boolean; burn_captions: boolean; normalize_audio: boolean }
  | { type: "export"; script_id: number; format: string; path: string }
  | { type: "download_model"; model_id: string }
  | { type: "steadiness"; project_id: number; force?: boolean }
  | { type: "faces"; project_id: number; force?: boolean }
  | { type: "speakers"; project_id: number; force?: boolean }
  | { type: "clips"; project_id: number; options: ClipOptions }
  | { type: "export_clips"; set_id: string; dir: string; accepted_only: boolean };

export interface Task {
  id: number;
  /** The run this task is; a saved run keeps it after the queue forgets the task. */
  run_id: string;
  project_id: number | null;
  kind: TaskKind;
  label: string;
  state: TaskState;
  progress: Progress | null;
  note: string | null;
  steps: RunStep[];
  links: RunLink[];
  summary: string | null;
  /** The raw result (a rendered file's path). */
  output: string | null;
  error: string | null;
  created_at: number;
  finished_at: number | null;
  chat_events?: ChatEvent[];
}

export interface PlannedSegment {
  video_id: number;
  path: string;
  in_s: number;
  out_s: number;
  timeline_start_s: number;
  beat_id: string;
  mute: boolean;
  has_audio: boolean;
}

export const enqueueIndex = (projectId: number) => call<number>("enqueue_index", { projectId });
export const enqueueSteadiness = (projectId: number, force = false) =>
  call<number>("enqueue_steadiness", { projectId, force });
export const enqueueFaces = (projectId: number, force = false) =>
  call<number>("enqueue_faces", { projectId, force });
export const enqueueSpeakers = (projectId: number, force = false) =>
  call<number>("enqueue_speakers", { projectId, force });
/** `out`: save the rendered MP4 there (demo export) instead of the previews folder. */
export const enqueuePreview = (
  scriptId: number,
  burnTitles: boolean,
  burnNarration: boolean,
  burnCaptions: boolean,
  normalizeAudio: boolean,
  out?: string,
) => call<number>("enqueue_preview", { scriptId, burnTitles, burnNarration, burnCaptions, normalizeAudio, out: out ?? null });
export const enqueueExport = (scriptId: number, format: string, path: string) =>
  call<number>("enqueue_export", { scriptId, format, path });
export const previewPlan = (scriptId: number) => call<PlannedSegment[]>("preview_plan", { scriptId });
export const getScriptPreview = (scriptId: number) =>
  call<string | null>("get_script_preview", { scriptId });
export const queueList = () => call<Task[]>("queue_list");
export const cancelTask = (id: number) => call<boolean>("cancel_task", { id });
export const clearFinishedTasks = () => call<void>("clear_finished_tasks");

export interface HitProject {
  id: number;
  name: string;
}

export interface Hit {
  video_id: number;
  path: string;
  start_s: number;
  end_s: number;
  score: number;
  kinds: string[];
  matched_by: string[];
  snippet: string;
  frame: string | null;
  projects?: HitProject[];
}

export interface SearchScope { folderId?: number | null; videoIds?: number[] | null }
export const search = (projectId?: number | null, query: string = "", limit = 30, scope: SearchScope = {}) =>
  call<{ hits: Hit[]; note: string | null }>("search", {
    projectId: projectId ?? null, query, limit, folderId: scope.folderId ?? null, videoIds: scope.videoIds ?? null,
  });

export const videoDetails = (videoId: number) =>
  call<VideoRow>("video_details", { videoId });

export const openSearchWindow = () => {
  if (isDesktop) {
    return call<void>("open_search_window");
  } else {
    window.open("?window=search", "ghostreel-search", "width=1020,height=720");
    return Promise.resolve();
  }
};

export const focusMainWindow = () => {
  if (isDesktop) {
    return call<void>("focus_main_window");
  }
  return Promise.resolve();
};
let mediaBasePromise: Promise<string> | null = null;
/** URL the player can stream (byte ranges) — WebKitGTK can't stream video from the asset protocol. */
export const mediaUrl = async (path: string) => {
  // Served over HTTP the web server streams ranges itself; the desktop media port is loopback-only.
  if (!isDesktop) return mediaPath(path);
  mediaBasePromise ??= call<string>("media_base");
  return `${await mediaBasePromise}?path=${encodeURIComponent(path)}`;
};
export const openExternal = (path: string, t: number) => call<void>("open_external", { path, t });
/** What the player should load: the original, or a playable proxy built on first open. */
export const videoPlayback = (videoId: number) => call<{ path: string; proxy: boolean }>("video_playback", { videoId });

/** One audio stream of a file, as ffprobe sees it. */
export interface AudioTrackInfo {
  /** 0-based among the file's audio streams. */
  index: number;
  channels: number;
  codec: string;
  /** The stream's title or handler name, if it has one. */
  label: string | null;
  /** The track GhostReel plays when a clip doesn't choose. */
  auto: boolean;
}

export const videoAudioTracks = (videoId: number) => call<AudioTrackInfo[]>("video_audio_tracks", { videoId });

/** Mean volume in dB per audio track over a source range; -91 or lower is silence. */
export const audioLevels = (videoId: number, inS: number, outS: number) =>
  call<number[]>("audio_levels", { videoId, inS, outS });

/** Path to one audio track extracted to a small playable file (cached by the backend). */
export const audioTrackProxy = (videoId: number, track: number) =>
  call<string>("audio_track_proxy", { videoId, track });

export { formatDuration as clock } from "./timeFormat";

export const fileName = (p: string) => p.split(/[\\/]/).pop() ?? p;

export const listProjects = () => call<ProjectSummary[]>("list_projects");
export const createProject = (
  name: string,
  fpsNum: number,
  fpsDen: number,
  width: number,
  height: number,
  pipeline?: Partial<PipelineConfig>,
) =>
  call<Project>("create_project", {
    name,
    fpsNum,
    fpsDen,
    width,
    height,
    pipeline: pipeline
      ? {
          probe: pipeline.probe ?? true,
          transcribe: pipeline.transcribe ?? true,
          frames: pipeline.frames ?? true,
          describe: pipeline.describe ?? true,
          embed: pipeline.embed ?? true,
        }
      : null,
  });
export const setProjectPipeline = (projectId: number, pipeline: PipelineConfig) =>
  call<Project>("set_project_pipeline", { projectId, pipeline });
export const getProjectGlossary = (projectId: number) =>
  call<Glossary>("get_project_glossary", { projectId });
/** A person in a project: face tracks grouped by identity, named for lower-third name tags. */
export interface PersonSummary {
  id: number;
  name: string | null;
  handle: string | null;
  role: string | null;
  videos: number[];
  tracks: number;
  screen_time_s: number;
  speaking_time_s: number;
}
export interface Person {
  id: number;
  project_id: number;
  name: string | null;
  handle: string | null;
  role: string | null;
}
export interface SpeakerTurn {
  start_s: number;
  end_s: number;
  speaker: number;
  person_id: number | null;
  name: string | null;
  confidence: number | null;
}
export const projectPeople = (projectId: number) => call<PersonSummary[]>("project_people", { projectId });
/** Omitted fields stay as they are; an empty string clears one. */
export const setPerson = (
  projectId: number,
  personId: number,
  fields: { name?: string; handle?: string; role?: string },
) => call<Person>("set_person", { projectId, personId, ...fields });
export const mergePeople = (projectId: number, keepId: number, mergeId: number) =>
  call<Person>("merge_people", { projectId, keepId, mergeId });
export const videoSpeakers = (videoId: number, fromS?: number, toS?: number) =>
  call<SpeakerTurn[]>("video_speakers", { videoId, fromS, toS });
/** Path of a cached face thumbnail, or null when the person has no usable face box. */
export const personThumbnail = (projectId: number, personId: number) =>
  call<string | null>("person_thumbnail", { projectId, personId });
export const setProjectGlossary = (projectId: number, glossary: Glossary) =>
  call<Glossary>("set_project_glossary", { projectId, glossary });
export const renameProject = (projectId: number, name: string) =>
  call<Project>("rename_project", { projectId, name });
export interface PurgeStats {
  videos: number;
  files_deleted: number;
  bytes_freed: number;
}
/** `purge`: also delete keyframes, previews and the index of footage no other project uses. */
export const removeProject = (projectId: number, purge = false) =>
  call<PurgeStats>("remove_project", { projectId, purge });
// Mirrors ghostreel-core's storage module.
export type StorageKind = "proxies" | "keyframes" | "previews" | "chat_images" | "chat_logs" | "exports" | "temp";
export type StorageUsage = Record<StorageKind, number>;
export interface ProjectStorage {
  project_id: number;
  name: string;
  videos: number;
  usage: StorageUsage;
  /** Part of `usage` that belongs to footage other projects use too. */
  shared: number;
}
export interface StorageReport {
  data_dir: string;
  total: number;
  usage: StorageUsage;
  models: number;
  database: number;
  /** Daily database snapshots under backups/ (never removed by cleanup). */
  backups: number;
  other: number;
  /** Files no project uses: footage no project watches, deleted scripts and chats, scratch. */
  unused: StorageUsage;
  unused_videos: number;
  projects: ProjectStorage[];
}
export interface CleanStats {
  files_deleted: number;
  bytes_freed: number;
  videos: number;
}
export const storageReport = () => call<StorageReport>("storage_report");
/** `unused` ignores `kinds` and removes everything nothing uses. Video files are never touched. */
export const cleanStorage = (scope: "all" | "project" | "unused", kinds: StorageKind[], projectId?: number) =>
  call<CleanStats>("clean_storage", { scope, projectId: projectId ?? null, kinds });
export const compactDatabase = () => call<number>("compact_database");
export const projectView = (projectId: number) => call<ProjectView>("project_view", { projectId });
export const addFolder = (projectId: number, path: string, recursive = true) =>
  call<void>("add_folder", { projectId, path, recursive });
export const removeFolder = (projectId: number, path: string) => call<void>("remove_folder", { projectId, path });

export const setVideoCategory = (videoId: number, category: string) =>
  call<void>("set_video_category", { videoId, category });
export const planOrganizeProject = (projectId: number) =>
  call<OrganizePlan>("plan_organize_project", { projectId });
export const executeOrganizeProject = (projectId: number) =>
  call<OrganizeReport>("execute_organize_project", { projectId });


export const FPS_PRESETS: { label: string; num: number; den: number }[] = [
  { label: "23.976", num: 24000, den: 1001 },
  { label: "24", num: 24, den: 1 },
  { label: "25", num: 25, den: 1 },
  { label: "29.97", num: 30000, den: 1001 },
  { label: "30", num: 30, den: 1 },
  { label: "50", num: 50, den: 1 },
  { label: "59.94", num: 60000, den: 1001 },
  { label: "60", num: 60, den: 1 },
];

export const fpsLabel = (num: number, den: number) =>
  FPS_PRESETS.find((p) => p.num === num && p.den === den)?.label ?? (num / den).toFixed(3);

export const humanSize = (bytes: number) =>
  bytes >= 1e9
    ? `${(bytes / 1e9).toFixed(1)} GB`
    : bytes >= 1e6
      ? `${Math.round(bytes / 1e6)} MB`
      : bytes > 0
        ? `${Math.max(1, Math.round(bytes / 1e3))} KB`
        : "0 MB";

export const humanDuration = (s: number) => {
  const t = Math.round(s);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  return h > 0 ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}:${String(sec).padStart(2, "0")}`;
};

// ---- scripts, chat, timeline preview & export (M8c) --------------------------------------

export type Fps = number | { num: number; den: number };

export interface ScriptClip {
  video_id: number;
  in_s: number;
  out_s: number;
  audio?: "source" | "mute";
  /** Source audio tracks to play (0-based among the file's audio streams). Absent = the file's
   * automatic track; empty = silent. `audio: "mute"` silences the clip regardless. */
  tracks?: number[];
  why?: string;
  /** How the clip is cropped to the cut's shape; absent = the script's framing. */
  framing?: Framing;
}

/** How a stretch is framed: follow the speaker, speaker over listener, a still crop, screen over
 * the speaker's camera tile, or the whole picture over a blurred copy of itself. */
export type Layout = "follow_speaker" | "stacked" | "fixed" | "screen" | "fit";

/** What a stretch of picture is, from the face boxes. */
export type Scene = "two_shot" | "single_face" | "screen_share" | "no_face";

/** Cropping a wide picture to a vertical or square cut. */
export interface Framing {
  layout: Layout;
  person_id?: number;
  /** For `fixed`: the part of the picture to show, 0..1 of the source. */
  box?: { x: number; y: number; w: number; h: number };
  /** Stretches (source seconds) framed with a chosen layout instead of their picture's. */
  overrides?: { from_s: number; to_s: number; layout: Layout }[];
}

/** A point of a crop path: the crop's corner in source pixels, `t` seconds into the clip. */
export interface CropKeyframe {
  t: number;
  x: number;
  y: number;
  /** Jump here instead of moving from the previous keyframe. */
  cut?: boolean;
}

export interface CropPath {
  w: number;
  h: number;
  keyframes: CropKeyframe[];
}

/** One stretch of a clip with its own layout and crop size; stretches cut hard into each other. */
export interface PlanSegment {
  /** Clip seconds. */
  start: number;
  end: number;
  scene: Scene;
  layout: Layout;
  /** One path; two for stacked and screen (top, bottom); none for fit. */
  paths: CropPath[];
}

export interface FramePlan {
  layout: Layout;
  src_w: number;
  src_h: number;
  out_w: number;
  out_h: number;
  /** The picture without black bars, in source pixels. */
  content: { x: number; y: number; w: number; h: number };
  segments: PlanSegment[];
}

/** One frame of a plan (`framing::FrameView`): the source rectangles that fill the frame in equal
 * bands top to bottom (one crop, or two stacked), or none to fit the whole `content` picture. */
export interface FrameView {
  src_w: number;
  src_h: number;
  out_w: number;
  out_h: number;
  content: { x: number; y: number; w: number; h: number };
  crops: { x: number; y: number; w: number; h: number }[];
}

export interface ClipFraming {
  beat_index: number;
  clip_index: number;
  video_id: number;
  in_s: number;
  plan: FramePlan;
}

/** Sound that runs across a beat while its pictures change under it — a J-cut. */
export interface AudioBed {
  video_id: number;
  in_s: number;
  out_s: number;
  why?: string;
  /** True when the pipeline laid it rather than the editor asking for it. */
  inferred?: boolean;
  /** Source audio tracks to play; absent = the file's automatic track, empty = silent. */
  tracks?: number[];
}

export interface Beat {
  id: string;
  purpose: string;
  narration?: string;
  on_screen_text?: string;
  /** When, where and how the on-screen text shows; absent = whole beat, default place and style. */
  title_layout?: TitleLayout | null;
  clips: ScriptClip[];
  bed?: AudioBed | null;
  notes?: string;
}

/** Burnt-in text style; every field optional (falls back to the title or caption defaults). */
export interface TextStyle {
  font?: string;
  /** Line height as a fraction of the output frame's height. */
  size?: number;
  bold?: boolean;
  uppercase?: boolean;
  color?: string;
  outline_color?: string;
  /** Fraction of the font size. */
  outline?: number;
  /** Fraction of the font size. */
  shadow?: number;
  box?: boolean;
  box_color?: string;
  box_opacity?: number;
  align?: "left" | "center" | "right";
  /** Fraction of the frame's width. */
  max_width?: number;
}

export interface TitleLayout {
  /** Seconds after the beat starts. */
  start_s?: number;
  end_s?: number;
  /** 0..1 of the output frame; x follows the alignment, y is the text's middle. */
  x?: number;
  y?: number;
  style?: TextStyle;
}

export interface CaptionStyle {
  position?: "bottom" | "center" | string;
  size?: number;
  style?: TextStyle;
  /** 0..1 of the output frame; y is the bottom of the text. */
  x?: number;
  y?: number;
  max_chars?: number;
  max_lines?: number;
}

/** A hand edit of one caption cue, tied to the source footage it was heard in. */
export interface CueEdit {
  video_id: number;
  src_start_s: number;
  src_end_s: number;
  text?: string;
  /** Source seconds. */
  start_s?: number;
  end_s?: number;
  hidden?: boolean;
}

export interface Script {
  title: string;
  target_duration_s?: number;
  fps?: Fps;
  width?: number;
  height?: number;
  /** Default framing for every clip (a clip's own wins). */
  framing?: Framing;
  beats: Beat[];
  /** Speech captions burnt into every render with this style. */
  captions?: CaptionStyle | null;
  title_style?: TextStyle | null;
  caption_edits?: CueEdit[];
  /** Which agents work on this cut; absent = the app's choice. */
  agents?: AgentSelection | null;
}

export interface Cue {
  start_s: number;
  end_s: number;
  text: string;
  video_id?: number;
  src_start_s?: number;
  src_end_s?: number;
  /** Cut time minus source time: source = cut time - offset. */
  offset_s?: number;
  /** How far the cue may move on the cut's clock (its clip). */
  min_s?: number;
  max_s?: number;
  edited?: boolean;
  hidden?: boolean;
}

/** A text style with every value decided by the backend. */
export interface ResolvedStyle {
  font: string;
  size: number;
  bold: boolean;
  uppercase: boolean;
  color: string;
  outline_color: string;
  outline: number;
  shadow: number;
  box: boolean;
  box_color: string;
  box_opacity: number;
  align: "left" | "center" | "right";
  max_width: number;
  box_pad: number;
  cell_em: number;
}

export type VAnchor = "middle" | "bottom";

export interface Overlay {
  kind: "title" | "caption";
  beat_index?: number;
  start_s: number;
  end_s: number;
  lines: string[];
  x: number;
  y: number;
  v_anchor: VAnchor;
  style: ResolvedStyle;
  cue?: Cue;
}

export interface Overlays {
  width: number;
  height: number;
  titles: Overlay[];
  captions: Overlay[];
  caption_style: ResolvedStyle;
  caption_x: number;
  caption_y: number;
  caption_v_anchor: VAnchor;
}

export interface Issue {
  severity: "error" | "warning" | "info";
  beat_id: string | null;
  clip_index: number | null;
  message: string;
  /** Deterministic ways to fix it, shown as buttons; applying one checks the script again. */
  fixes?: Fix[];
}

/** A change that fixes an issue: move a clip's bounds, frame a stretch another way, or run an
 * indexing stage (turning it on first when `enable`). Times are source seconds. */
export type FixAction =
  | { kind: "trim"; beat_id: string; clip_index: number; in_s: number; out_s: number }
  | { kind: "set_layout"; beat_id: string; clip_index: number; from_s: number; to_s: number; layout: Layout }
  | { kind: "run_stage"; stage: string; enable: boolean };

export type Fix = FixAction & { label: string };

export interface FixedScriptView {
  script: Script;
  issues: Issue[];
  note: string;
}

export interface StoredScript {
  id: number;
  project_id: number;
  session_id: number | null;
  title: string;
  version: number;
  created_at: number;
  script: Script;
}

export interface ScriptSummary {
  id: number;
  session_id: number | null;
  title: string;
  version: number;
  created_at: number;
  beats: number;
  clips: number;
  duration_s: number;
  /** The clip set this script belongs to; null for ordinary cuts. */
  set_id?: string | null;
  /** "clip" = belongs to a clip set. */
  kind?: string | null;
}

/** How a chat builds its cuts; kept on the chat for every turn in it. */
export interface ChatStyle {
  /** Pictures only on a theme: no interviews, no voice-over. */
  broll: boolean;
  /** With `broll`: clips play their own ambient sound instead of being muted for music. */
  natural_sound: boolean;
}

export interface ChatSession {
  id: number;
  project_id: number;
  title: string;
  created_at: number;
  updated_at: number;
  style?: ChatStyle;
}

export interface ToolCallRecord {
  tool: string;
  args: any;
  summary: string;
}

export interface ChatMessage {
  id: number;
  session_id: number;
  role: "user" | "assistant" | "tool";
  content: string;
  images?: string[];
  tool_calls: ToolCallRecord[] | null;
  script_id: number | null;
  created_at: number;
  performance?: ChatPerformance | null;
}

export interface InferenceCall {
  model: string;
  backend: string;
  prompt_tokens: number | null;
  output_tokens: number | null;
  generation_seconds: number | null;
  request_seconds: number;
}

export interface ChatPerformance {
  elapsed_seconds: number;
  calls: InferenceCall[];
}

export interface ChatTurnView {
  session_id: number;
  reply: string;
  script_id: number | null;
  issues: Issue[];
}

export interface ScriptView {
  stored: StoredScript;
  issues: Issue[];
}

export interface SaveScriptView {
  script_id: number;
  issues: Issue[];
  /** What the deterministic repair changed before saving (empty: saved as given). */
  notes: string[];
  /** The script as saved. */
  script: Script;
}

export type ChatEvent =
  | { kind: "tool_started"; tool: string; args: any }
  | { kind: "tool_finished"; tool: string; summary: string }
  | { kind: "drafting" }
  | { kind: "validating" };

export interface ChatProgress {
  session_id: number;
  event: ChatEvent;
}

export const chatTurn = (
  projectId: number,
  sessionId: number | null,
  message: string,
  images?: string[],
  style?: ChatStyle,
  scriptId?: number | null,
) =>
  call<ChatTurnView>("chat_turn", {
    projectId,
    sessionId,
    message,
    images: images ?? [],
    style,
    scriptId: scriptId ?? null,
  });

/**
 * Build a cut by choosing instead of writing: Jev picks the quotes and the shots out of the index
 * and code assembles them. Returns the saved script id. Needs a Jev key (Settings).
 */
/** A cut Jev chose, and the conversation opened to refine it in. */
export type BuiltCut = { script_id: number; session_id: number };

export const buildScriptWithJev = (
  projectId: number,
  sessionId: number | null,
  brief: string,
  targetS: number,
) => call<BuiltCut>("build_script_with_jev", { projectId, sessionId, brief, targetS });

export const chatSessions = (projectId: number) =>
  call<ChatSession[]>("chat_sessions", { projectId });

export const deleteChatSession = (sessionId: number) => call<boolean>("delete_chat_session", { sessionId });

export const chatMessages = (sessionId: number) =>
  call<ChatMessage[]>("chat_messages", { sessionId });

/** One entry of a chat's model log: a prompt, the thinking, an answer, or helper output. */
export interface LogEntry {
  seq: number;
  ts_ms: number;
  session_id: number | null;
  kind: "prompt" | "thinking" | "answer" | "helper" | "tool" | "error" | "info";
  text: string;
}

export const chatLog = (sessionId: number, afterSeq: number) =>
  call<LogEntry[]>("chat_log", { sessionId, afterSeq });

export async function openChatLog(sessionId: number) {
  if (isDesktop) return call<void>("open_chat_log", { sessionId });
  const a = document.createElement("a");
  a.href = `/api/chat-log?session_id=${sessionId}`;
  a.download = `${sessionId}.jsonl`;
  a.click();
}

export const listScripts = (projectId: number) =>
  call<ScriptSummary[]>("list_scripts", { projectId });

export const getScript = (scriptId: number) =>
  call<ScriptView>("get_script", { scriptId });

export const saveScript = (projectId: number, script: Script, sessionId: number | null) =>
  call<SaveScriptView>("save_script", { projectId, script, sessionId });

/** Check an (unsaved) script against the index as it is now. */
export const checkScript = (projectId: number, script: Script) =>
  call<Issue[]>("check_script", { projectId, script });

/** Apply one fix to an (unsaved) script; the result is checked again but not saved. */
export const applyScriptFix = (projectId: number, script: Script, fix: FixAction) =>
  call<FixedScriptView>("apply_script_fix", { projectId, script, fix });

/** Deletes go to the trash: each returns the entry, so the caller can offer Undo. */
export const deleteScript = (scriptId: number) =>
  call<TrashItem>("delete_script", { scriptId });

export const deleteScriptCut = (projectId: number, title: string) =>
  call<TrashItem>("delete_script_cut", { projectId, title });

export const scriptCaptions = (script: Script) =>
  call<Cue[]>("script_captions", { script });

/** Titles and captions laid out as a render burns them (hidden cues included, flagged). */
export const scriptOverlays = (script: Script) =>
  call<Overlays>("script_overlays", { script });

export const scriptFraming = (script: Script) =>
  call<ClipFraming[]>("script_framing", { script });

// ---- models management -------------------------------------------------------------------

export type ModelKind = "whisper" | "vision" | "vision_projector" | "embedding" | "face" | "speaker";

export interface CatalogEntry {
  id: string;
  kind: ModelKind;
  file_name: string;
  url: string;
  size_bytes: number;
  speed: number;
  accuracy: number;
  note: string;
  languages: string;
  mmproj_file_name?: string;
  mmproj_url?: string;
  mmproj_size_bytes?: number;
  vram_mb?: number;
  bundle?: { repo: string; revision: string; files: { name: string; size_bytes: number }[] };
}

export interface ModelStatus {
  entry: CatalogEntry;
  installed_path: string | null;
  in_own_dir: boolean;
  partial_bytes: number | null;
  external?: { name: string; quant: string | null; source: string; mmproj_path: string | null };
}

export interface ModelsStatusView {
  dir: string;
  models: ModelStatus[];
  current_whisper_model?: string;
  current_vision_model?: string;
}

// ---- AI settings (backend per capability) ------------------------------------------------

/** Model settings for one capability: frame descriptions, or the script chat. */
export interface VisionSettings {
  backend: Backend;
  url: string;
  model: string;
  local_model: string;
  api_key_set: boolean;
  /** Context window of the local helper, in tokens. */
  ctx_tokens: number;
  /** KV cache precision: f16 | q8_0 | q4_0 (q4_0 ≈ 4× the context per GB). */
  kv_cache: string;
  /** auto | on | off */
  flash_attn: string;
  cpu_moe_layers: number;
  moe_blocks?: number | null;
  /** Optional cap on rounds of tool calls a model run may make; 0 = no cap (the default). */
  max_tool_rounds: number;
  think: boolean;
  /** Frames described at once against a server; only meaningful when the server has slots. */
  describe_concurrency: number;
  cli: CliSettings;
}

/** A coding-agent CLI used instead of a model. */
export interface CliSettings {
  tool: string;
  command: string;
  model: string;
  timeout_secs: number;
  concurrency: number;
  /** Whether the binary was found on this machine. */
  installed: boolean;
}

export interface SttSettings {
  backend: Backend;
  url: string;
  model: string;
}

export interface EmbedSettings {
  backend: Backend;
  url: string;
  model: string;
}

export interface FrameSettings {
  max_interval_s: number;
  long_side: number;
}

/** What a server admits it can do; drives whether a control is offered at all. */
export interface ServerCaps {
  /** Requests handled at once. null = the server didn't say (LM Studio, Ollama…), not "one". */
  slots: number | null;
  slot_ctx: number | null;
  /** llama.cpp router: model and flags changeable without a restart. */
  router: boolean;
}

export interface JevSettings {
  enabled: boolean;
  /** Whether every finished cut is read editorially. Separate from `enabled`, which also gates choosing. */
  judge: boolean;
  /** Whether a key exists at all. The key itself never comes back. */
  has_key: boolean;
  /** The key is in TYPESAFE_API_KEY, so the field is not editable here. */
  key_from_env: boolean;
  model: string;
}

export interface JevSettingsPatch {
  enabled?: boolean;
  judge?: boolean;
  /** "" clears a stored key. */
  api_key?: string;
  model?: string;
}

export interface AiSettings {
  /** Frame descriptions (indexing). */
  vision: VisionSettings;
  /** Script chat. */
  chat_model: VisionSettings;
  stt: SttSettings;
  embed: EmbedSettings;
  frames: FrameSettings;
  jev: JevSettings;
  vision_caps: ServerCaps;
  /** Clip workers: many small scoring jobs run in parallel. */
}


export interface BackendsResolution {
  vision: Resolution;
  /** Script chat (its own backend). */
  chat: Resolution;
  embeddings: Resolution;
  stt: Resolution;
}

export interface VisionSettingsPatch {
  backend?: string;
  /** Frames described at once against a server. Only meaningful when the server has slots. */
  describe_concurrency?: number;
  url?: string;
  model?: string;
  local_model?: string;
  api_key?: string;
  ctx_tokens?: number;
  kv_cache?: string;
  flash_attn?: string;
  cpu_moe_layers?: number;
  max_tool_rounds?: number;
  think?: boolean;
  cli?: {
    tool?: string;
    command?: string;
    model?: string;
    timeout_secs?: number;
    concurrency?: number;
  };
}

export interface SttSettingsPatch {
  backend?: string;
  url?: string;
  model?: string;
}

export interface EmbedSettingsPatch {
  backend?: string;
  url?: string;
  model?: string;
}

export interface FrameSettingsPatch {
  max_interval_s?: number;
  long_side?: number;
}

export interface AiSettingsPatch {
  vision?: VisionSettingsPatch;
  chat_model?: VisionSettingsPatch;
  stt?: SttSettingsPatch;
  embed?: EmbedSettingsPatch;
  frames?: FrameSettingsPatch;
  jev?: JevSettingsPatch;
}

export const getAiSettings = () => call<AiSettings>("get_ai_settings");
/** Run one describe call through the configured CLI agent. `capability`: "vision" | "chat_model". */
export const testCliAgent = (capability: string) => call<string>("test_cli_agent", { capability });
export const setAiSettings = (patch: AiSettingsPatch) => call<AiSettings>("set_ai_settings", { patch });
export const probeBackends = () => call<BackendsResolution>("probe_backends");
export const serverModels = (url: string) => call<string[]>("server_models", { url });

export const modelsStatus = () => call<ModelsStatusView>("models_status");
export const enqueueModelDownload = (modelId: string) =>
  call<number>("enqueue_model_download", { modelId });
export const removeModel = (modelId: string) =>
  call<void>("remove_model", { modelId });
export const setWhisperModel = (modelId: string) =>
  call<void>("set_whisper_model", { modelId });
export const openModelsDir = () => call<void>("open_models_dir");

export const redoProjectStage = (projectId: number, stage: string) =>
  call<number>("redo_project_stage", { projectId, stage });

/** Compact bar meter like "▰▰▰▱▱" for a 1–5 score. */
export const scoreMeter = (n: number) =>
  "▰".repeat(Math.min(5, Math.max(0, n))) + "▱".repeat(Math.max(0, 5 - n));

export interface ChatSettings {
  system_prompt: string;
  default_system_prompt: string;
}
export const getChatSettings = () => call<ChatSettings>("get_chat_settings");
export const setChatSystemPrompt = (prompt: string) => call<ChatSettings>("set_chat_system_prompt", { prompt });

export const appVersion = () => call<string>("app_version");

// ---- web access (serving this frontend over HTTP) ----------------------------------------

export interface WebStatus {
  enabled: boolean;
  running: boolean;
  bind: string;
  port: number;
  /** What to open on another device; empty until the server is up. */
  urls: string[];
  auth_enabled: boolean;
  auth_user: string;
  /** Why the server is not running, e.g. the port is taken. */
  error: string | null;
}

export interface WebConfigPatch {
  enabled?: boolean;
  bind?: string;
  port?: number;
  auth_enabled?: boolean;
  auth_user?: string;
  /** "" leaves the stored password alone; `web_status` never returns it. */
  auth_password?: string;
}

export const webStatus = () => call<WebStatus>("web_status");
export const setWebConfig = (patch: WebConfigPatch) => call<WebStatus>("set_web_config", { patch });

export interface PlaybackProgress {
  phase: string;
  fraction: number | null;
  encoded_s: number;
  encoder: string | null;
}
export const videoPlaybackProgress = (videoId: number) =>
  call<PlaybackProgress | null>("video_playback_progress", { videoId });

// ---- Clip sets ---------------------------------------------------------------------------

export type ClipFormat = "original" | "9:16" | "1:1";

export interface ClipOptions {
  brief: string;
  count: number;
  min_s: number;
  max_s: number;
  format: ClipFormat;
  /** Empty = every video in the library. */
  video_ids: number[];
  window_s?: number;
  /** Let code pick the clips instead of the orchestrator model. */
  worker_only: boolean;
  /** Framing for 9:16 / 1:1 clips. */
  layout?: "follow_speaker" | "stacked" | "fixed";
  /** This run's agents; what it leaves out comes from the app's choice. */
  agents?: AgentSelection;
}

export interface ClipVisual {
  label: "faces" | "screen" | "mixed" | "unknown" | string;
  screen_share: number;
  frames: number;
  faces?: { faces: number; avg_box_h: number } | null;
  /** Shares of the range by picture kind, measured from the faces (absent: not indexed). */
  layout?: { two_shot: number; single_face: number; screen_share: number; no_face: number } | null;
}

export type ClipStatus = "new" | "accepted" | "rejected";

export interface ClipItem {
  script_id: number;
  title: string;
  hook: string;
  quote: string;
  why: string;
  score: number;
  video_id: number;
  in_s: number;
  out_s: number;
  visual: ClipVisual;
  status: ClipStatus;
  /** Warnings the checks find now (recomputed whenever the set is shown). */
  warnings: number;
  issues?: Issue[];
  thumb?: string | null;
  /** For a clip that crops its footage: what its framing plan shows of `thumb`. */
  thumb_view?: FrameView | null;
}

export interface ClipSet {
  id: string;
  project_id: number;
  brief: string;
  created_at: number;
  options: Partial<ClipOptions> & Record<string, unknown>;
  items: ClipItem[];
  log: Record<string, unknown>;
}

export const generateClips = (projectId: number, options: ClipOptions) =>
  call<number>("generate_clips", { projectId, options });
export const clipsPreflight = (projectId: number, options: ClipOptions) =>
  call<Preflight>("clips_preflight", { projectId, options });
export const listClipSets = (projectId: number) => call<ClipSet[]>("list_clip_sets", { projectId });
export const getClipSet = (setId: string) => call<ClipSet>("get_clip_set", { setId });
export const setClipStatus = (setId: string, scriptId: number, status: ClipStatus) =>
  call<ClipSet>("set_clip_status", { setId, scriptId, status });
export const deleteClipSet = (setId: string) => call<TrashItem>("delete_clip_set", { setId });
export const exportClipSet = (setId: string, dir: string, acceptedOnly: boolean) =>
  call<number>("export_clip_set", { setId, dir, acceptedOnly });

// ---- Agent pool ----------------------------------------------------------------------------

export type AgentRole = "orchestrator" | "worker" | "chat" | "vision";

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";
export const EFFORTS: Effort[] = ["low", "medium", "high", "xhigh", "max"];
/** The CLI tools that take an effort; the others ignore it. */
export const EFFORT_TOOLS = ["claude", "codex"];

/** One configured agent (`[[agents]]`). */
export interface AgentSpec {
  id: string;
  name?: string;
  /** Linked to a config section whose model settings it uses. */
  source?: "chat_model" | "chat_worker" | "vision" | null;
  backend: Backend;
  tool?: string;
  model?: string;
  /** How hard the model thinks (claude, codex); absent = the tool's default. */
  effort?: Effort | null;
  url?: string;
  /** Write-only: never sent back; empty keeps the stored key. */
  api_key?: string;
  local_model?: string;
  ctx_tokens?: number;
  timeout_secs?: number;
  max_concurrent: number;
  roles: AgentRole[];
  enabled: boolean;
  priority: number;
}

/** A choice of agents; every field optional (run overrides cut overrides app). */
export interface AgentSelection {
  orchestrator?: string | null;
  /** No orchestrator model: code picks from the workers' scores. */
  pick_by_score?: boolean | null;
  workers?: string[] | null;
  max_parallel?: number | null;
  /** Calls at once per worker, by agent id (a local model stays at one). */
  concurrency?: Record<string, number> | null;
}

/** What a run makes, so the summary can name it (`agents::WorkTarget`). */
export interface WorkTarget {
  count?: number | null;
  format?: ClipFormat | null;
}

export type SelectionSource = "run" | "cut" | "app" | "default";

export interface ResolvedAgents {
  orchestrator: string | null;
  orchestrator_from: SelectionSource;
  workers: string[];
  workers_from: SelectionSource;
  chat: string | null;
  max_parallel: number;
  capacity: number;
  /** "up to 6 at once: claude ×2 · codex ×2 · ghostpen ×2" */
  capacity_line: string;
  /** Chosen to run without an orchestrator model. */
  picks_by_score: boolean;
  /** Calls at once each worker takes, by id. */
  concurrency: Record<string, number>;
  /** One plain sentence on who does what, from the backend. */
  summary: string;
}

/** Where an agent stands now (`agents::AgentStatus`). */
export type AgentState =
  | { state: "ready" }
  | { state: "busy"; in_use: number; capacity: number }
  | { state: "out_of_quota"; until: number }
  | { state: "not_installed" }
  | { state: "unreachable" }
  | { state: "unavailable" };

export type AgentStatus = AgentState & { id: string; label: string; detail: string };

export interface AgentView {
  spec: AgentSpec;
  effective: AgentSpec;
  has_api_key: boolean;
  capacity: number;
  capacity_locked: string | null;
  linked_to: string | null;
  summary: string;
  /** Its name, or its id when it has none. */
  display_name: string;
}

export interface AgentsView {
  implicit: boolean;
  agents: AgentView[];
  app_selection: AgentSelection;
  resolved: ResolvedAgents | null;
  error: string | null;
  roles: AgentRole[];
  cli_tools: string[];
}

export type CheckLevel = "ok" | "warn" | "error";

export interface AgentAvailability {
  id: string;
  ok: boolean;
  level: CheckLevel;
  resolved: string;
  detail: string;
  model_listed: boolean | null;
  error?: AgentError | null;
}

/** One thing checked before a run (`preflight::Check`). */
export interface PreflightCheck {
  level: CheckLevel;
  scope: "agent" | "project";
  /** The agent id, or the indexing stage. */
  subject: string;
  message: string;
  error?: AgentError | null;
}

export interface Preflight {
  checks: PreflightCheck[];
}

export const listAgents = () => call<AgentsView>("list_agents");
export const saveAgents = (agents: AgentSpec[], selection: AgentSelection) =>
  call<AgentsView>("save_agents", { agents, selection });
export const testAgent = (id: string) => call<AgentAvailability>("test_agent", { id });
export const resolveAgents = (run?: AgentSelection | null, cut?: AgentSelection | null, target?: WorkTarget | null) =>
  call<ResolvedAgents>("resolve_agents", target ? { run: run ?? null, cut: cut ?? null, target } : { run: run ?? null, cut: cut ?? null });
export const agentStatus = () => call<AgentStatus[]>("agent_status");
