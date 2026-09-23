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
  /** static, tripod, stabilised, handheld, or unknown. */
  camera: string;
}

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
  phase_done: number;
  phase_total: number;
  fraction: number;
  eta_secs: number | null;
  elapsed_secs: number;
  current: string | null;
  /** No measurable end — the bar animates rather than claiming a percentage. */
  indeterminate?: boolean;
}

export const PHASE_LABELS: Record<string, string> = {
  researching: "Reading the footage",
  drafting: "Writing the script",
  hash: "Reading new files",
  probe: "Reading video details",
  download: "Downloading speech model",
  download_model: "Downloading model",
  transcribe_server: "Transcribing (GhostPen)",
  transcribe_local: "Transcribing",
  frames: "Picking keyframes",
  download_vision: "Downloading vision model",
  describe_server: "Describing frames",
  describe_local: "Describing frames",
};

/** "a few seconds", "42 s", "about 2 min", "about 1 h 20 min" — same wording as the CLI. */
export const etaText = (secs: number) => {
  const s = Math.max(0, Math.round(secs));
  if (s < 10) return "a few seconds";
  if (s < 60) return `${s} s`;
  if (s < 3600) return `about ${Math.ceil(s / 60)} min`;
  return `about ${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")} min`;
};

export interface IndexSummary {
  new: number;
  changed: number;
  unchanged: number;
  removed: number;
  jobs_done: number;
  jobs_failed: number;
  unsettled: number;
  cancelled: boolean;
}

export type TaskState = "queued" | "running" | "done" | "failed" | "cancelled";

export type TaskKind =
  | { type: "index"; project_id: number }
  | { type: "chat"; project_id: number; session_id: number }
  | { type: "render_preview"; script_id: number; burn_titles: boolean; burn_narration: boolean; normalize_audio: boolean }
  | { type: "export"; script_id: number; format: string; path: string }
  | { type: "download_model"; model_id: string };

export interface Task {
  id: number;
  kind: TaskKind;
  label: string;
  state: TaskState;
  progress: Progress | null;
  note: string | null;
  summary: IndexSummary | null;
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
/** `out`: save the rendered MP4 there (demo export) instead of the previews folder. */
export const enqueuePreview = (
  scriptId: number,
  burnTitles: boolean,
  burnNarration: boolean,
  normalizeAudio: boolean,
  out?: string,
) => call<number>("enqueue_preview", { scriptId, burnTitles, burnNarration, normalizeAudio, out: out ?? null });
export const enqueueExport = (scriptId: number, format: string, path: string) =>
  call<number>("enqueue_export", { scriptId, format, path });
export const previewPlan = (scriptId: number) => call<PlannedSegment[]>("preview_plan", { scriptId });
export const getScriptPreview = (scriptId: number) =>
  call<string | null>("get_script_preview", { scriptId });
export const queueList = () => call<Task[]>("queue_list");
export const cancelTask = (id: number) => call<boolean>("cancel_task", { id });
export const clearFinishedTasks = () => call<void>("clear_finished_tasks");

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
}

export const search = (projectId: number, query: string, limit = 30) =>
  call<{ hits: Hit[]; note: string | null }>("search", { projectId, query, limit });
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

export const clock = (s: number) => {
  const t = Math.max(0, Math.floor(s));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = String(t % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
};

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
export const projectView = (projectId: number) => call<ProjectView>("project_view", { projectId });
export const addFolder = (projectId: number, path: string, recursive = true) =>
  call<void>("add_folder", { projectId, path, recursive });
export const removeFolder = (projectId: number, path: string) => call<void>("remove_folder", { projectId, path });


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
  bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;

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
  why?: string;
}

/** Sound that runs across a beat while its pictures change under it — a J-cut. */
export interface AudioBed {
  video_id: number;
  in_s: number;
  out_s: number;
  why?: string;
  /** True when the pipeline laid it rather than the editor asking for it. */
  inferred?: boolean;
}

export interface Beat {
  id: string;
  purpose: string;
  narration?: string;
  on_screen_text?: string;
  clips: ScriptClip[];
  bed?: AudioBed | null;
  notes?: string;
}

export interface Script {
  title: string;
  target_duration_s?: number;
  fps?: Fps;
  width?: number;
  height?: number;
  beats: Beat[];
}

export interface Issue {
  severity: "error" | "warning" | "info";
  beat_id: string | null;
  clip_index: number | null;
  message: string;
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
}

export interface ChatSession {
  id: number;
  project_id: number;
  title: string;
  created_at: number;
  updated_at: number;
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

export const chatTurn = (projectId: number, sessionId: number | null, message: string, images?: string[]) =>
  call<ChatTurnView>("chat_turn", { projectId, sessionId, message, images: images ?? [] });

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

export const listScripts = (projectId: number) =>
  call<ScriptSummary[]>("list_scripts", { projectId });

export const getScript = (scriptId: number) =>
  call<ScriptView>("get_script", { scriptId });

export const saveScript = (projectId: number, script: Script, sessionId: number | null) =>
  call<SaveScriptView>("save_script", { projectId, script, sessionId });

// ---- models management -------------------------------------------------------------------

export type ModelKind = "whisper" | "vision" | "vision_projector" | "embedding";

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
}

export interface ModelStatus {
  entry: CatalogEntry;
  installed_path: string | null;
  in_own_dir: boolean;
  partial_bytes: number | null;
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
