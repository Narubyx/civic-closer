/** Typed client for the VOXEL-SAM AR backend (Phase 1: Modal + FastAPI). */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  checkpoints: Record<string, boolean>;
  limits: { max_video_mb: number; max_frames: number };
}

export interface KeyframeResult {
  index: number;
  png_b64: string;
  mask_b64: string;
  bbox_xyxy: [number, number, number, number];
  sharpness: number;
}

export interface SegmentMeta {
  sam_backend: string;
  fps: number;
  total_frames: number;
  width: number;
  height: number;
  scale: number;
  point_original: [number, number];
  point_scaled: [number, number];
  sam_calls: number;
}

export interface SegmentResponse {
  job_id: string;
  num_keyframes: number;
  width: number;
  height: number;
  keyframes: KeyframeResult[];
  meta: SegmentMeta;
}

export interface CycleReport {
  ok: boolean;
  errors: string[];
  stats: Record<string, number | string | boolean | null>;
}

export interface CycleRecord {
  cycle: number;
  kind: string;
  exit_code: number;
  duration_s: number;
  report: CycleReport;
  stderr_tail: string;
}

export interface RunManifest {
  run_id: string;
  backend: string;
  status: string;
  cycles: CycleRecord[];
  model: string | null;
  usdz: { ok?: boolean; converter?: string; error?: string; path?: string };
}

/** Mirrors geometry.json as written by baseten/infer.py. */
export interface GeometrySummary {
  source: string;
  vertices: [number, number, number][];
  normals: [number, number, number][];
  edges: [number, number][];
  faces: [number, number, number][] | null;
  source_keyframes: number[];
  confidence: number;
  estimated_height_m: number;
  notes?: string;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function fetchJson<T>(url: string, init?: RequestInit, timeoutMs = 30000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: ctrl.signal });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new ApiError(res.status, `${res.status} ${res.statusText}${text ? `: ${text.slice(0, 300)}` : ''}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(0, err instanceof Error ? err.message : String(err));
  } finally {
    clearTimeout(timer);
  }
}

export function fetchHealth(modalUrl: string): Promise<HealthResponse> {
  return fetchJson<HealthResponse>(`${modalUrl.replace(/\/$/, '')}/health`);
}

export function postSegment(
  modalUrl: string,
  file: File,
  x: number,
  y: number,
  numKeyframes: number,
): Promise<SegmentResponse> {
  const form = new FormData();
  form.append('file', file);
  form.append('x', String(x));
  form.append('y', String(y));
  form.append('num_keyframes', String(numKeyframes));
  // Segmenting on a cold GPU can take minutes: generous timeout.
  return fetchJson<SegmentResponse>(`${modalUrl.replace(/\/$/, '')}/segment`, { method: 'POST', body: form }, 900000);
}

export function fetchArtifact<T>(artifactsUrl: string, name: string): Promise<T> {
  return fetchJson<T>(`${artifactsUrl.replace(/\/$/, '')}/${name}`, undefined, 15000);
}
