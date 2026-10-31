# VOXEL-SAM AR — File execution plan

3D reverse-engineering pipeline for WebAR: 360° video → SAM segmentation →
vertex cloud → watertight mesh via an agent with a feedback loop → `.GLB`/`.USDZ`
→ WebAR viewer.

Strict creation order. Each phase closes with an acceptance criterion verifiable
via terminal / API, with no GUI on the backend.

---

## Phase 0 — Repo base (this delivery)

| # | File | Purpose | Acceptance |
|---|------|---------|------------|
| 0.1 | `README.md` | Vision, architecture, quickstart, curl examples | `README.md` exists and documents `/health` and `/segment` |
| 0.2 | `.gitignore` | Python, Node, Modal, video/mesh artifacts | Clean `git status` after local build |
| 0.3 | `LICENSE` | MIT | Exists |
| 0.4 | `docs/PLAN.md` | This file | Exists |
| 0.5 | `modal/requirements.txt` | Pinned Step 1 deps (real PyPI packages only) | `pip install -r modal/requirements.txt` OK on Python 3.11 |
| 0.6 | `modal/app.py` | **STEP 1 COMPLETE**: FastAPI + Modal GPU + SAM + 360° tracking + 24 black-background keyframes | `python -m py_compile modal/app.py` OK; `modal deploy modal/app.py` exposes `/health` and `/segment` |

## Phase 1 — Ingest & vision (Modal GPU) — DELIVERED HERE

- `modal/app.py` (the step's only file):
  - `modal.App("voxel-sam-ar-ingest")`, `debian_slim` Python 3.11 image, A10G GPU.
  - `Segmenter` class with `@modal.enter()` (lazy SAM checkpoint load on GPU)
    and `@modal.method()` (segmentation + tracking + keyframes).
  - `SamVideoAdapter`: tries `sam3` → `sam2` → `segment_anything` (per-frame SAM
    ViT-H + Lucas-Kanade optical-flow tracking as the deterministic fallback).
  - FastAPI: `GET /health`, `POST /segment` (multipart: `file`, `x`, `y`,
    `num_keyframes=24`), `POST /segment-b64` (JSON), `GET /jobs/{job_id}`.
  - Output: 24 PNGs with black background (purged alpha), plus RLE-lite base64
    masks, bboxes, frame indices and metadata. Persisted to a `modal.Volume`.
  - Local entrypoint `modal run modal/app.py --video ./demo.mp4 --x 320 --y 240`
    for testing without deploying.

## Phase 2 — Geometric extraction (Baseten, `muse-spark-open-1026`)

| # | File | Purpose |
|---|------|---------|
| 2.1 | `baseten/schema.py` | Strict Pydantic models: `VertexMatrix`, `EdgeControlPoint`, `SurfaceNormal`, `GeometryEstimate` (length, range and meter-unit validation). |
| 2.2 | `baseten/infer.py` | Baseten client: sends the 24-keyframe batch (base64), uses Instructor/Outlines to force the JSON schema, retries with backoff, validates against `schema.py`, saves `geometry.json`. |
| 2.3 | `baseten/requirements.txt` | `baseten`, `instructor`, `outlines`, `pydantic>=2`, `httpx`. |

`geometry.json` contract (summary): `vertices: float[N,3]`, `edges: int[M,2]`,
`normals: float[N,3]`, `units: "meters"`, `source_keyframes: int[24]`, `confidence: float`.

## Phase 3 — Agentic engine + feedback loop (Muse Spark 1.3 max via Meta Model API, Modal Sandbox)

| # | File | Purpose |
|---|------|---------|
| 3.1 | `agent/mesh_tools.py` | Deterministic mesh API: `triangulate()`, `repair_watertight()`, `smooth()`, `check_topology()` (detects crossed faces, holes, non-manifold, flipped normals) with Open3D + Trimesh + SciPy. Every check returns `MeshReport{ok, errors[], stats{}}`. |
| 3.2 | `agent/prompts.py` | System prompt + retry template: receives `geometry.json` + `stderr` + `MeshReport` and must return ONLY an executable Python script calling `mesh_tools`. Includes the 10-cycle budget and parameter-escalation rules. |
| 3.3 | `agent/loop.py` | Autonomous loop: 1) asks the model for a script → 2) runs it in the Modal Sandbox with timeout → 3) captures `stdout/stderr` + `MeshReport` → 4) on failure, re-injects the error and retries (max 10) → 5) on success exports `model.glb` + converts to `.usdz` via `assimp`/`blender --background --python`. Logs every cycle to `agent/runs/{run_id}/cycle_XX.{py,log,json}`. |
| 3.4 | `agent/requirements.txt` | `open3d`, `trimesh[easy]`, `scipy`, `numpy`, `boto3`/`httpx` (Meta Model API). |

Phase 3 acceptance: mesh with `watertight=true`, `winding_consistent=true`,
`euler_valid=true`, 0 topology errors, and generated `model.glb` + `model.usdz`.

## Phase 4 — WebAR presentation (TypeScript)

| # | File | Purpose |
|---|------|---------|
| 4.1 | `web/package.json`, `web/vite.config.ts` | Vite + React + TS app (light; Next.js alternative documented in README). |
| 4.2 | `web/src/App.tsx` | Video upload, (x,y) inputs, agent progress view (SSE `/events`), GLB/USDZ download. |
| 4.3 | `web/src/components/ArViewer.tsx` | `@google/model-viewer`: 3D render + AR button (Scene Viewer / Quick Look), client-side QR (`qrcode.react`) pointing at the public `.usdz`/`.glb` URL. |
| 4.4 | `web/src/lib/api.ts` | Typed client for `/segment`, `/geometry`, `/agent/runs`, `/events`. |

Phase 4 acceptance: upload video → watch the agent log live → scan QR →
model projected next to the real object on the phone.

## Phase 5 — E2E, CI & demo

- `scripts/e2e.sh`: test video → `/segment` → Baseten → agent → `model.glb`; fails if any step returns errors.
- `.github/workflows/ci.yml`: `py_compile` + `ruff` (backend), `tsc --noEmit` (web).
- `assets/demo.mp4` (optional, <25 MB) + `assets/expected_geometry.json` for golden tests.

---

## Implementation rules (all phases)

1. No placeholders: every delivered file compiles and runs; undelivered work is NOT created.
2. Backend is 100% API + terminal; the only GUI is `web/`.
3. Every agent terminal error (`stderr`) is persisted and re-injected into the model.
4. Artifacts are versioned by `job_id`/`run_id` on volumes, never fixed paths.
