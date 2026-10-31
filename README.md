# VOXEL-SAM AR

3D reverse-engineering pipeline for Augmented Reality: record a 360° video of an
everyday object, isolate its geometry, extract 3D vertices, refine the mesh with
an autonomous agent (feedback loop, max 10 cycles) and publish a WebAR-ready
`.GLB` / `.USDZ`.

```
360° .mp4 ──▶ [1. Modal GPU: SAM + tracking] ──▶ 24 black-background keyframes
                    ──▶ [2. Baseten: muse-spark-open-1026] ──▶ geometry.json
                    ──▶ [3. Muse Spark 1.3 max agent: Trimesh/Open3D, retries] ──▶ model.glb + model.usdz
                    ──▶ [4. Vite web + model-viewer] ──▶ QR → AR on the phone
```

## Status

- [x] Phase 0 — Repo base
- [x] Phase 1 — `modal/app.py`: FastAPI + Modal GPU + SAM + tracking + 24 keyframes
- [x] Phase 2 — `baseten/`: `geometry.json` via `muse-spark-open-1026` + silhouette fallback
- [x] Phase 3 — `agent/`: watertight feedback loop (baseline + model, ≤10 cycles, GLB+USDZ)
- [x] Phase 4 — `web/`: Vite console + `model-viewer` viewer + WebAR QR

Detailed file plan: [docs/PLAN.md](docs/PLAN.md).

## Quickstart — Step 1 (Modal + SAM)

Requirements: Python 3.11+, Modal account (`pip install modal && modal setup`).

```bash
# 1. Local deps (syntax / type checking only)
pip install -r modal/requirements.txt

# 2. Local entrypoint test (needs remote Modal GPU)
modal run modal/app.py --video ./assets/demo.mp4 --x 320 --y 240 --num-keyframes 24

# 3. Deploy the API
modal deploy modal/app.py
# → https://<your-user>--voxel-sam-ar-ingest-fastapi-app.modal.run
```

### SAM checkpoint configuration

`modal/app.py` resolves the model in this order (no placeholders, real fallback):

1. `SAM3_MODULE` + `SAM31_CHECKPOINT` (hackathon `sam3` wheel, if present).
2. `SAM2_CHECKPOINT` (`sam2` package, video predictor).
3. Classic per-frame SAM ViT-H (`segment_anything`) + Lucas-Kanade optical-flow
   tracking. This path always works with `modal/requirements.txt`.

Environment variables (or Modal secret `voxel-sam-ar`):

| Variable | Default | Description |
|---|---|---|
| `SAM31_CHECKPOINT` | `/models/sam-3.1/sam31_h.pt` | SAM 3.1 checkpoint (.pt) |
| `SAM2_CHECKPOINT` | `/models/sam-2/sam2_h.pt` | SAM 2 checkpoint (.pt) |
| `SAM_VITH_CHECKPOINT` | `/models/sam/sam_vit_h_4b8939.pth` | SAM ViT-H checkpoint |
| `SAM_MODEL_TYPE` | `vit_h` | `vit_h` / `vit_l` / `vit_b` for classic SAM |
| `MAX_VIDEO_MB` | `200` | Max .mp4 size |
| `MAX_FRAMES` | `1500` | Decoded frames limit |
| `TARGET_SHORT_SIDE` | `720` | Short-side resize (speed/VRAM) |

## API — Step 1

### `GET /health`

```bash
curl https://<app>.modal.run/health
# {"status":"ok","sam_backend":"sam_vith_optflow","cuda_available":true,...}
```

### `POST /segment` (multipart)

Fields: `file` (.mp4), `x` (float, px in original frame), `y` (float),
`num_keyframes` (default 24, 4–64), `job_id` (optional).

```bash
curl -X POST https://<app>.modal.run/segment \
  -F "file=@demo.mp4;type=video/mp4" \
  -F "x=320" -F "y=240" -F "num_keyframes=24"
```

Response (summary):

```json
{
  "job_id": "seg_9f3a...",
  "num_keyframes": 24,
  "width": 1280,
  "height": 720,
  "keyframes": [{"index": 0, "png_b64": "...", "mask_b64": "...", "bbox_xyxy": [x1,y1,x2,y2]}],
  "meta": {"sam_backend": "sam_vith_optflow", "fps": 30.0, "total_frames": 312}
}
```

Each `png_b64` is a PNG with the segmented object on a **black background**
(purged alpha). `mask_b64` is the binary mask (gray PNG). The same PNGs persist
to `modal.Volume("voxel-sam-ar-jobs")` under `jobs/{job_id}/keyframe_XX.png`.

### `POST /segment-b64` (JSON)

```json
{"video_b64": "<base64 mp4>", "x": 320, "y": 240, "num_keyframes": 24}
```

### `GET /jobs/{job_id}`

Lists the job's persisted artifacts (useful for Phase 2 / Baseten).

## Quickstart — Step 2 (3D geometry)

```bash
pip install -r baseten/requirements.txt

# With Baseten (primary remote, strict JSON via Instructor):
export BASETEN_API_KEY=...  # BASETEN_MODEL_NAME default: muse-spark-open-1026
python -m baseten.infer --keyframes-dir ./keyframes_out/<job_id> --meta ./keyframes_out/<job_id>/meta.json

# Without a key (deterministic silhouette fallback, numpy+PIL):
python -m baseten.infer --keyframes-dir ./keyframes_out/<job_id> --no-remote --height-m 0.20
# → <keyframes-dir>/geometry.json  {vertices, normals, edges, faces?, confidence, ...}
```

Contract: meters, Y-up, centered object; `vertices[i]` ↔ unit `normals[i]`;
see [baseten/schema.py](baseten/schema.py). If remote fails or no key is set,
the CLI degrades to the local fallback with exit 0 (warning on stderr).

## Quickstart — Step 3 (mesh feedback loop)

```bash
pip install -r agent/requirements.txt

# Deterministic baseline (no model): triangulate/repair/audit/export
python -m agent.loop --geometry ./geometry.json --exec local --max-cycles 1

# Full agentic loop (≤10 cycles) in an isolated Modal Sandbox:
export META_MODEL_API_KEY=... META_MODEL_BASE_URL=...  # OpenAI-compatible API
python -m agent.loop --geometry ./geometry.json --exec modal --max-cycles 10
# → agent/runs/<run_id>/{cycle_XX.py, cycle_XX.log, cycle_XX.report.json,
#                        manifest.json, model.glb, model.usdz}
```

Cycle 0 is the baseline; if it fails and a key exists, the model receives
`stderr` + `MeshReport` and rewrites the script (Poisson → ball pivoting →
alpha + repair/taubin ladder). The USDZ is authored as USDA + STORED zip and
revalidated with `usd-core` when available. Exit 0 = watertight mesh, 0 errors.

## Quickstart — Step 4 (WebAR console)

```bash
cd web
npm install
npm run dev              # http://localhost:5173 — open in the browser
```

The console talks live to Phase 1 (`/health`, `/segment`), shows any Phase 3
run manifest, and presents the `model-viewer` viewer (GLB + USDZ) with a QR to
this same page for opening it on the phone and tapping AR. Includes real demo
artifacts in `web/public/{demo,models}`. Mobile demo details: [web/README.md](web/README.md).

## Layout

```
voxel-sam-ar/
  README.md  LICENSE  .gitignore
  docs/PLAN.md
  modal/
    app.py            # STEP 1 complete
    requirements.txt
  baseten/
    schema.py         # STEP 2: GeometryEstimate contract (strict Pydantic)
    infer.py          # STEP 2: Baseten/Instructor client + silhouette fallback
    requirements.txt
  agent/
    mesh_tools.py     # STEP 3: triangulate/repair/smooth/audit/export
    prompts.py        # STEP 3: prompts + model code extraction
    baseline_poisson.py  # STEP 3: deterministic cycle 0
    loop.py           # STEP 3: orchestrator (local + Modal Sandbox, ≤10 cycles)
    requirements.txt
  web/                # STEP 4: Vite console + model-viewer + QR (see web/README.md)
```

## Full demo flow

1. `modal deploy modal/app.py` → live API (or use the current deployment).
2. `cd web && npm run dev` → segment your video from stage 2.
3. `python -m baseten.infer …` + `python -m agent.loop …` → `model.glb`/`.usdz`.
4. Copy the run into `web/public/` (or point `VITE_ARTIFACTS_URL` at it), reload,
   scan the QR with the phone and tap AR next to the real object.
