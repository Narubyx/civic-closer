# VOXEL-SAM AR — WebAR Console (Phase 4)

Vite + React + [`@google/model-viewer`](https://modelviewer.dev) + QR. Four
stages, all against real backends (no mocks):

1. **Modal backend (live)** — `GET /health` of the Phase 1 deployment.
2. **Segment (live)** — upload an `.mp4` + `(x, y)` point to `POST /segment`
   and display the keyframes returned by the GPU.
3. **Geometry + agent** — reads `manifest.json` + `geometry.json` from the
   artifacts URL (default: `./demo`, a copy of a real run). For your own run:
   `python -m agent.loop …` and serve the dir with `python -m http.server`.
4. **WebAR viewer + QR** — `model-viewer` with GLB (Android/desktop) and USDZ
   (iOS Quick Look); the QR points at this page's own URL.

## Develop

```bash
cd web
cp .env.example .env   # adjust VITE_MODAL_URL if you redeployed
npm install
npm run dev            # http://localhost:5173
```

## Demo on a phone (real AR)

```bash
npm run build
npx vite preview --host --port 4173   # or host dist/ anywhere
```

Open the URL on the phone (same network if local) or scan the QR shown by the
page itself; tap **"View in your space (AR)"**.

## Bundled artifacts

`public/demo/` (sample manifest + geometry + keyframe) and `public/models/`
(`model.glb` + `model.usdz`) are real outputs of the Phase 1–3 pipeline on the
synthetic test video. Replace them with your own run for the final demo.
