# VOXEL-SAM AR

Pipeline de ingeniería inversa 3D para Realidad Aumentada: graba un video 360° de un
objeto cotidiano, aísla su geometría, extrae vértices 3D, refina la malla con un agente
autónomo (feedback loop, máx 10 ciclos) y publica un `.GLB` / `.USDZ` listo para WebAR.

```
.mp4 360° ──▶ [1. Modal GPU: SAM + tracking] ──▶ 24 keyframes fondo negro
                    ──▶ [2. Baseten: muse-spark-open-1026] ──▶ geometry.json
                    ──▶ [3. Agente Muse Spark 1.3 max: Trimesh/Open3D, reintentos] ──▶ model.glb + model.usdz
                    ──▶ [4. Web Vite + model-viewer] ──▶ QR → AR en el móvil
```

## Estado

- [x] Fase 0 — Base del repo
- [x] Fase 1 — `modal/app.py`: FastAPI + Modal GPU + SAM + tracking + 24 keyframes
- [x] Fase 2 — `baseten/`: `geometry.json` vía `muse-spark-open-1026` + fallback por siluetas
- [ ] Fase 3 — `agent/` feedback loop watertight
- [ ] Fase 4 — `web/` visor WebAR + QR

Plan detallado de archivos: [docs/PLAN.md](docs/PLAN.md).

## Quickstart — Paso 1 (Modal + SAM)

Requisitos: Python 3.11+, cuenta Modal (`pip install modal && modal setup`).

```bash
# 1. Dependencias locales (solo para validación de sintaxis / tipos)
pip install -r modal/requirements.txt

# 2. Prueba local del entrypoint (requiere GPU remota de Modal)
modal run modal/app.py --video ./assets/demo.mp4 --x 320 --y 240 --num-keyframes 24

# 3. Despliegue del API
modal deploy modal/app.py
# → https://<tu-usuario>--voxel-sam-ar-ingest-fastapi-app.modal.run
```

### Configuración del checkpoint SAM

`modal/app.py` resuelve el modelo en este orden (sin placeholders, con fallback real):

1. `SAM3_MODULE` + `SAM31_CHECKPOINT` (rueda `sam3` del hackatón, si existe).
2. `SAM2_CHECKPOINT` (paquete `sam2`, predictor de video).
3. SAM ViT-H clásico (`segment_anything`) por frame + tracking por flujo óptico
   Lucas-Kanade. Este camino siempre funciona con `modal/requirements.txt`.

Variables de entorno (o secreto Modal `voxel-sam-ar`):

| Variable | Default | Descripción |
|---|---|---|
| `SAM31_CHECKPOINT` | `/models/sam-3.1/sam31_h.pt` | Checkpoint SAM 3.1 (.pt) |
| `SAM2_CHECKPOINT` | `/models/sam-2/sam2_h.pt` | Checkpoint SAM 2 (.pt) |
| `SAM_VITH_CHECKPOINT` | `/models/sam/sam_vit_h_4b8939.pth` | Checkpoint SAM ViT-H |
| `SAM_MODEL_TYPE` | `vit_h` | `vit_h` / `vit_l` / `vit_b` para SAM clásico |
| `MAX_VIDEO_MB` | `200` | Tamaño máximo del .mp4 |
| `MAX_FRAMES` | `1500` | Límite de frames decodificados |
| `TARGET_SHORT_SIDE` | `720` | Redimensionado del lado corto (velocidad/VRAM) |

## API — Paso 1

### `GET /health`

```bash
curl https://<app>.modal.run/health
# {"status":"ok","sam_backend":"sam_vith_optflow","cuda_available":true,...}
```

### `POST /segment` (multipart)

Campos: `file` (.mp4), `x` (float, px en frame original), `y` (float),
`num_keyframes` (default 24, 4–64), `job_id` (opcional).

```bash
curl -X POST https://<app>.modal.run/segment \
  -F "file=@demo.mp4;type=video/mp4" \
  -F "x=320" -F "y=240" -F "num_keyframes=24"
```

Respuesta (resumen):

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

Cada `png_b64` es un PNG con el objeto segmentado sobre **fondo negro** (alfa purgado).
`mask_b64` es la máscara binaria (PNG gris). Los mismos PNG se persisten en el
`modal.Volume("voxel-sam-ar-jobs")` bajo `jobs/{job_id}/keyframe_XX.png`.

### `POST /segment-b64` (JSON)

```json
{"video_b64": "<mp4 en base64>", "x": 320, "y": 240, "num_keyframes": 24}
```

### `GET /jobs/{job_id}`

Lista artefactos persistidos del job (útil para la Fase 2 / Baseten).

## Quickstart — Paso 2 (geometría 3D)

```bash
pip install -r baseten/requirements.txt

# Con Baseten (remoto primario, JSON estricto vía Instructor):
export BASETEN_API_KEY=...  # BASETEN_MODEL_NAME default: muse-spark-open-1026
python -m baseten.infer --keyframes-dir ./keyframes_out/<job_id> --meta ./keyframes_out/<job_id>/meta.json

# Sin clave (fallback determinista por siluetas, numpy+PIL):
python -m baseten.infer --keyframes-dir ./keyframes_out/<job_id> --no-remote --height-m 0.20
# → <keyframes-dir>/geometry.json  {vertices, normals, edges, faces?, confidence, ...}
```

Contrato: metros, Y-up, objeto centrado; `vertices[i]` ↔ `normals[i]` unitaria;
ver [baseten/schema.py](baseten/schema.py). Si el remoto falla o no hay clave,
el CLI degrada al fallback local con exit 0 (aviso en stderr).

## Estructura

```
voxel-sam-ar/
  README.md  LICENSE  .gitignore
  docs/PLAN.md
  modal/
    app.py            # PASO 1 completo
    requirements.txt
  baseten/
    schema.py         # PASO 2: contrato GeometryEstimate (Pydantic estricto)
    infer.py          # PASO 2: cliente Baseten/Instructor + fallback por siluetas
    requirements.txt
  agent/              # Fase 3 (pendiente)
  web/                # Fase 4 (pendiente)
```

## Roadmap inmediato

1. `agent/loop.py` + `agent/mesh_tools.py` (feedback loop watertight, 10 ciclos).
2. `web/` (Vite + model-viewer + QR).
