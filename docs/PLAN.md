# VOXEL-SAM AR — Plan de ejecución de archivos

Pipeline de ingeniería inversa 3D para WebAR: video 360° → segmentación SAM →
nube de vértices → malla watertight vía agente con feedback loop → .GLB/.USDZ → visor WebAR.

Orden estricto de creación. Cada fase cierra con un criterio de aceptación verificable
por terminal / API, sin GUI en el backend.

---

## Fase 0 — Base del repo (esta entrega)

| # | Archivo | Propósito | Aceptación |
|---|---------|-----------|------------|
| 0.1 | `README.md` | Visión, arquitectura, quickstart, ejemplos curl | `README.md` existe y documenta `/health` y `/segment` |
| 0.2 | `.gitignore` | Python, Node, Modal, artefactos de video/malla | `git status` limpio tras build local |
| 0.3 | `LICENSE` | MIT | Existe |
| 0.4 | `docs/PLAN.md` | Este archivo | Existe |
| 0.5 | `modal/requirements.txt` | Deps pineadas del Paso 1 (solo paquetes reales de PyPI) | `pip install -r modal/requirements.txt` OK en Python 3.11 |
| 0.6 | `modal/app.py` | **PASO 1 COMPLETO**: FastAPI + Modal GPU + SAM + tracking 360° + 24 keyframes con fondo negro | `python -m py_compile modal/app.py` OK; `modal deploy modal/app.py` expone `/health` y `/segment` |

## Fase 1 — Ingesta y visión (Modal GPU) — ENTREGADA AQUÍ

- `modal/app.py` (único archivo del paso):
  - `modal.App("voxel-sam-ar-ingest")`, imagen `debian_slim` Python 3.11, GPU A10G.
  - Clase `Segmenter` con `@modal.enter()` (carga perezosa del checkpoint SAM en GPU)
    y `@modal.method()` (segmentación + tracking + keyframes).
  - Adaptador `SamVideoAdapter`: intenta `sam3` → `sam2` → `segment_anything` (SAM ViT-H
    por frame + tracking por flujo óptico LK como fallback determinista).
  - FastAPI: `GET /health`, `POST /segment` (multipart: `file`, `x`, `y`,
    `num_keyframes=24`), `POST /segment-b64` (JSON), `GET /jobs/{job_id}`.
  - Salida: 24 PNG con fondo negro (alfa purgado), máscaras RLE-lite (base64),
    bboxes, índices de frame y metadatos. Persistencia en `modal.Volume`.
  - Entrypoint local `modal run modal/app.py --video ./demo.mp4 --x 320 --y 240`
    para prueba sin desplegar.

## Fase 2 — Extracción geométrica (Baseten, `muse-spark-open-1026`)

| # | Archivo | Propósito |
|---|---------|-----------|
| 2.1 | `baseten/schema.py` | Modelos Pydantic estrictos: `VertexMatrix`, `EdgeControlPoint`, `SurfaceNormal`, `GeometryEstimate` (validación de longitudes, rangos, unidades en metros). |
| 2.2 | `baseten/infer.py` | Cliente Baseten: envía lote de 24 keyframes (base64), usa Instructor/Outlines para forzar JSON-schema, reintenta con backoff, valida contra `schema.py`, guarda `geometry.json`. |
| 2.3 | `baseten/requirements.txt` | `baseten`, `instructor`, `outlines`, `pydantic>=2`, `httpx`. |

Contrato `geometry.json` (resumen): `vertices: float[N,3]`, `edges: int[M,2]`,
`normals: float[N,3]`, `units: "meters"`, `source_keyframes: int[24]`, `confidence: float`.

## Fase 3 — Motor agéntico + feedback loop (Muse Spark 1.3 max vía Meta Model API, Sandbox Modal)

| # | Archivo | Propósito |
|---|---------|-----------|
| 3.1 | `agent/mesh_tools.py` | API determinista de malla: `triangulate()`, `repair_watertight()`, `smooth()`, `check_topology()` (detecta caras cruzadas, huecos, no-manifold, normales invertidas) con Open3D + Trimesh + SciPy. Cada chequeo devuelve `MeshReport{ok, errors[], stats{}}`. |
| 3.2 | `agent/prompts.py` | System prompt + plantilla de reintento: recibe `geometry.json` + `stderr` + `MeshReport` y debe devolver SOLO un script Python ejecutable que llama a `mesh_tools`. Incluye presupuesto de 10 ciclos y reglas de escalada de parámetros. |
| 3.3 | `agent/loop.py` | Bucle autónomo: 1) pide script al modelo → 2) ejecuta en Sandbox Modal con timeout → 3) captura `stdout/stderr` + `MeshReport` → 4) si falla, reinyecta error y reintenta (máx 10) → 5) al éxito exporta `model.glb` + convierte a `.usdz` vía `assimp`/`blender --background --python`. Registra cada ciclo en `agent/runs/{run_id}/cycle_XX.{py,log,json}`. |
| 3.4 | `agent/requirements.txt` | `open3d`, `trimesh[easy]`, `scipy`, `numpy`, `boto3`/`httpx` (Meta Model API). |

Criterio de aceptación Fase 3: malla con `watertight=true`, `winding_consistent=true`,
`euler_valid=true`, 0 errores topológicos, y `model.glb` + `model.usdz` generados.

## Fase 4 — Presentación WebAR (TypeScript)

| # | Archivo | Propósito |
|---|---------|-----------|
| 4.1 | `web/package.json`, `web/vite.config.ts` | App Vite + React + TS (ligera; alternativa Next.js documentada en README). |
| 4.2 | `web/src/App.tsx` | Subida de video, inputs (x,y), visor de progreso del agente (SSE `/events`), descarga GLB/USDZ. |
| 4.3 | `web/src/components/ArViewer.tsx` | `@google/model-viewer`: render 3D + botón AR (Scene Viewer / Quick Look), QR generado en cliente (`qrcode.react`) apuntando a la URL pública del `.usdz`/`.glb`. |
| 4.4 | `web/src/lib/api.ts` | Cliente tipado de `/segment`, `/geometry`, `/agent/runs`, `/events`. |

Criterio de aceptación Fase 4: subir video → ver log del agente en vivo → escanear QR →
modelo proyectado junto al objeto real en el móvil.

## Fase 5 — E2E, CI y demo

- `scripts/e2e.sh`: video de prueba → `/segment` → Baseten → agente → `model.glb`; falla si algún paso no devuelve 0 errores.
- `.github/workflows/ci.yml`: `py_compile` + `ruff` (backend), `tsc --noEmit` (web).
- `assets/demo.mp4` (opcional, <25 MB) + `assets/expected_geometry.json` para tests dorados.

---

## Reglas de implementación (todas las fases)

1. Sin placeholders: todo archivo entregado compila y corre; lo no entregado NO se crea.
2. Backend 100% API + terminal; la única GUI es `web/`.
3. Todo error de terminal (`stderr`) del agente se persiste y se reinyecta al modelo.
4. Artefactos versionados por `job_id`/`run_id` en volúmenes, nunca rutas fijas.
