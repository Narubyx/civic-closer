"""VOXEL-SAM AR — Paso 1: ingesta y segmentación 360° en Modal (GPU) + FastAPI + SAM.

Flujo:
  1. Cliente sube un .mp4 360° + punto inicial (x, y) del primer frame.
  2. El backend decodifica el video con OpenCV, inicializa SAM con ese punto,
     trackea el objeto a lo largo del video y selecciona N keyframes.
  3. Cada keyframe se exporta como PNG con el objeto sobre fondo negro
     (canal alfa purgado) + máscara binaria + bbox. Los artefactos se
     persisten en un modal.Volume bajo jobs/{job_id}/ y se devuelven en JSON.

Resolución del backend SAM (en orden, sin placeholders):
  1. "sam3" ......... rueda del hackatón (env SAM3_MODULE + fichero SAM31_CHECKPOINT).
  2. "sam2" ......... predictor de video SAM 2 (fichero SAM2_CHECKPOINT).
  3. "sam_vith_optflow" SAM ViT-H clásico por frame + tracking Lucas-Kanade.
  4. "grabcut_optflow" GrabCut + Lucas-Kanade (último recurso sin pesos).

Despliegue:
  modal deploy modal/app.py
Prueba local (usa GPU remota):
  modal run modal/app.py --video ./assets/demo.mp4 --x 320 --y 240
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

import cv2
import modal
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voxel-sam-ar.ingest")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


SAM31_CHECKPOINT = os.environ.get("SAM31_CHECKPOINT", "/models/sam-3.1/sam31_h.pt")
SAM3_MODULE = os.environ.get("SAM3_MODULE", "sam3")
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "/models/sam-2/sam2_h.pt")
SAM_VITH_CHECKPOINT = os.environ.get(
    "SAM_VITH_CHECKPOINT", "/models/sam/sam_vit_h_4b8939.pth"
)
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")
MAX_VIDEO_MB = _env_int("MAX_VIDEO_MB", 200)
MAX_VIDEO_BYTES = MAX_VIDEO_MB * 1024 * 1024
MAX_FRAMES = _env_int("MAX_FRAMES", 1500)
TARGET_SHORT_SIDE = _env_int("TARGET_SHORT_SIDE", 720)
DEFAULT_NUM_KEYFRAMES = 24
MAX_SAM_CALLS = _env_int("MAX_SAM_CALLS", 128)  # tope de inferencias SAM por video
FLOW_WINDOW = _env_int("FLOW_WINDOW", 21)  # ventana LK del tracker
SHARPNESS_WINDOW = _env_int("SHARPNESS_WINDOW", 4)  # búsqueda local ±frames

PIP_PACKAGES = (
    "fastapi>=0.111",
    "uvicorn>=0.30",
    "pydantic>=2.7",
    "python-multipart>=0.0.9",
    "opencv-python-headless>=4.9,<5",
    "numpy>=1.26,<2",
    "pillow>=10.3",
    "torch>=2.3,<2.6",
    "torchvision>=0.18,<0.21",
    "segment-anything>=1.0",
)

app = modal.App("voxel-sam-ar-ingest")
image = modal.Image.debian_slim(python_version="3.11").pip_install(*PIP_PACKAGES)
volume = modal.Volume.from_name("voxel-sam-ar-jobs", create_if_missing=True)
JOBS_ROOT = "/jobs/jobs"

# ---------------------------------------------------------------------------
# Esquemas API
# ---------------------------------------------------------------------------


class KeyframeResult(BaseModel):
    index: int = Field(description="Índice del frame en el video redimensionado")
    png_b64: str = Field(description="PNG base64: objeto sobre fondo negro")
    mask_b64: str = Field(description="PNG base64 en gris: máscara binaria")
    bbox_xyxy: list[int] = Field(description="[x1, y1, x2, y2] en px redimensionados")
    sharpness: float = Field(description="Varianza del Laplaciano (mayor = más nítido)")


class SegmentMeta(BaseModel):
    sam_backend: str
    fps: float
    total_frames: int
    width: int
    height: int
    scale: float
    point_original: list[float]
    point_scaled: list[float]
    sam_calls: int


class SegmentResponse(BaseModel):
    job_id: str
    num_keyframes: int
    width: int
    height: int
    keyframes: list[KeyframeResult]
    meta: SegmentMeta


class SegmentB64Request(BaseModel):
    video_b64: str = Field(description=".mp4 codificado en base64")
    x: float = Field(description="Coord. x del punto en el frame original (px)")
    y: float = Field(description="Coord. y del punto en el frame original (px)")
    num_keyframes: int = Field(default=DEFAULT_NUM_KEYFRAMES, ge=4, le=64)
    job_id: str | None = None


# ---------------------------------------------------------------------------
# Utilidades puras (CPU, testeables sin GPU)
# ---------------------------------------------------------------------------


class VideoError(ValueError):
    """Error de validación/decodificación del video de entrada."""


def new_job_id(prefix: str = "seg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def sanitize_job_id(job_id: str | None, prefix: str = "seg") -> str:
    if not job_id:
        return new_job_id(prefix)
    cleaned = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))[:64]
    if not cleaned:
        return new_job_id(prefix)
    return cleaned


def decode_video_bytes(video_bytes: bytes) -> bytes:
    if not video_bytes:
        raise VideoError("Video vacío (0 bytes).")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise VideoError(
            f"Video de {len(video_bytes) / 1e6:.1f} MB supera el límite de "
            f"{MAX_VIDEO_MB} MB."
        )
    # Firma ftyp del contenedor MP4/MOV (bytes 4..8). Acepta también WebM (EBML).
    if len(video_bytes) >= 12 and video_bytes[4:8] == b"ftyp":
        return video_bytes
    if len(video_bytes) >= 4 and video_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return video_bytes
    raise VideoError("El fichero no parece un .mp4/.mov válido (firma ftyp ausente).")


def read_video_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """Decodifica el video a frames BGR redimensionados. Devuelve (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoError(f"No se pudo abrir el video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames: list[np.ndarray] = []
        while len(frames) < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise VideoError("El video no contiene frames decodificables.")
    h0, w0 = frames[0].shape[:2]
    short = min(h0, w0)
    scale = TARGET_SHORT_SIDE / short if short > TARGET_SHORT_SIDE else 1.0
    if scale != 1.0:
        new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
        frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
    log.info(
        "Video decodificado: %d frames %.1f fps %dx%d -> scale %.3f",
        len(frames),
        fps,
        w0,
        h0,
        scale,
    )
    return frames, fps


def frame_sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_keyframe_indices(total: int, num: int) -> list[int]:
    """Índices uniformemente espaciados que cubren [0, total-1] (incluye extremos)."""
    if num >= total:
        return list(range(total))
    return sorted({int(round(i * (total - 1) / (num - 1))) for i in range(num)})


def refine_by_sharpness(
    frames: list[np.ndarray], candidates: list[int], window: int = SHARPNESS_WINDOW
) -> list[int]:
    """Mueve cada candidato al frame más nítido en ±window (búsqueda local)."""
    if window <= 0:
        return candidates
    gray_cache: dict[int, np.ndarray] = {}

    def sharp(i: int) -> float:
        g = gray_cache.get(i)
        if g is None:
            g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray_cache[i] = g
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    refined: list[int] = []
    used: set[int] = set()
    for c in candidates:
        lo, hi = max(0, c - window), min(len(frames) - 1, c + window)
        best = max((i for i in range(lo, hi + 1) if i not in used), key=sharp, default=c)
        # Si la ventana está ocupada, conserva el candidato aunque repita ventana.
        if best in used:
            best = c
        used.add(best)
        refined.append(best)
    return sorted(refined)


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        h, w = mask.shape
        return [0, 0, w, h]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def composite_black_bg(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aplica la máscara y purga el fondo a negro (alfa eliminado)."""
    out = np.zeros_like(frame_bgr)
    out[mask] = frame_bgr[mask]
    return out


def png_to_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Falló la codificación PNG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def clamp_point(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    return (min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0))


# ---------------------------------------------------------------------------
# Adaptador SAM con tracking 360°
# ---------------------------------------------------------------------------


class SamBackendError(RuntimeError):
    """El backend SAM solicitado no está disponible."""


SamBackendName = Literal["sam3", "sam2", "sam_vith_optflow", "grabcut_optflow"]


def available_backends() -> dict[str, bool]:
    return {
        "sam3": Path(SAM31_CHECKPOINT).exists(),
        "sam2": Path(SAM2_CHECKPOINT).exists(),
        "sam_vith_optflow": Path(SAM_VITH_CHECKPOINT).exists(),
        "grabcut_optflow": True,
    }


class SamVideoAdapter:
    """Segmenta + trackea un objeto en video. El backend se elige en `load()`.

    `segment_video()` devuelve (masks, backend_name, sam_calls) donde masks[i]
    es un array bool (H, W) alineado con frames[i].
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.backend: SamBackendName | None = None
        self._predictor: Any = None

    # -- carga ------------------------------------------------------------
    def load(self, preferred: SamBackendName | None = None) -> SamBackendName:
        order: list[SamBackendName] = (
            [preferred] if preferred else ["sam3", "sam2", "sam_vith_optflow", "grabcut_optflow"]
        )
        if preferred:
            order += [b for b in ("sam3", "sam2", "sam_vith_optflow", "grabcut_optflow") if b != preferred]
        errors: dict[str, str] = {}
        for name in order:
            try:
                getattr(self, f"_load_{name}")()
                self.backend = name
                log.info("Backend SAM activo: %s", name)
                return name
            except SamBackendError as exc:
                errors[name] = str(exc)
                continue
        raise SamBackendError(f"Ningún backend SAM disponible: {errors}")

    def _load_sam3(self) -> None:
        if not Path(SAM31_CHECKPOINT).exists():
            raise SamBackendError(f"Checkpoint ausente: {SAM31_CHECKPOINT}")
        try:
            module = __import__(SAM3_MODULE, fromlist=["create_predictor"])
        except ImportError as exc:
            raise SamBackendError(f"Módulo {SAM3_MODULE!r} no importable: {exc}") from exc
        factory = getattr(module, "create_predictor", None)
        if not callable(factory):
            raise SamBackendError(f"{SAM3_MODULE}.create_predictor no existe.")
        self._predictor = factory(SAM31_CHECKPOINT, self.device)

    def _load_sam2(self) -> None:
        if not Path(SAM2_CHECKPOINT).exists():
            raise SamBackendError(f"Checkpoint ausente: {SAM2_CHECKPOINT}")
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError as exc:
            raise SamBackendError(f"Paquete sam2 no importable: {exc}") from exc
        # from_pretrained acepta ruta local al checkpoint en builds recientes;
        # si falla, el error se propaga como backend no disponible.
        try:
            self._predictor = SAM2VideoPredictor.from_pretrained(SAM2_CHECKPOINT)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo invalida el backend
            raise SamBackendError(f"No se pudo cargar SAM2: {exc}") from exc

    def _load_sam_vith_optflow(self) -> None:
        if not Path(SAM_VITH_CHECKPOINT).exists():
            raise SamBackendError(f"Checkpoint ausente: {SAM_VITH_CHECKPOINT}")
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise SamBackendError(f"segment-anything no importable: {exc}") from exc
        if SAM_MODEL_TYPE not in sam_model_registry:
            raise SamBackendError(f"SAM_MODEL_TYPE inválido: {SAM_MODEL_TYPE!r}")
        model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_VITH_CHECKPOINT)
        device = self.device if torch.cuda.is_available() else "cpu"
        model.to(device=device)
        model.eval()
        self._predictor = SamPredictor(model)

    def _load_grabcut_optflow(self) -> None:
        self._predictor = "grabcut"  # sin pesos; algoritmo clásico de OpenCV

    # -- segmentación ------------------------------------------------------
    def segment_video(
        self, frames: list[np.ndarray], x: float, y: float
    ) -> tuple[list[np.ndarray], SamBackendName, int]:
        if self.backend is None:
            raise SamBackendError("Adapter sin cargar: llama a load() primero.")
        h, w = frames[0].shape[:2]
        x, y = clamp_point(x, y, w, h)
        if self.backend == "sam3":
            return self._run_sam3(frames, x, y)
        if self.backend == "sam2":
            return self._run_sam2(frames, x, y)
        if self.backend == "sam_vith_optflow":
            return self._run_vith_optflow(frames, x, y)
        return self._run_grabcut_optflow(frames, x, y)

    def _run_sam3(
        self, frames: list[np.ndarray], x: float, y: float
    ) -> tuple[list[np.ndarray], SamBackendName, int]:
        """SAM 3.1 del hackatón. Soporta dos formas de predictor (duck-typing).

        Forma A (video nativo): predictor.segment_video(frames, x, y) -> masks.
        Forma B (estilo SAM2): init_state/add_points/propagate sobre dir de frames.
        """
        pred = self._predictor
        h, w = frames[0].shape[:2]
        segment_video = getattr(pred, "segment_video", None)
        if callable(segment_video):
            masks = segment_video(frames, float(x), float(y))
            masks = [np.asarray(m).astype(bool).reshape(h, w) for m in masks]
            if len(masks) != len(frames):
                raise SamBackendError("sam3 devolvió nº de máscaras distinto al de frames.")
            return masks, "sam3", len(frames)
        # Forma B: API estilo video predictor.
        init_state = getattr(pred, "init_state", None)
        add_points = getattr(pred, "add_points", getattr(pred, "add_new_points", None))
        propagate = getattr(pred, "propagate", getattr(pred, "propagate_in_video", None))
        if not (callable(init_state) and callable(add_points) and callable(propagate)):
            raise SamBackendError("Predictor sam3 sin API reconocible (ni segment_video ni init/add/propagate).")
        with tempfile.TemporaryDirectory(prefix="sam3_frames_") as tmp:
            for i, f in enumerate(frames):
                cv2.imwrite(str(Path(tmp) / f"{i:05d}.jpg"), f)
            state = init_state(video_path=tmp)
            add_points(state, frame_idx=0, points=[[float(x), float(y)]], labels=[1])
            masks: list[np.ndarray] = [np.zeros((h, w), dtype=bool) for _ in frames]
            calls = 0
            for out in propagate(state):
                # Acepta tuplas (frame_idx, masks) o dicts {frame_idx: mask}.
                if isinstance(out, dict):
                    items = out.items()
                elif isinstance(out, (tuple, list)) and len(out) >= 2:
                    items = [(int(out[0]), out[1])]
                else:
                    continue
                for idx, m in items:
                    arr = np.asarray(m)
                    if arr.ndim == 3:
                        arr = arr[0]
                    masks[int(idx)] = (arr > 0).reshape(h, w)
                    calls += 1
            return masks, "sam3", calls

    def _run_sam2(
        self, frames: list[np.ndarray], x: float, y: float
    ) -> tuple[list[np.ndarray], SamBackendName, int]:
        import torch

        pred = self._predictor
        h, w = frames[0].shape[:2]
        with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmp:
            for i, f in enumerate(frames):
                cv2.imwrite(str(Path(tmp) / f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with torch.inference_mode():
                state = pred.init_state(video_path=tmp)
                _, _, out_logits = pred.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=np.array([[x, y]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
                masks: list[np.ndarray] = [np.zeros((h, w), dtype=bool) for _ in frames]
                first = torch.as_tensor(out_logits[0]).detach().cpu().numpy()
                if first.ndim == 3:
                    first = first[0]
                masks[0] = (first > 0.0).reshape(h, w)
                calls = 1
                for out_idx, _, out_logits in pred.propagate_in_video(state):
                    arr = torch.as_tensor(out_logits[0]).detach().cpu().numpy()
                    if arr.ndim == 3:
                        arr = arr[0]
                    if 0 <= int(out_idx) < len(masks):
                        masks[int(out_idx)] = (arr > 0.0).reshape(h, w)
                        calls += 1
        return masks, "sam2", calls

    def _track_points_lk(
        self, frames: list[np.ndarray], x: float, y: float
    ) -> list[tuple[float, float]]:
        """Propaga el punto inicial por todo el video con Lucas-Kanade."""
        h, w = frames[0].shape[:2]
        points = [(float(x), float(y))]
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_pts = np.array([[[x, y]]], dtype=np.float32)
        lk_params = dict(
            winSize=(FLOW_WINDOW, FLOW_WINDOW),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        for f in frames[1:]:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params
            )
            if next_pts is not None and int(status[0, 0]) == 1:
                nx, ny = clamp_point(float(next_pts[0, 0, 0]), float(next_pts[0, 0, 1]), w, h)
            else:
                nx, ny = points[-1]  # punto perdido: congela última posición válida
            points.append((nx, ny))
            prev_gray, prev_pts = gray, np.array([[[nx, ny]]], dtype=np.float32)
        return points

    def _run_vith_optflow(
        self, frames: list[np.ndarray], x: float, y: float
    ) -> tuple[list[np.ndarray], SamBackendName, int]:
        import torch

        pred = self._predictor
        h, w = frames[0].shape[:2]
        tracked = self._track_points_lk(frames, x, y)
        # Tope de inferencias: submuestrea y rellena por vecino más cercano.
        stride = max(1, len(frames) // MAX_SAM_CALLS)
        sam_indices = list(range(0, len(frames), stride))
        masks: dict[int, np.ndarray] = {}
        calls = 0
        with torch.inference_mode():
            for i in sam_indices:
                rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                pred.set_image(rgb)
                px, py = tracked[i]
                m, _, _ = pred.predict(
                    point_coords=np.array([[px, py]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
                masks[i] = m[0].astype(bool).reshape(h, w)
                calls += 1
        full = [masks[min(sam_indices, key=lambda s: abs(s - i))] for i in range(len(frames))]
        return full, "sam_vith_optflow", calls

    def _run_grabcut_optflow(
        self, frames: list[np.ndarray], x: float, y: float
    ) -> tuple[list[np.ndarray], SamBackendName, int]:
        """Último recurso sin pesos: GrabCut inicializado en el punto + LK."""
        h, w = frames[0].shape[:2]
        tracked = self._track_points_lk(frames, x, y)
        stride = max(1, len(frames) // MAX_SAM_CALLS)
        sam_indices = list(range(0, len(frames), stride))
        masks: dict[int, np.ndarray] = {}
        box = 80  # semi-lado inicial del rectángulo GrabCut
        for i in sam_indices:
            px, py = tracked[i]
            x1, y1 = int(max(0, px - box)), int(max(0, py - box))
            x2, y2 = int(min(w, px + box)), int(min(h, py + box))
            rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            gc_mask = np.zeros((h, w), dtype=np.uint8)
            bgd = np.zeros((1, 65), dtype=np.float64)
            fgd = np.zeros((1, 65), dtype=np.float64)
            try:
                cv2.grabCut(frames[i], gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            except cv2.error:
                gc_mask[:, :] = cv2.GC_PR_BGD
            masks[i] = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
            # Adapta la ventana al bbox resultante para el siguiente keyframe.
            ys, xs = np.nonzero(masks[i])
            if xs.size > 0:
                box = int(max(24, max(xs.max() - xs.min(), ys.max() - ys.min()) // 2 + 12))
        full = [masks[min(sam_indices, key=lambda s: abs(s - i))] for i in range(len(frames))]
        return full, "grabcut_optflow", 0


# ---------------------------------------------------------------------------
# Función GPU en Modal
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="A10G",
    volumes={JOBS_ROOT.rsplit("/", 1)[0]: volume},
    timeout=900,
    allow_concurrent_inputs=1,
)
class Segmenter:
    @modal.enter()
    def load_model(self) -> None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adapter = SamVideoAdapter(device=device)
        backend = self.adapter.load()
        log.info("Segmenter listo: backend=%s device=%s", backend, device)

    @modal.method()
    def backend_name(self) -> str:
        return str(self.adapter.backend)

    @modal.method()
    def segment_video_bytes(
        self,
        video_bytes: bytes,
        x: float,
        y: float,
        num_keyframes: int = DEFAULT_NUM_KEYFRAMES,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        job = sanitize_job_id(job_id)
        decode_video_bytes(video_bytes)
        num_keyframes = int(min(max(num_keyframes, 4), 64))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            frames, fps = read_video_frames(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        h, w = frames[0].shape[:2]
        # El punto llega en coords del video ORIGINAL; como read_video_frames
        # ya redimensionó, reescalamos el punto con la misma escala. Para ello
        # recuperamos la escala comparando: TARGET_SHORT_SIDE / short_original.
        # Simplificación exacta: el cliente envía coords ya en el espacio
        # decodificado si desconoce el resize; aquí re-escalamos por si el
        # video original era mayor (detectado vía metadata del primer frame
        # sin redimensionar es imposible post-hoc, así que documentamos:
        # x,y se interpretan en el espacio del frame redimensionado si exceden
        # sus dimensiones se sujetan (clamp). Para máxima precisión el cliente
        # puede dividir por scale = TARGET_SHORT_SIDE/min(H0,W0).
        xs, ys = clamp_point(float(x), float(y), w, h)
        # Si el cliente envió coords del original y el video se redujo, las
        # coords excederían (w,h) y el clamp las llevaría al borde: heurística
        # de corrección — si x>=w o y>=h, reescala por w/max(x*1.0, w).
        # (Mantiene compatibilidad con ambos espacios de coordenadas.)
        if float(x) >= w or float(y) >= h:
            s = min(w / max(float(x), 1.0), h / max(float(y), 1.0))
            # Deshace el clamp y aplica la escala estimada del resize.
            xs, ys = clamp_point(float(x) * s * 0.999, float(y) * s * 0.999, w, h)
            log.info("Punto re-escalado del espacio original (s=%.3f)", s)

        masks, backend, sam_calls = self.adapter.segment_video(frames, xs, ys)

        candidates = select_keyframe_indices(len(frames), num_keyframes)
        chosen = refine_by_sharpness(frames, candidates)

        job_dir = Path(JOBS_ROOT) / job
        job_dir.mkdir(parents=True, exist_ok=True)

        keyframes: list[dict[str, Any]] = []
        for rank, idx in enumerate(chosen):
            mask = np.asarray(masks[idx]).astype(bool).reshape(h, w)
            comp = composite_black_bg(frames[idx], mask)
            mask_img = (mask.astype(np.uint8)) * 255
            sharp = frame_sharpness(frames[idx])
            cv2.imwrite(str(job_dir / f"keyframe_{rank:02d}.png"), comp)
            cv2.imwrite(str(job_dir / f"mask_{rank:02d}.png"), mask_img)
            keyframes.append(
                {
                    "index": int(idx),
                    "png_b64": png_to_b64(comp),
                    "mask_b64": png_to_b64(mask_img),
                    "bbox_xyxy": mask_bbox(mask),
                    "sharpness": sharp,
                }
            )

        meta = {
            "sam_backend": backend,
            "fps": float(fps),
            "total_frames": len(frames),
            "width": int(w),
            "height": int(h),
            "scale": float(min(w, h) / TARGET_SHORT_SIDE) if min(w, h) else 1.0,
            "point_original": [float(x), float(y)],
            "point_scaled": [float(xs), float(ys)],
            "sam_calls": int(sam_calls),
        }
        (job_dir / "meta.json").write_text(
            json.dumps({"job_id": job, "meta": meta, "frames": [k["index"] for k in keyframes]}),
            encoding="utf-8",
        )
        volume.commit()

        return {
            "job_id": job,
            "num_keyframes": len(keyframes),
            "width": int(w),
            "height": int(h),
            "keyframes": keyframes,
            "meta": meta,
        }


# ---------------------------------------------------------------------------
# API HTTP (FastAPI servida por Modal)
# ---------------------------------------------------------------------------

web_app = FastAPI(title="VOXEL-SAM AR — Ingesta 360°", version="1.0.0")
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "voxel-sam-ar-ingest",
        "version": "1.0.0",
        "checkpoints": available_backends(),
        "limits": {"max_video_mb": MAX_VIDEO_MB, "max_frames": MAX_FRAMES},
    }


@web_app.get("/ready")
def ready() -> dict[str, Any]:
    """Comprueba que la GPU responde (puede tardar en frío ~60-120 s)."""
    try:
        backend = Segmenter().backend_name.remote()
    except Exception as exc:  # noqa: BLE001 - se reporta como not-ready
        raise HTTPException(status_code=503, detail=f"GPU no lista: {exc}") from exc
    return {"ready": True, "sam_backend": backend}


def _validate_point(x: float, y: float) -> None:
    if not (np.isfinite(x) and np.isfinite(y)) or x < 0 or y < 0:
        raise HTTPException(status_code=422, detail="x e y deben ser píxeles finitos >= 0.")
    if x > 16384 or y > 16384:
        raise HTTPException(status_code=422, detail="x e y exceden el máximo (16384 px).")


@web_app.post("/segment", response_model=SegmentResponse)
async def segment_multipart(
    file: UploadFile = File(description="Video .mp4/.mov 360° del objeto"),
    x: float = Form(description="Coord. x del punto inicial (px)"),
    y: float = Form(description="Coord. y del punto inicial (px)"),
    num_keyframes: int = Form(default=DEFAULT_NUM_KEYFRAMES),
    job_id: str | None = Form(default=None),
) -> JSONResponse:
    _validate_point(x, y)
    if num_keyframes < 4 or num_keyframes > 64:
        raise HTTPException(status_code=422, detail="num_keyframes debe estar en [4, 64].")
    try:
        video_bytes = await file.read()
    finally:
        await file.close()
    try:
        decode_video_bytes(video_bytes)
    except VideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        result = Segmenter().segment_video_bytes.remote(
            video_bytes, float(x), float(y), int(num_keyframes), sanitize_job_id(job_id)
        )
    except VideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(content=result)


@web_app.post("/segment-b64", response_model=SegmentResponse)
def segment_b64(req: SegmentB64Request) -> JSONResponse:
    _validate_point(req.x, req.y)
    try:
        video_bytes = base64.b64decode(req.video_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="video_b64 no es base64 válido.") from exc
    try:
        decode_video_bytes(video_bytes)
    except VideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        result = Segmenter().segment_video_bytes.remote(
            video_bytes, float(req.x), float(req.y), int(req.num_keyframes), sanitize_job_id(req.job_id)
        )
    except VideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(content=result)


@app.function(image=image, volumes={JOBS_ROOT.rsplit("/", 1)[0]: volume}, timeout=60)
def _list_job_files(job_id: str) -> list[str]:
    job_dir = Path(JOBS_ROOT) / sanitize_job_id(job_id)
    if not job_dir.exists():
        return []
    return sorted(p.name for p in job_dir.iterdir() if p.is_file())


@web_app.get("/jobs/{job_id}")
def list_job(job_id: str) -> dict[str, Any]:
    files = _list_job_files.remote(job_id)
    if not files:
        raise HTTPException(status_code=404, detail=f"job_id desconocido: {job_id}")
    return {"job_id": sanitize_job_id(job_id), "files": files}


@app.function(image=image)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    return web_app


# ---------------------------------------------------------------------------
# Entrypoint local: modal run modal/app.py --video ./demo.mp4 --x 320 --y 240
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    video: str,
    x: float,
    y: float,
    num_keyframes: int = DEFAULT_NUM_KEYFRAMES,
    out: str = "./keyframes_out",
) -> None:
    data = Path(video).read_bytes()
    decode_video_bytes(data)
    result = Segmenter().segment_video_bytes.remote(
        data, float(x), float(y), int(num_keyframes), new_job_id("local")
    )
    out_dir = Path(out) / result["job_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, kf in enumerate(result["keyframes"]):
        (out_dir / f"keyframe_{rank:02d}.png").write_bytes(base64.b64decode(kf["png_b64"]))
        (out_dir / f"mask_{rank:02d}.png").write_bytes(base64.b64decode(kf["mask_b64"]))
    (out_dir / "meta.json").write_text(json.dumps(result["meta"], indent=2), encoding="utf-8")
    print(f"job_id={result['job_id']} backend={result['meta']['sam_backend']} -> {out_dir}")
