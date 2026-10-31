"""VOXEL-SAM AR — Phase 2: segmented keyframes -> `geometry.json`.

Remote path (primary): keyframe batch to the hackathon model
`muse-spark-open-1026` on Baseten via an OpenAI-compatible endpoint, with
Instructor in strict JSON mode (`response_model=GeometryEstimate`) + retries.

Local path (deterministic fallback): silhouette-based 3D lifting under a
uniform circular orbit assumption (360° video): each scanline of each mask
contributes two edge vertices projected at the matching azimuth. No weights,
numpy+PIL only. Used when Baseten is not configured or fails.

Usage:
  python -m baseten.infer --keyframes-dir ./kf --meta ./kf/meta.json
  python -m baseten.infer --keyframes-dir ./kf --no-remote --height-m 0.25

Env (remote path):
  BASETEN_API_KEY    (required for remote)
  BASETEN_BASE_URL   (default https://inference.baseten.co/v1)
  BASETEN_MODEL_NAME (default muse-spark-open-1026)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from baseten.schema import GeometryEstimate
except ImportError:  # direct script run: python baseten/infer.py
    from schema import GeometryEstimate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voxel-sam-ar.baseten")

DEFAULT_BASE_URL = "https://inference.baseten.co/v1"
DEFAULT_MODEL_NAME = "muse-spark-open-1026"
DEFAULT_HEIGHT_M = 0.20
MASK_THRESHOLD = 24  # min RGB sum to count as object pixel
TRANSPORT_RETRIES = 3
INSTRUCTOR_RETRIES = 2

SYSTEM_PROMPT = """You are a 3D geometry estimator. You receive 360° KEYFRAMES of the same
object segmented on a black background, ordered by increasing azimuth (full turn).

Return EXCLUSIVELY a JSON with this shape (no markdown, no extra text):
{
  "vertices": [[x,y,z], ...],        // 32..512 points, meters, object CENTERED at origin
  "normals": [[nx,ny,nz], ...],      // one per vertex (self-normalized)
  "edges": [[i,j], ...],             // 0-based index pairs (edges/ridges)
  "faces": [[a,b,c], ...] | null,    // optional triangles, 0-based indices
  "source_keyframes": [int, ...],    // frame indices used
  "confidence": 0.0..1.0,
  "estimated_height_m": float,       // estimated real object height in meters
  "notes": "brief string"
}
Rules: Y-up frame; |coord| <= 50; vertices spread over the WHOLE surface visible
in the views (not just one face); `normals` with the same length as `vertices`;
valid indices; `estimated_height_m` consistent with an everyday object.
The client sets the "source" field; do not include it.
"""


class BasetenConfigError(RuntimeError):
    pass


class BasetenTransportError(RuntimeError):
    pass


class GeometryValidationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Keyframe loading
# ---------------------------------------------------------------------------


def load_keyframes(
    keyframes_dir: Path, meta_path: Path | None = None
) -> tuple[list[np.ndarray], list[int]]:
    pngs = sorted(keyframes_dir.glob("keyframe_*.png"))
    if not pngs:
        raise GeometryValidationError(f"No keyframe_*.png in {keyframes_dir}.")
    frames: list[np.ndarray] = []
    for p in pngs:
        with Image.open(p) as img:
            frames.append(np.asarray(img.convert("RGB")))
    indices = list(range(len(pngs)))
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frames_idx = meta.get("frames") or meta.get("meta", {}).get("frames")
            if isinstance(frames_idx, list) and len(frames_idx) == len(pngs):
                indices = [int(v) for v in frames_idx]
        except (ValueError, AttributeError) as exc:
            log.warning("meta.json unreadable (%s); using 0..N-1 indices.", exc)
    log.info("Keyframes loaded: %d (%s).", len(frames), keyframes_dir)
    return frames, indices


def _pil_to_data_uri(img: Image.Image, max_side: int, jpeg_q: int) -> str:
    rgb = img.convert("RGB")
    rgb.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=jpeg_q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Remote path: Baseten + Instructor (strict JSON)
# ---------------------------------------------------------------------------


def estimate_remote(
    frames: list[np.ndarray],
    frame_indices: list[int],
    max_side: int = 768,
    jpeg_q: int = 72,
) -> GeometryEstimate:
    try:
        import instructor
        import openai
    except ImportError as exc:
        raise BasetenConfigError(f"Missing remote dependency: {exc}") from exc

    api_key = os.environ.get("BASETEN_API_KEY", "").strip()
    if not api_key:
        raise BasetenConfigError("BASETEN_API_KEY missing.")
    base_url = os.environ.get("BASETEN_BASE_URL", DEFAULT_BASE_URL).strip()
    model = os.environ.get("BASETEN_MODEL_NAME", DEFAULT_MODEL_NAME).strip()
    log.info("Remote: model=%s base_url=%s (%d images).", model, base_url, len(frames))

    raw = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)
    client = instructor.from_openai(raw, mode=instructor.Mode.JSON)

    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Estimate the 3D geometry of the object from these {len(frames)} "
                f"360° keyframes (uniform azimuth, full turn). Frame "
                f"indices: {frame_indices}. Return only the JSON."
            ),
        }
    ]
    for arr in frames:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _pil_to_data_uri(Image.fromarray(arr), max_side, jpeg_q)},
            }
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": parts},
    ]

    last_err: Exception | None = None
    for attempt in range(1, TRANSPORT_RETRIES + 1):
        try:
            estimate = client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=GeometryEstimate,
                max_retries=INSTRUCTOR_RETRIES,
            )
            estimate.source = "muse-spark-open-1026"
            return estimate
        except Exception as exc:  # noqa: BLE001 - classified below
            last_err = exc
            name = type(exc).__name__
            retryable = any(
                k in name
                for k in (
                    "Connection",
                    "Timeout",
                    "RateLimit",
                    "InternalServer",
                    "ServiceUnavailable",
                    "BadGateway",
                    "APIError",
                )
            ) or "validation" not in name.lower()
            # Instructor validation errors already retried inside;
            # only transport ones deserve another outer cycle.
            if "validation" in name.lower() or "Validation" in name:
                raise GeometryValidationError(
                    f"Model returned invalid JSON after retries: {exc}"
                ) from exc
            if not retryable or attempt == TRANSPORT_RETRIES:
                break
            backoff = 2.0**attempt + random.uniform(0, 1)
            log.warning("Remote attempt %d failed (%s); retrying in %.1fs.", attempt, exc, backoff)
            time.sleep(backoff)
    raise BasetenTransportError(f"Baseten failed after {TRANSPORT_RETRIES} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Local path: silhouette lifting (deterministic, numpy)
# ---------------------------------------------------------------------------


def _longest_run(mask_row: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(mask_row)
    if idx.size == 0:
        return None
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[splits + 1]))
    ends = np.concatenate((idx[splits], [idx[-1]]))
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def estimate_from_silhouettes(
    frames: list[np.ndarray],
    frame_indices: list[int],
    height_m: float = DEFAULT_HEIGHT_M,
) -> GeometryEstimate:
    """Rebuilds a scanline visual-hull tube under a uniform orbit.

    Camera on a circular orbit of radius R looking at the origin; image
    "right" vector at azimuth θ: right(θ) = (-sinθ, 0, cosθ). Each silhouette
    edge (du pixels from center) lifts to P = du*s*right(θ) + (0, y, 0).
    """
    if not frames:
        raise GeometryValidationError("No frames to estimate from.")
    if not (0.001 < height_m < 50.0):
        raise GeometryValidationError(f"height_m out of range: {height_m}.")

    h, w = frames[0].shape[:2]
    cx_img = w / 2.0
    masks = [(arr.sum(axis=2) > MASK_THRESHOLD) for arr in frames]
    bboxes: list[tuple[int, int] | None] = []
    for m in masks:
        rows = np.flatnonzero(m.any(axis=1))
        bboxes.append((int(rows[0]), int(rows[-1])) if rows.size else None)
    valid = [i for i, b in enumerate(bboxes) if b is not None]
    if not valid:
        raise GeometryValidationError("All masks are empty.")

    heights = [bboxes[i][1] - bboxes[i][0] + 1 for i in valid]  # type: ignore[index]
    horizon = float(np.median([(bboxes[i][0] + bboxes[i][1]) / 2 for i in valid]))  # type: ignore[index]
    s = height_m / float(np.median(heights))  # meters per pixel

    n_total = len(frames)
    n_valid = len(valid)
    n_scan = max(2, min(24, 1022 // max(1, 2 * n_valid)))

    verts: list[list[float]] = []
    norms: list[list[float]] = []
    grid: dict[tuple[int, int, int], int] = {}  # (rank_valid, slot, side) -> idx

    for rank, i in enumerate(valid):
        theta = 2.0 * np.pi * i / n_total  # azimuth by position in the sequence
        right = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        y0, y1 = bboxes[i]  # type: ignore[misc]
        rows = np.linspace(y0, y1, n_scan).round().astype(int)
        for slot, r in enumerate(rows):
            run = _longest_run(masks[i][r])
            if run is None:
                # Empty scanline: reuse the nearest valid neighboring row.
                run = next(
                    (_longest_run(masks[i][rr]) for rr in rows if _longest_run(masks[i][rr])),
                    None,
                )
                if run is None:
                    continue
            for side, u in enumerate(run):
                du = (u - cx_img) * s
                y = (r - horizon) * s
                p = du * right + np.array([0.0, y, 0.0])
                n = np.sign(du) * right if abs(du) > 1e-12 else np.array([0.0, 1.0, 0.0])
                grid[(rank, slot, side)] = len(verts)
                verts.append([float(p[0]), float(p[1]), float(p[2])])
                norms.append([float(n[0]), float(n[1]), float(n[2])])

    edges: list[list[int]] = []
    for (rank, slot, side), vi in grid.items():
        down = grid.get((rank, slot + 1, side))
        if down is not None:
            edges.append([vi, down])
        nxt = grid.get(((rank + 1) % n_valid, slot, side))
        if nxt is not None and nxt != vi:
            edges.append([vi, nxt])

    faces: list[list[int]] = []
    for rank in range(n_valid):
        rn = (rank + 1) % n_valid
        for slot in range(n_scan - 1):
            for side in (0, 1):
                a = grid.get((rank, slot, side))
                b = grid.get((rn, slot, side))
                c = grid.get((rn, slot + 1, side))
                d = grid.get((rank, slot + 1, side))
                if None in (a, b, c, d):
                    continue
                assert a is not None and b is not None and c is not None and d is not None
                faces.append([a, b, c])
                faces.append([a, c, d])

    # Top/bottom fan caps over the end rings.
    ys = [v[1] for v in verts]
    top_idx, bot_idx = len(verts), len(verts) + 1
    verts += [[0.0, max(ys), 0.0], [0.0, min(ys), 0.0]]
    norms += [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    for rank in range(n_valid):
        rn = (rank + 1) % n_valid
        for slot, apex, flip in ((n_scan - 1, top_idx, False), (0, bot_idx, True)):
            for side in (0, 1):
                a = grid.get((rank, slot, side))
                b = grid.get((rn, slot, side))
                if a is None or b is None:
                    continue
                faces.append([apex, b, a] if flip else [apex, a, b])

    used_frames = [frame_indices[i] for i in valid]
    return GeometryEstimate(
        source="silhouette_fallback",
        vertices=verts,
        normals=norms,
        edges=edges[:4096],
        faces=faces[:8192],
        source_keyframes=used_frames,
        confidence=0.35,
        estimated_height_m=height_m,
        notes=(
            f"Silhouette fallback: uniform 360° orbit, {n_valid}/{n_total} valid "
            f"views, {n_scan} scanlines, scale {s:.5f} m/px from height {height_m} m."
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="360° keyframes -> geometry.json (Phase 2).")
    p.add_argument("--keyframes-dir", required=True, type=Path)
    p.add_argument("--meta", type=Path, default=None, help="meta.json with real indices")
    p.add_argument("--out", type=Path, default=None, help="default: <keyframes-dir>/geometry.json")
    p.add_argument("--height-m", type=float, default=DEFAULT_HEIGHT_M)
    p.add_argument("--no-remote", action="store_true", help="local fallback only")
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--jpeg-q", type=int, default=72)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frames, indices = load_keyframes(args.keyframes_dir, args.meta)
    except GeometryValidationError as exc:
        log.error("Invalid input: %s", exc)
        return 2

    estimate: GeometryEstimate | None = None
    if not args.no_remote:
        try:
            estimate = estimate_remote(frames, indices, args.max_side, args.jpeg_q)
            log.info("Remote geometry OK: %s", estimate.summary())
        except BasetenConfigError as exc:
            log.warning("Remote not configured (%s); using local fallback.", exc)
        except (BasetenTransportError, GeometryValidationError) as exc:
            log.error("Remote failed (%s); using local fallback.", exc)

    if estimate is None:
        try:
            estimate = estimate_from_silhouettes(frames, indices, args.height_m)
            log.info("Local geometry OK: %s", estimate.summary())
        except GeometryValidationError as exc:
            log.error("Local fallback failed: %s", exc)
            return 1

    out = args.out or (args.keyframes_dir / "geometry.json")
    out.write_text(estimate.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), **estimate.summary()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
