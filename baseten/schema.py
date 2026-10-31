"""VOXEL-SAM AR — Phase 2: strict 3D geometry schemas.

`GeometryEstimate` is the contract between inference (Baseten `muse-spark-open-1026`
or the deterministic silhouette fallback) and the Phase 3 agentic engine.

Conventions:
  - Units: meters (`units="meters"`).
  - Frame: Y-up (`coordinate_frame="y_up"`), object centered at the origin.
  - `vertices[i]` corresponds to `normals[i]` (unit normals).
  - `edges`/`faces` reference `vertices` indices (0-based).
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MIN_VERTICES = 8
MAX_VERTICES = 1024
MAX_EDGES = 4096
MAX_FACES = 8192
COORD_ABS_MAX = 50.0
NORMAL_NORM_MIN = 1e-9


class GeometryEstimate(BaseModel):
    """3D geometry estimate of an object from 360° keyframes."""

    model_config = {"extra": "forbid"}

    source: Literal["muse-spark-open-1026", "silhouette_fallback"] = "muse-spark-open-1026"
    units: Literal["meters"] = "meters"
    coordinate_frame: Literal["y_up"] = "y_up"

    vertices: list[list[float]] = Field(
        min_length=MIN_VERTICES,
        max_length=MAX_VERTICES,
        description="Nx3 vertex matrix in meters, object centered at origin.",
    )
    normals: list[list[float]] = Field(
        min_length=MIN_VERTICES,
        max_length=MAX_VERTICES,
        description="Unit normal per vertex (normalized during validation).",
    )
    edges: list[list[int]] = Field(
        default_factory=list,
        max_length=MAX_EDGES,
        description="Index pairs (0-based) as edge control points.",
    )
    faces: list[list[int]] | None = Field(
        default=None,
        max_length=MAX_FACES,
        description="Optional triangles as index triples (0-based).",
    )
    source_keyframes: list[int] = Field(
        min_length=1, description="Frame indices that informed the estimate."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_height_m: float = Field(
        gt=0.001, lt=50.0, description="Estimated real-world object height in meters."
    )
    notes: str = Field(default="", max_length=2000)

    # -- field validators ---------------------------------------------------
    @field_validator("vertices", mode="after")
    @classmethod
    def _check_vertices(cls, rows: list[list[float]]) -> list[list[float]]:
        for i, row in enumerate(rows):
            if len(row) != 3:
                raise ValueError(f"vertices[{i}] must have 3 coords, has {len(row)}.")
            for c in row:
                if not isinstance(c, (int, float)) or not math.isfinite(c):
                    raise ValueError(f"vertices[{i}] contains a non-finite value: {row}.")
                if abs(c) > COORD_ABS_MAX:
                    raise ValueError(
                        f"vertices[{i}]={row} exceeds ±{COORD_ABS_MAX} m."
                    )
        return [[float(c) for c in row] for row in rows]

    @field_validator("normals", mode="after")
    @classmethod
    def _check_normals(cls, rows: list[list[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, row in enumerate(rows):
            if len(row) != 3:
                raise ValueError(f"normals[{i}] must have 3 comps, has {len(row)}.")
            vals = [float(c) for c in row]
            if not all(math.isfinite(c) for c in vals):
                raise ValueError(f"normals[{i}] contains a non-finite value: {row}.")
            norm = math.sqrt(sum(c * c for c in vals))
            if norm < NORMAL_NORM_MIN:
                raise ValueError(f"normals[{i}] is a (near-)zero vector.")
            out.append([c / norm for c in vals])  # deterministic normalization
        return out

    @field_validator("edges", mode="after")
    @classmethod
    def _check_edges(cls, rows: list[list[int]]) -> list[list[int]]:
        for i, row in enumerate(rows):
            if len(row) != 2:
                raise ValueError(f"edges[{i}] must be a pair, is: {row}.")
            a, b = row
            if not isinstance(a, int) or not isinstance(b, int):
                raise ValueError(f"edges[{i}] must contain integers: {row}.")
            if a < 0 or b < 0:
                raise ValueError(f"edges[{i}] with negative index: {row}.")
            if a == b:
                raise ValueError(f"edges[{i}] degenerate (self-loop): {row}.")
        return rows

    @field_validator("faces", mode="after")
    @classmethod
    def _check_faces(cls, rows: list[list[int]] | None) -> list[list[int]] | None:
        if rows is None:
            return None
        for i, row in enumerate(rows):
            if len(row) != 3:
                raise ValueError(f"faces[{i}] must be a triple, is: {row}.")
            if not all(isinstance(v, int) for v in row):
                raise ValueError(f"faces[{i}] must contain integers: {row}.")
            if any(v < 0 for v in row):
                raise ValueError(f"faces[{i}] with negative index: {row}.")
            if len(set(row)) != 3:
                raise ValueError(f"faces[{i}] degenerate (repeated vertices): {row}.")
        return rows

    @field_validator("source_keyframes", mode="after")
    @classmethod
    def _check_keyframes(cls, idx: list[int]) -> list[int]:
        if any((not isinstance(v, int)) or v < 0 for v in idx):
            raise ValueError(f"source_keyframes with invalid values: {idx}.")
        return idx

    # -- cross-field consistency ----------------------------------------------
    @model_validator(mode="after")
    def _check_consistency(self) -> GeometryEstimate:
        n = len(self.vertices)
        if len(self.normals) != n:
            raise ValueError(
                f"normals has {len(self.normals)} rows, vertices has {n}."
            )
        for i, (a, b) in enumerate(self.edges):
            if a >= n or b >= n:
                raise ValueError(f"edges[{i}]=[{a},{b}] out of range (N={n}).")
        if self.faces:
            for i, tri in enumerate(self.faces):
                if any(v >= n for v in tri):
                    raise ValueError(f"faces[{i}]={tri} out of range (N={n}).")
        return self

    # -- utilities ------------------------------------------------------------
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    def to_arrays(self) -> dict[str, Any]:
        """Returns vertices/normals/faces as numpy arrays (lazy import)."""
        import numpy as np

        out: dict[str, Any] = {
            "vertices": np.asarray(self.vertices, dtype=np.float64),
            "normals": np.asarray(self.normals, dtype=np.float64),
        }
        if self.faces:
            out["faces"] = np.asarray(self.faces, dtype=np.int64)
        if self.edges:
            out["edges"] = np.asarray(self.edges, dtype=np.int64)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "vertices": len(self.vertices),
            "edges": len(self.edges),
            "faces": len(self.faces) if self.faces else 0,
            "keyframes": len(self.source_keyframes),
            "confidence": self.confidence,
            "estimated_height_m": self.estimated_height_m,
        }
