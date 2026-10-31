"""VOXEL-SAM AR — Fase 2: esquemas estrictos de geometría 3D.

`GeometryEstimate` es el contrato entre la inferencia (Baseten `muse-spark-open-1026`
o el fallback determinista por siluetas) y el motor agéntico de la Fase 3.

Convenciones:
  - Unidades: metros (`units="meters"`).
  - Marco: Y-up (`coordinate_frame="y_up"`), objeto centrado en el origen.
  - `vertices[i]` se corresponde con `normals[i]` (normales unitarias).
  - `edges`/`faces` referencian índices de `vertices` (0-based).
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
    """Estimación geométrica 3D de un objeto a partir de keyframes 360°."""

    model_config = {"extra": "forbid"}

    source: Literal["muse-spark-open-1026", "silhouette_fallback"] = "muse-spark-open-1026"
    units: Literal["meters"] = "meters"
    coordinate_frame: Literal["y_up"] = "y_up"

    vertices: list[list[float]] = Field(
        min_length=MIN_VERTICES,
        max_length=MAX_VERTICES,
        description="Matriz de vértices Nx3 en metros, objeto centrado en origen.",
    )
    normals: list[list[float]] = Field(
        min_length=MIN_VERTICES,
        max_length=MAX_VERTICES,
        description="Normal unitaria por vértice (se normaliza en validación).",
    )
    edges: list[list[int]] = Field(
        default_factory=list,
        max_length=MAX_EDGES,
        description="Pares de índices (0-based) como puntos de control de bordes.",
    )
    faces: list[list[int]] | None = Field(
        default=None,
        max_length=MAX_FACES,
        description="Triángulos opcionales como ternas de índices (0-based).",
    )
    source_keyframes: list[int] = Field(
        min_length=1, description="Índices de frame que informaron la estimación."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_height_m: float = Field(
        gt=0.001, lt=50.0, description="Altura real estimada del objeto en metros."
    )
    notes: str = Field(default="", max_length=2000)

    # -- validadores de campo -------------------------------------------------
    @field_validator("vertices", mode="after")
    @classmethod
    def _check_vertices(cls, rows: list[list[float]]) -> list[list[float]]:
        for i, row in enumerate(rows):
            if len(row) != 3:
                raise ValueError(f"vertices[{i}] debe tener 3 coords, tiene {len(row)}.")
            for c in row:
                if not isinstance(c, (int, float)) or not math.isfinite(c):
                    raise ValueError(f"vertices[{i}] contiene un valor no finito: {row}.")
                if abs(c) > COORD_ABS_MAX:
                    raise ValueError(
                        f"vertices[{i}]={row} excede ±{COORD_ABS_MAX} m."
                    )
        return [[float(c) for c in row] for row in rows]

    @field_validator("normals", mode="after")
    @classmethod
    def _check_normals(cls, rows: list[list[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, row in enumerate(rows):
            if len(row) != 3:
                raise ValueError(f"normals[{i}] debe tener 3 comps, tiene {len(row)}.")
            vals = [float(c) for c in row]
            if not all(math.isfinite(c) for c in vals):
                raise ValueError(f"normals[{i}] contiene un valor no finito: {row}.")
            norm = math.sqrt(sum(c * c for c in vals))
            if norm < NORMAL_NORM_MIN:
                raise ValueError(f"normals[{i}] es (casi) vector cero.")
            out.append([c / norm for c in vals])  # normalización determinista
        return out

    @field_validator("edges", mode="after")
    @classmethod
    def _check_edges(cls, rows: list[list[int]]) -> list[list[int]]:
        for i, row in enumerate(rows):
            if len(row) != 2:
                raise ValueError(f"edges[{i}] debe ser un par, es: {row}.")
            a, b = row
            if not isinstance(a, int) or not isinstance(b, int):
                raise ValueError(f"edges[{i}] debe contener enteros: {row}.")
            if a < 0 or b < 0:
                raise ValueError(f"edges[{i}] con índice negativo: {row}.")
            if a == b:
                raise ValueError(f"edges[{i}] degenerada (lazo): {row}.")
        return rows

    @field_validator("faces", mode="after")
    @classmethod
    def _check_faces(cls, rows: list[list[int]] | None) -> list[list[int]] | None:
        if rows is None:
            return None
        for i, row in enumerate(rows):
            if len(row) != 3:
                raise ValueError(f"faces[{i}] debe ser terna, es: {row}.")
            if not all(isinstance(v, int) for v in row):
                raise ValueError(f"faces[{i}] debe contener enteros: {row}.")
            if any(v < 0 for v in row):
                raise ValueError(f"faces[{i}] con índice negativo: {row}.")
            if len(set(row)) != 3:
                raise ValueError(f"faces[{i}] degenerada (vértices repetidos): {row}.")
        return rows

    @field_validator("source_keyframes", mode="after")
    @classmethod
    def _check_keyframes(cls, idx: list[int]) -> list[int]:
        if any((not isinstance(v, int)) or v < 0 for v in idx):
            raise ValueError(f"source_keyframes con valores inválidos: {idx}.")
        return idx

    # -- consistencia cruzada ---------------------------------------------------
    @model_validator(mode="after")
    def _check_consistency(self) -> GeometryEstimate:
        n = len(self.vertices)
        if len(self.normals) != n:
            raise ValueError(
                f"normals tiene {len(self.normals)} filas, vertices tiene {n}."
            )
        for i, (a, b) in enumerate(self.edges):
            if a >= n or b >= n:
                raise ValueError(f"edges[{i}]=[{a},{b}] fuera de rango (N={n}).")
        if self.faces:
            for i, tri in enumerate(self.faces):
                if any(v >= n for v in tri):
                    raise ValueError(f"faces[{i}]={tri} fuera de rango (N={n}).")
        return self

    # -- utilidades -------------------------------------------------------------
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    def to_arrays(self) -> dict[str, Any]:
        """Devuelve vertices/normals/faces como arrays numpy (import perezoso)."""
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
