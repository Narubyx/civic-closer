"""VOXEL-SAM AR — Phase 3: deterministic mesh toolbox.

API used by the candidate scripts the agent writes (and by the baseline):
triangulate [`triangulate`], repair [`repair`], smooth [`smooth`], audit
[`check`] and export [`save_glb`, `to_usdz`].

No heavy module-level imports (all lazy): `MeshReport` and the contract can be
imported in any lightweight environment; mesh-operating functions import
numpy/trimesh/open3d/scipy at call time.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

CANDIDATE_CONTRACT = """CANDIDATE SCRIPT CONTRACT (mandatory):
- Inputs colocated with the script: `geometry.json`, `mesh_tools.py`.
- The script must: 1) load the geometry, 2) build/repair/smooth the mesh
  ONLY with mesh_tools functions, 3) write `candidate.glb` and `report.json`
  (serialized MeshReport) to that same directory, 4) print the report JSON.
- Forbidden: network, threads, absolute paths outside the working directory."""


# ---------------------------------------------------------------------------
# Topology report
# ---------------------------------------------------------------------------


@dataclass
class MeshReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MeshReport:
        return cls(
            ok=bool(d.get("ok", False)),
            errors=list(d.get("errors", [])),
            stats=dict(d.get("stats", {})),
        )

    @classmethod
    def from_json(cls, s: str) -> MeshReport:
        return cls.from_dict(json.loads(s))


def _failure(reason: str, **stats: Any) -> MeshReport:
    return MeshReport(ok=False, errors=[reason], stats=dict(stats))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_inputs(workdir: str | Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Returns (vertices, normals|None, faces|None, meta) as numpy arrays."""
    import numpy as np

    workdir = Path(workdir)
    geo = json.loads((workdir / "geometry.json").read_text(encoding="utf-8"))
    verts = np.asarray(geo["vertices"], dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 4:
        raise ValueError(f"invalid vertices: shape {verts.shape}.")
    normals = None
    if geo.get("normals"):
        normals = np.asarray(geo["normals"], dtype=np.float64)
        if normals.shape != verts.shape:
            raise ValueError(f"normals {normals.shape} vs vertices {verts.shape}.")
    faces = None
    if geo.get("faces"):
        faces = np.asarray(geo["faces"], dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"invalid faces: shape {faces.shape}.")
        if faces.min() < 0 or faces.max() >= len(verts):
            raise ValueError("faces with out-of-range indices.")
    meta = {k: geo.get(k) for k in ("source", "confidence", "estimated_height_m", "units")}
    return verts, normals, faces, meta


def estimate_normals_pca(points: Any, k: int = 12) -> Any:
    """k-neighbor PCA normals (numpy only; +Y oriented when ambiguous)."""
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    k = int(min(max(k, 3), n - 1))
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    nn = np.argpartition(d2, kth=k, axis=1)[:, 1 : k + 1]
    normals = np.zeros_like(pts)
    for i in range(n):
        cov = np.cov((pts[nn[i]] - pts[i]).T)
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, int(np.argmin(vals))]
        if v[1] < 0:
            v = -v
        normals[i] = v
    return normals


# ---------------------------------------------------------------------------
# Triangulation (requires open3d)
# ---------------------------------------------------------------------------


def _avg_spacing(points: Any) -> float:
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    sample = pts if len(pts) <= 2000 else pts[np.random.RandomState(0).choice(len(pts), 2000)]
    d2 = ((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min(axis=1)).mean())


def triangulate(
    points: Any,
    normals: Any | None = None,
    method: Literal["poisson", "ball_pivoting", "alpha"] = "poisson",
    **params: Any,
) -> Any:
    """Reconstructs a surface from a point cloud. Returns trimesh.Trimesh."""
    import numpy as np
    import open3d as o3d
    import trimesh

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4:
        raise ValueError("Need >= 4 points to triangulate.")
    spacing = _avg_spacing(pts)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=spacing * float(params.get("normal_radius_scale", 4.0)), max_nn=30
            )
        )
    try:
        pcd.orient_normals_consistent_tangent_plane(int(params.get("orient_knn", 30)))
    except RuntimeError:
        pass

    if method == "poisson":
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=int(params.get("depth", 9)),
            scale=float(params.get("scale", 1.1)),
            linear_fit=bool(params.get("linear_fit", False)),
        )
        dens = np.asarray(densities)
        keep = dens > np.quantile(dens, float(params.get("density_quantile", 0.05)))
        mesh_o3d.remove_vertices_by_mask(~keep)
    elif method == "ball_pivoting":
        scale = float(params.get("radii_scale", 2.0))
        radii = o3d.utility.DoubleVector([spacing * scale, spacing * scale * 2])
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    elif method == "alpha":
        alpha = spacing * float(params.get("alpha_scale", 4.0))
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    else:
        raise ValueError(f"Unknown method: {method!r}.")

    v = np.asarray(mesh_o3d.vertices)
    f = np.asarray(mesh_o3d.triangles)
    if len(v) == 0 or len(f) == 0:
        raise ValueError(f"Triangulation {method} produced an empty mesh.")
    return trimesh.Trimesh(v, f, process=True)


# ---------------------------------------------------------------------------
# Build / repair / smooth
# ---------------------------------------------------------------------------


def build_from_faces(vertices: Any, faces: Any) -> Any:
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=True)
    if len(mesh.faces) == 0:
        raise ValueError("build_from_faces produced an empty mesh.")
    return mesh


def repair(mesh: Any, max_iters: int = 3) -> Any:
    """Cleanup + hole filling + winding. Returns the repaired mesh."""
    import numpy as np
    import trimesh

    for _ in range(max(1, int(max_iters))):
        mesh.remove_unreferenced_vertices()
        # Degenerate faces (area ~0) via explicit mask.
        try:
            areas = np.asarray(mesh.area_faces, dtype=np.float64)
        except Exception:  # noqa: BLE001
            areas = None
        if areas is not None and 0 < int((areas > 1e-12).sum()) < len(mesh.faces):
            mesh.update_faces(areas > 1e-12)
        # Duplicate faces (same vertex triple, any order).
        key = np.sort(np.asarray(mesh.faces), axis=1)
        _, idx = np.unique(key, axis=0, return_index=True)
        if 0 < len(idx) < len(mesh.faces):
            keep = np.zeros(len(mesh.faces), dtype=bool)
            keep[np.sort(idx)] = True
            mesh.update_faces(keep)
        mesh.merge_vertices()
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda p: p.area)
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:  # noqa: BLE001 - fill is best-effort per iteration
            pass
        try:
            trimesh.repair.fix_winding(mesh)
        except Exception:  # noqa: BLE001 - fix_winding is best-effort
            pass
        if bool(mesh.is_watertight) and float(mesh.volume) < 0:
            mesh.invert()
        if bool(mesh.is_watertight) and bool(mesh.is_winding_consistent):
            break
    return mesh


def smooth(
    mesh: Any,
    method: Literal["taubin", "laplacian"] = "taubin",
    iterations: int = 10,
    lambd: float = 0.5,
    mu: float = -0.53,
) -> Any:
    import numpy as np
    import open3d as o3d
    import trimesh

    m = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    if method == "taubin":
        m = m.filter_smooth_taubin(
            number_of_iterations=int(iterations), lambda_filter=float(lambd), mu=float(mu)
        )
    elif method == "laplacian":
        m = m.filter_smooth_simple(number_of_iterations=int(iterations), lambda_filter=float(lambd))
    else:
        raise ValueError(f"Unknown smoothing: {method!r}.")
    return trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=True)


# ---------------------------------------------------------------------------
# Topology audit
# ---------------------------------------------------------------------------


def check(mesh: Any) -> MeshReport:
    import numpy as np

    try:
        nv, nf = int(len(mesh.vertices)), int(len(mesh.faces))
    except TypeError:
        return _failure("empty or invalid mesh")
    if nv == 0 or nf == 0:
        return _failure("empty mesh (0 vertices or 0 faces)")

    edges = np.sort(np.asarray(mesh.edges), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    n_boundary = int((counts == 1).sum())
    n_nonmanifold = int((counts > 2).sum())
    n_degenerate = int((np.asarray(mesh.area_faces) < 1e-12).sum())
    watertight = bool(mesh.is_watertight)
    winding = bool(mesh.is_winding_consistent)
    try:
        euler = int(mesh.euler_number)
    except Exception:  # noqa: BLE001
        euler = -999
    try:
        volume = float(mesh.volume)
    except Exception:  # noqa: BLE001
        volume = float("nan")
    stats = {
        "vertices": nv,
        "faces": nf,
        "watertight": watertight,
        "winding_consistent": winding,
        "euler_number": euler,
        "boundary_edges": n_boundary,
        "nonmanifold_edges": n_nonmanifold,
        "degenerate_faces": n_degenerate,
        "volume": volume,
        "area": float(mesh.area),
        "bbox_min": [float(c) for c in np.asarray(mesh.bounds)[0]],
        "bbox_max": [float(c) for c in np.asarray(mesh.bounds)[1]],
    }
    errors: list[str] = []
    if not watertight:
        errors.append(f"mesh holes: {n_boundary} boundary edges")
    if not winding:
        errors.append("inconsistent normals (non-uniform winding)")
    if n_nonmanifold:
        errors.append(f"non-manifold geometry: {n_nonmanifold} edges")
    if n_degenerate:
        errors.append(f"degenerate faces: {n_degenerate}")
    if watertight and euler != -999 and euler % 2 != 0:
        errors.append(f"invalid euler for closed mesh: {euler}")
    return MeshReport(ok=not errors, errors=errors, stats=stats)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def save_glb(mesh: Any, path: str | Path) -> Path:
    path = Path(path)
    path.write_bytes(bytes(mesh.export(file_type="glb")))
    return path


def _fmt_vec(v: Any) -> str:
    return "(" + ", ".join(f"{float(c):.7g}" for c in v) + ")"


def _write_usda(mesh: Any, usda_path: Path) -> tuple[int, int]:
    """Authors a static USDA (subset: 1 Xform + 1 Mesh). Returns (N, F)."""
    import numpy as np

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if verts.size == 0 or faces.size == 0:
        raise ValueError("Empty mesh: no USDZ.")
    if not (np.isfinite(verts).all() and np.isfinite(faces).all()
            and np.isfinite(normals).all()):
        raise ValueError("Mesh with non-finite values: no USDZ.")
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    counts = ", ".join(["3"] * len(faces))
    idx = ", ".join(str(int(v)) for v in faces.ravel())
    pt_lines = [
        ", ".join(_fmt_vec(p) for p in verts[i : i + 8])
        for i in range(0, len(verts), 8)
    ]
    n_lines = [
        ", ".join(_fmt_vec(p) for p in normals[i : i + 8])
        for i in range(0, len(normals), 8)
    ]
    usda = (
        "#usda 1.0\n(\n\tmetersPerUnit = 1\n\tupAxis = \"Y\"\n)\n\n"
        "def Xform \"Model\"\n{\n"
        "    def Mesh \"Mesh\"\n    {\n"
        f"        float3[] extent = [{_fmt_vec(lo)}, {_fmt_vec(hi)}]\n"
        f"        int[] faceVertexCounts = [{counts}]\n"
        f"        int[] faceVertexIndices = [{idx}]\n"
        "        normal3f[] primvars:normals = [\n"
        + "".join(f"            {line},\n" for line in n_lines)
        + "        ] (\n            elementSize = 1\n"
        "            interpolation = \"vertex\"\n        )\n"
        "        point3f[] points = [\n"
        + "".join(f"            {line},\n" for line in pt_lines)
        + "        ]\n"
        "        uniform token subdivisionScheme = \"none\"\n"
        "        bool doubleSided = true\n"
        "    }\n}\n"
    )
    usda_path.write_text(usda, encoding="utf-8")
    return int(len(verts)), int(len(faces))


def _validate_usdz(usdz_path: Path, expect_points: int) -> None:
    """Structural check + re-open with usd-core when available."""
    import zipfile

    with zipfile.ZipFile(usdz_path) as zf:
        names = zf.namelist()
        if names != ["model.usda"]:
            raise ValueError(f"Unexpected USDZ: {names}.")
        if zf.getinfo("model.usda").compress_type != zipfile.ZIP_STORED:
            raise ValueError("USDZ must use ZIP_STORED (no compression).")
        head = zf.read("model.usda")[:64]
        if not head.startswith(b"#usda 1.0"):
            raise ValueError("USDZ content without USDA header.")
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return  # no usd-core: structural check only
    stage = Usd.Stage.Open(str(usdz_path))
    prim = stage.GetPrimAtPath("/Model/Mesh")
    if not prim.IsValid():
        raise ValueError("USDZ without /Model/Mesh prim.")
    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    if pts is None or len(pts) != expect_points:
        raise ValueError(f"USDZ with {len(pts) if pts else 0} points, expected {expect_points}.")


def to_usdz(glb_path: str | Path, usdz_path: str | Path, timeout: int = 300) -> str:
    """Converts .glb to .usdz (authored USDA + STORED zip). Returns converter.

    No system dependencies: works the same locally and in the Sandbox.
    """
    import tempfile
    import trimesh
    import zipfile

    _ = timeout  # reserved for future external converters
    mesh = trimesh.load(str(glb_path), force="mesh")
    with tempfile.TemporaryDirectory(prefix="voxel_usd_") as tmp:
        usda = Path(tmp) / "model.usda"
        n_points, _ = _write_usda(mesh, usda)
        with zipfile.ZipFile(usdz_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(usda, arcname="model.usda")
    _validate_usdz(Path(usdz_path), n_points)
    return "usda+zip"


def summarize_inputs(workdir: str | Path) -> dict[str, Any]:
    import numpy as np

    verts, normals, faces, meta = load_inputs(workdir)
    return {
        **meta,
        "num_vertices": int(len(verts)),
        "has_normals": normals is not None,
        "has_faces": faces is not None,
        "num_faces_in": int(len(faces)) if faces is not None else 0,
        "bbox_min": [float(c) for c in verts.min(axis=0)],
        "bbox_max": [float(c) for c in verts.max(axis=0)],
    }


def main(argv: list[str] | None = None) -> int:
    """Minimal CLI: `python mesh_tools.py check <mesh.glb|obj|ply|stl>`."""
    import trimesh

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "check":
        print("usage: python mesh_tools.py check <mesh.glb|obj|ply|stl|off>")
        return 2
    mesh = trimesh.load(args[1], force="mesh")
    print(check(mesh).to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
