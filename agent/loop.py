"""VOXEL-SAM AR — Fase 3: bucle agéntico de refinamiento de malla.

Ciclo 0 ejecuta el baseline determinista (`baseline_poisson.py`). Si la malla no
es estanca, el modelo (Meta Model API, OpenAI-compatible) recibe el resumen
geométrico + stderr + MeshReport y devuelve un script candidato; se ejecuta en el
backend (`local` = subprocess, `modal` = Sandbox + Volume), se audita, y se
reitera hasta 10 ciclos. Al éxito exporta `model.glb` (+ `model.usdz` si hay
conversor) y un manifiesto completo del run.

Uso:
  python -m agent.loop --geometry ./geometry.json --exec local
  python -m agent.loop --geometry ./geometry.json --exec modal --max-cycles 10

Env (ciclos 1..N, requieren modelo):
  META_MODEL_API_KEY   (sin clave: solo corre el baseline y reporta)
  META_MODEL_BASE_URL  (endpoint OpenAI-compatible, requerido con clave)
  META_MODEL_NAME      (default muse-spark-1.3-max)

Códigos de salida: 0 éxito (malla estanca) · 3 ciclos agotados/baseline
insuficiente sin clave · 2 entrada inválida · 1 error de infraestructura.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from agent.mesh_tools import MeshReport
    from agent.prompts import MAX_CYCLES, SYSTEM_PROMPT, build_user_prompt, extract_code
except ImportError:  # ejecución con cwd=agent/
    from mesh_tools import MeshReport
    from prompts import MAX_CYCLES, SYSTEM_PROMPT, build_user_prompt, extract_code

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voxel-sam-ar.agent")

AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "muse-spark-1.3-max"
RUNS_VOLUME = "voxel-sam-ar-runs"
STDERR_TAIL = 2000


class LoopError(RuntimeError):
    pass


class ModelConfigError(LoopError):
    pass


class ModelError(LoopError):
    pass


class BackendError(LoopError):
    pass


# ---------------------------------------------------------------------------
# Estructuras
# ---------------------------------------------------------------------------


@dataclass
class ExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


@dataclass
class CycleRecord:
    cycle: int
    kind: str  # "baseline" | "model"
    exit_code: int
    duration_s: float
    report: dict[str, Any] = field(default_factory=dict)
    stderr_tail: str = ""


@dataclass
class RunManifest:
    run_id: str
    backend: str
    status: str = "running"  # success | exhausted | baseline_only_failed | error
    cycles: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    usdz: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Backends de ejecución
# ---------------------------------------------------------------------------


class LocalBackend:
    """Ejecuta candidatos con el Python actual en un directorio de trabajo."""

    name = "local"

    def __init__(self, workdir: Path, exec_timeout: int) -> None:
        self.workdir = workdir
        self.exec_timeout = exec_timeout
        self.workdir.mkdir(parents=True, exist_ok=True)

    def setup(self, files: dict[str, bytes]) -> None:
        for name, data in files.items():
            (self.workdir / name).write_bytes(data)

    def put(self, name: str, data: bytes) -> None:
        (self.workdir / name).write_bytes(data)

    def run_script(self, name: str) -> ExecResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, str(self.workdir / name)],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=self.exec_timeout,
            )
            return ExecResult(proc.returncode, proc.stdout[-8000:], proc.stderr[-8000:],
                              time.monotonic() - start)
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            return ExecResult(124, out[-8000:], f"timeout {self.exec_timeout}s\n{err}"[-8000:],
                              time.monotonic() - start)

    def fetch(self, name: str) -> bytes:
        p = self.workdir / name
        if not p.exists():
            raise BackendError(f"local: falta {name}.")
        return p.read_bytes()

    def has_file(self, name: str) -> bool:
        return (self.workdir / name).exists()

    def close(self) -> None:
        pass


class ModalBackend:
    """Ejecuta candidatos en Modal Sandboxes (CPU) con volumen persistente.

    Un Sandbox solo ve los ficheros subidos ANTES de su creación (montaje por
    snapshot), así que se genera un Sandbox fresco por ciclo: sala limpia +
    todos los uploads visibles. La imagen queda cacheada tras el primer ciclo.
    """

    name = "modal"

    def __init__(self, run_id: str, exec_timeout: int) -> None:
        try:
            import modal
        except ImportError as exc:
            raise BackendError(f"modal no instalado: {exc}") from exc
        self._modal = modal
        self.run_id = run_id
        self.exec_timeout = exec_timeout
        self.remote = f"/work/{run_id}"
        self.image = (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install("open3d", "trimesh[easy]", "scipy", "numpy>=1.26,<2", "usd-core")
        )
        self.volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
        self.app = modal.App.lookup("voxel-sam-ar-agent", create_if_missing=True)
        self.sb: Any = None

    def _spawn(self) -> None:
        self.close()
        log.info("Creando Sandbox Modal (imagen malla + usd-core)…")
        self.sb = self._modal.Sandbox.create(
            app=self.app, image=self.image, volumes={"/work": self.volume}, timeout=3600
        )

    def _exec(self, *cmd: str, timeout: int | None = None) -> ExecResult:
        start = time.monotonic()
        proc = self.sb.exec(*cmd, timeout=timeout or self.exec_timeout)
        out = proc.stdout.read()
        err = proc.stderr.read()
        code = proc.wait()
        return ExecResult(int(code), out[-8000:], err[-8000:], time.monotonic() - start)

    def setup(self, files: dict[str, bytes]) -> None:
        import tempfile

        remotes = [f"{self.run_id}/{name}" for name in files]
        for rp in remotes:  # put_file no sobrescribe: limpieza idempotente
            try:
                self.volume.remove_file(rp)
            except Exception:  # noqa: BLE001 - ausente u otro; put lo dirá
                pass
        with tempfile.TemporaryDirectory(prefix="voxel_agent_") as tmp:
            locals_: list[tuple[str, str]] = []
            for name, data in files.items():
                lp = Path(tmp) / name
                lp.write_bytes(data)
                locals_.append((str(lp), f"{self.run_id}/{name}"))
            with self.volume.batch_upload() as batch:
                for lp, rp in locals_:
                    batch.put_file(lp, rp)

    def put(self, name: str, data: bytes) -> None:
        self.setup({name: data})

    def run_script(self, name: str) -> ExecResult:
        self._spawn()  # el Sandbox debe nacer DESPUÉS de subir el script
        return self._exec("python3", f"{self.remote}/{name}")

    def fetch(self, name: str) -> bytes:
        # Descarga vía base64 sobre stdout (solo usa exec; binario seguro).
        code = (
            "import base64,sys;"
            f"sys.stdout.write(base64.b64encode(open({self.remote + '/' + name!r},'rb').read()).decode())"
        )
        res = self._exec("python3", "-c", code)
        if res.exit_code != 0:
            raise BackendError(f"modal: falta {name}: {res.stderr[-500:]}")
        return base64.b64decode(res.stdout.strip())

    def has_file(self, name: str) -> bool:
        res = self._exec("test", "-f", f"{self.remote}/{name}")
        return res.exit_code == 0

    def close(self) -> None:
        sb, self.sb = self.sb, None
        if sb is None:
            return
        try:
            sb.terminate()
        except Exception:  # noqa: BLE001 - cierre best-effort
            pass


# ---------------------------------------------------------------------------
# Cliente del modelo
# ---------------------------------------------------------------------------


class ModelClient:
    def __init__(self) -> None:
        self.api_key = (os_environ("META_MODEL_API_KEY")).strip()
        self.base_url = (os_environ("META_MODEL_BASE_URL")).strip()
        self.model = (os_environ("META_MODEL_NAME") or DEFAULT_MODEL_NAME).strip()
        if not self.api_key:
            raise ModelConfigError("META_MODEL_API_KEY ausente.")
        if not self.base_url:
            raise ModelConfigError("META_MODEL_BASE_URL ausente.")
        try:
            import openai
        except ImportError as exc:
            raise ModelConfigError(f"openai no instalado: {exc}") from exc
        self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=180.0)

    def draft_script(self, geometry_summary: dict[str, Any], history: list[dict[str, Any]],
                     cycle: int) -> str:
        import time as _t
        import random as _r

        user = build_user_prompt(geometry_summary, history, cycle)
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                    max_tokens=6000,
                )
                return extract_code(resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001 - transporte o parseo
                last = exc
                log.warning("Modelo intento %d falló: %s", attempt, exc)
                _t.sleep(2.0**attempt + _r.uniform(0, 1))
        raise ModelError(f"Modelo falló 3 veces: {last}")


def os_environ(name: str) -> str:
    import os

    return os.environ.get(name, "")


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------


def _geometry_summary_local(geometry_path: Path) -> dict[str, Any]:
    import numpy as np

    geo = json.loads(geometry_path.read_text(encoding="utf-8"))
    verts = np.asarray(geo["vertices"], dtype=np.float64)
    return {
        "source": geo.get("source"),
        "num_vertices": int(len(verts)),
        "has_normals": bool(geo.get("normals")),
        "has_faces": bool(geo.get("faces")),
        "num_faces_in": int(len(geo["faces"])) if geo.get("faces") else 0,
        "confidence": geo.get("confidence"),
        "estimated_height_m": geo.get("estimated_height_m"),
        "bbox_min": [float(c) for c in verts.min(axis=0)],
        "bbox_max": [float(c) for c in verts.max(axis=0)],
    }


def run_loop(
    geometry_path: Path,
    run_dir: Path,
    max_cycles: int = MAX_CYCLES,
    backend_name: str = "local",
    exec_timeout: int = 600,
    run_id: str | None = None,
) -> tuple[RunManifest, int]:
    run_id = run_id or new_run_id()
    max_cycles = int(min(max(max_cycles, 1), MAX_CYCLES))
    run_dir = run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not geometry_path.exists():
        print(json.dumps({"error": f"geometry.json ausente: {geometry_path}"}))
        return RunManifest(run_id, backend_name, status="error"), 2
    try:
        geometry_summary = _geometry_summary_local(geometry_path)
    except Exception as exc:  # noqa: BLE001 - entrada inválida
        print(json.dumps({"error": f"geometry.json inválido: {exc}"}))
        return RunManifest(run_id, backend_name, status="error"), 2

    geometry_bytes = geometry_path.read_bytes()
    mesh_tools_bytes = (AGENT_DIR / "mesh_tools.py").read_bytes()
    baseline_bytes = (AGENT_DIR / "baseline_poisson.py").read_bytes()

    manifest = RunManifest(run_id=run_id, backend=backend_name)
    (run_dir / "geometry.json").write_bytes(geometry_bytes)

    backend: LocalBackend | ModalBackend
    try:
        if backend_name == "modal":
            backend = ModalBackend(run_id, exec_timeout)
        elif backend_name == "local":
            backend = LocalBackend(run_dir / "exec", exec_timeout)
        else:
            raise BackendError(f"backend desconocido: {backend_name!r}.")
        backend.setup({"geometry.json": geometry_bytes, "mesh_tools.py": mesh_tools_bytes})
    except Exception as exc:  # noqa: BLE001 - infra
        manifest.status = "error"
        _write_manifest(run_dir, manifest)
        log.error("Backend falló: %s", exc)
        return manifest, 1

    history: list[dict[str, Any]] = []
    model_client: ModelClient | None = None
    if max_cycles > 1:
        try:
            model_client = ModelClient()
            manifest.model = model_client.model
        except ModelConfigError as exc:
            log.warning("Sin modelo (%s): solo baseline.", exc)

    try:
        for cycle in range(max_cycles):
            tag = f"cycle_{cycle:02d}"
            if cycle == 0:
                code, kind = baseline_bytes.decode("utf-8"), "baseline"
            else:
                assert model_client is not None
                try:
                    code = model_client.draft_script(geometry_summary, history, cycle)
                except (ModelError, ModelConfigError) as exc:
                    manifest.status = "error"
                    history.append({"cycle": cycle, "kind": "model", "error": str(exc)})
                    break
                kind = "model"
            (run_dir / f"{tag}.py").write_text(code, encoding="utf-8")
            backend.put(f"{tag}.py", code.encode("utf-8"))
            res = backend.run_script(f"{tag}.py")
            (run_dir / f"{tag}.log").write_text(
                f"exit={res.exit_code} duration_s={res.duration_s:.1f}\n"
                f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}",
                encoding="utf-8",
            )
            report_dict: dict[str, Any] = {"ok": False, "errors": ["sin report.json"], "stats": {}}
            try:
                report_dict = json.loads(backend.fetch("report.json").decode("utf-8"))
                (run_dir / f"{tag}.report.json").write_text(
                    json.dumps(report_dict, indent=2), encoding="utf-8"
                )
            except BackendError:
                (run_dir / f"{tag}.report.json").write_text(
                    json.dumps(report_dict, indent=2), encoding="utf-8"
                )
            rec = CycleRecord(cycle, kind, res.exit_code, res.duration_s,
                              report_dict, res.stderr[-STDERR_TAIL:])
            history.append(asdict(rec))
            manifest.cycles = history
            _write_manifest(run_dir, manifest)
            ok = bool(report_dict.get("ok")) and res.exit_code == 0
            log.info("ciclo %d (%s): exit=%d ok=%s errores=%s", cycle, kind,
                     res.exit_code, report_dict.get("ok"), report_dict.get("errors"))
            if ok:
                try:
                    glb = backend.fetch("candidate.glb")
                except BackendError as exc:
                    manifest.status = "error"
                    history.append({"cycle": cycle, "error": f"glb ausente: {exc}"})
                    break
                (run_dir / "model.glb").write_bytes(glb)
                manifest.usdz = _try_usdz(backend, run_dir)
                manifest.status = "success"
                _write_manifest(run_dir, manifest)
                return manifest, 0
            if cycle == 0 and model_client is None:
                manifest.status = "baseline_only_failed"
                _write_manifest(run_dir, manifest)
                return manifest, 3
        manifest.status = "exhausted"
        _write_manifest(run_dir, manifest)
        return manifest, 3
    finally:
        backend.close()


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    (run_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")


def _try_usdz(backend: LocalBackend | ModalBackend, run_dir: Path) -> dict[str, Any]:
    if isinstance(backend, LocalBackend):
        sys.path.insert(0, str(AGENT_DIR))
        try:
            from mesh_tools import to_usdz

            conv = to_usdz(run_dir / "model.glb", run_dir / "model.usdz")
            return {"ok": True, "converter": conv, "path": "model.usdz"}
        except Exception as exc:  # noqa: BLE001 - conversión best-effort
            return {"ok": False, "error": str(exc)[-500:]}
        finally:
            sys.path.remove(str(AGENT_DIR))
    assert isinstance(backend, ModalBackend)
    snippet = (
        "import sys;sys.path.insert(0,'/work/" + backend.run_id + "');"
        "from mesh_tools import to_usdz;"
        f"print(to_usdz('/work/{backend.run_id}/candidate.glb','/work/{backend.run_id}/model.usdz'))"
    )
    res = backend._exec("python3", "-c", snippet, timeout=300)
    if res.exit_code != 0 or not backend.has_file("model.usdz"):
        return {"ok": False, "error": res.stderr[-500:] or res.stdout[-500:]}
    try:
        (run_dir / "model.usdz").write_bytes(backend.fetch("model.usdz"))
    except BackendError as exc:
        return {"ok": False, "error": str(exc)[-500:]}
    return {"ok": True, "converter": res.stdout.strip() or "sandbox", "path": "model.usdz"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feedback loop de malla watertight (Fase 3).")
    p.add_argument("--geometry", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=AGENT_DIR / "runs")
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-cycles", type=int, default=MAX_CYCLES)
    p.add_argument("--exec", choices=["local", "modal"], default="local")
    p.add_argument("--exec-timeout", type=int, default=600)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, code = run_loop(
        geometry_path=args.geometry,
        run_dir=args.out_dir,
        max_cycles=args.max_cycles,
        backend_name=args.exec,
        exec_timeout=args.exec_timeout,
        run_id=args.run_id,
    )
    print(manifest.to_json())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
