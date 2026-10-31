"""VOXEL-SAM AR — Fase 3: prompts del agente de malla + extracción de código.

El modelo recibe el resumen geométrico + el historial de ciclos (stderr y
MeshReport) y debe devolver EXCLUSIVAMENTE un script Python ejecutable que use
`mesh_tools`. Sin explicaciones fuera del código.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

try:
    from agent.mesh_tools import CANDIDATE_CONTRACT
except ImportError:  # ejecución con cwd=agent/
    from mesh_tools import CANDIDATE_CONTRACT

MAX_CYCLES = 10
STDERR_TAIL = 2000

SYSTEM_PROMPT = """Eres un ingeniero de geometría 3D autónomo. Escribes scripts Python que convierten
una nube de vértices (`geometry.json`) en una malla estanca (watertight) lista para AR.

{CANDIDATE_CONTRACT}

API disponible (módulo mesh_tools, ya importable):
- load_inputs(workdir) -> (verts, normals|None, faces|None, meta)
- build_from_faces(verts, faces) -> mesh
- triangulate(points, normals=None, method="poisson"|"ball_pivoting"|"alpha", **params) -> mesh
  poisson: depth(4..12), scale, linear_fit, density_quantile(0..0.3)
  ball_pivoting: radii_scale(0.5..8), normal_radius_scale
  alpha: alpha_scale(1..12)
- repair(mesh, max_iters=1..5) -> mesh
- smooth(mesh, method="taubin"|"laplacian", iterations, lambd, mu) -> mesh
- check(mesh) -> MeshReport(ok, errors[], stats{})
- save_glb(mesh, path), summarize_inputs(workdir)

DIRECTORIO DE TRABAJO: Path(__file__).parent (úsalo; prohibidas rutas absolutas).
ÉXITO: report.ok == true con 0 errores.

ESCALERA DE PARÁMETROS (sigue este orden salvo que el historial indique otra cosa):
1. Si hay faces: build_from_faces + repair(max_iters=3) + check.
2. Poisson depth=8, density_quantile=0.05 + repair + check.
3. Poisson depth=10, linear_fit=true + repair(5) + check.
4. Ball pivoting radii_scale=2 + repair + check.
5. Ball pivoting radii_scale=4, normal_radius_scale=6 + repair + check.
6. Alpha alpha_scale=4 + repair + check.
7. Alpha alpha_scale=8 + repair + taubin suave + check.
8. Mejor método hasta ahora + repair(5) + taubin(iterations=20) + check.
9. Último intento: método ganador con parámetros más agresivos + repair(5).

REGLAS DE SALIDA (estrictas):
- Devuelve SOLO código Python ejecutable. Sin markdown, sin explicaciones.
- El script debe terminar escribiendo candidate.glb + report.json e imprimiendo el report.
- Si el error es de sintaxis/import, corrige el código, no los parámetros.
- Si hay huecos: sube repair/max_iters o cambia de método según la escalera.
- Si hay no-manifold/degeneradas: repair + taubin ligero, evita alpha_scale extremo.
""".replace("{CANDIDATE_CONTRACT}", CANDIDATE_CONTRACT)


class PromptError(ValueError):
    pass


def _shorten(text: str, limit: int = STDERR_TAIL) -> str:
    text = text or ""
    return text if len(text) <= limit else "…[recortado]…\n" + text[-limit:]


def build_user_prompt(
    geometry_summary: dict[str, Any],
    history: list[dict[str, Any]],
    cycle: int,
) -> str:
    """Construye el prompt del ciclo con resumen + historial de intentos."""
    lines = [
        f"CICLO {cycle}/{MAX_CYCLES - 1} (el ciclo 0 fue el baseline determinista).",
        "RESUMEN GEOMÉTRICO:",
        json.dumps(geometry_summary, indent=2, sort_keys=True),
        "",
    ]
    if not history:
        lines.append("Sin intentos previos del modelo. Empieza por la escalera (paso 1-2).")
    else:
        lines.append("HISTORIAL DE INTENTOS (más reciente al final):")
        for h in history:
            lines.append(
                f"--- ciclo {h.get('cycle')} exit={h.get('exit_code')} "
                f"ok={h.get('report', {}).get('ok')} "
                f"errores={h.get('report', {}).get('errors', [])} "
                f"stats={json.dumps(h.get('report', {}).get('stats', {}), sort_keys=True)}"
            )
            if h.get("stderr_tail"):
                lines.append(f"stderr:\n{_shorten(str(h['stderr_tail']))}")
    lines += [
        "",
        "Escribe el script Python completo del siguiente intento. SOLO código.",
    ]
    return "\n".join(lines)


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Extrae código Python de la respuesta (tolera fences) y valida sintaxis."""
    if not text or not text.strip():
        raise PromptError("Respuesta vacía del modelo.")
    match = _FENCE_RE.search(text)
    code = match.group(1) if match else text
    code = code.strip()
    if not code:
        raise PromptError("Sin código extraíble en la respuesta.")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise PromptError(f"Código con sintaxis inválida: {exc}") from exc
    return code + "\n"
