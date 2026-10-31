"""VOXEL-SAM AR — Phase 3: mesh-agent prompts + code extraction.

The model receives the geometry summary + the cycle history (stderr and
MeshReport) and must return EXCLUSIVELY an executable Python script using
`mesh_tools`. No explanations outside the code.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

try:
    from agent.mesh_tools import CANDIDATE_CONTRACT
except ImportError:  # run with cwd=agent/
    from mesh_tools import CANDIDATE_CONTRACT

MAX_CYCLES = 10
STDERR_TAIL = 2000

SYSTEM_PROMPT = """You are an autonomous 3D geometry engineer. You write Python scripts that turn
a vertex cloud (`geometry.json`) into a watertight mesh ready for AR.

{CANDIDATE_CONTRACT}

Available API (mesh_tools module, already importable):
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

WORKING DIRECTORY: Path(__file__).parent (use it; absolute paths forbidden).
SUCCESS: report.ok == true with 0 errors.

PARAMETER LADDER (follow this order unless history says otherwise):
1. If faces exist: build_from_faces + repair(max_iters=3) + check.
2. Poisson depth=8, density_quantile=0.05 + repair + check.
3. Poisson depth=10, linear_fit=true + repair(5) + check.
4. Ball pivoting radii_scale=2 + repair + check.
5. Ball pivoting radii_scale=4, normal_radius_scale=6 + repair + check.
6. Alpha alpha_scale=4 + repair + check.
7. Alpha alpha_scale=8 + repair + gentle taubin + check.
8. Best method so far + repair(5) + taubin(iterations=20) + check.
9. Last attempt: winning method with more aggressive parameters + repair(5).

OUTPUT RULES (strict):
- Return ONLY executable Python code. No markdown, no explanations.
- The script must finish by writing candidate.glb + report.json and printing the report.
- On syntax/import errors, fix the code, not the parameters.
- On holes: raise repair/max_iters or switch methods per the ladder.
- On non-manifold/degenerate: repair + light taubin, avoid extreme alpha_scale.
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
    """Builds the cycle prompt with summary + attempt history."""
    lines = [
        f"CYCLE {cycle}/{MAX_CYCLES - 1} (cycle 0 was the deterministic baseline).",
        "GEOMETRY SUMMARY:",
        json.dumps(geometry_summary, indent=2, sort_keys=True),
        "",
    ]
    if not history:
        lines.append("No prior model attempts. Start with the ladder (steps 1-2).")
    else:
        lines.append("ATTEMPT HISTORY (most recent last):")
        for h in history:
            lines.append(
                f"--- cycle {h.get('cycle')} exit={h.get('exit_code')} "
                f"ok={h.get('report', {}).get('ok')} "
                f"errors={h.get('report', {}).get('errors', [])} "
                f"stats={json.dumps(h.get('report', {}).get('stats', {}), sort_keys=True)}"
            )
            if h.get("stderr_tail"):
                lines.append(f"stderr:\n{_shorten(str(h['stderr_tail']))}")
    lines += [
        "",
        "Write the complete Python script for the next attempt. CODE ONLY.",
    ]
    return "\n".join(lines)


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Extracts Python code from the reply (tolerates fences) and checks syntax."""
    if not text or not text.strip():
        raise PromptError("Empty model reply.")
    match = _FENCE_RE.search(text)
    code = match.group(1) if match else text
    code = code.strip()
    if not code:
        raise PromptError("No extractable code in the reply.")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise PromptError(f"Code with invalid syntax: {exc}") from exc
    return code + "\n"
