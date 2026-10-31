"""VOXEL-SAM AR — Phase 3, cycle 0: deterministic baseline.

Uses faces when present; otherwise Poisson with conservative parameters. Repairs,
audits and exports candidate.glb + report.json. Follows the candidate contract.
"""

import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))

import mesh_tools as mt  # noqa: E402

METHOD = "faces+repair"


def main() -> int:
    verts, normals, faces, meta = mt.load_inputs(WORK)
    if faces is not None:
        mesh = mt.build_from_faces(verts, faces)
    else:
        global METHOD
        METHOD = "poisson(depth=8)+repair"
        mesh = mt.triangulate(verts, normals, method="poisson", depth=8)
    mesh = mt.repair(mesh, max_iters=3)
    report = mt.check(mesh)
    mt.save_glb(mesh, WORK / "candidate.glb")
    payload = json.loads(report.to_json())
    payload["stats"]["method"] = METHOD
    payload["stats"]["input"] = {
        "vertices": int(len(verts)),
        "faces_in": int(len(faces)) if faces is not None else 0,
    }
    (WORK / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
