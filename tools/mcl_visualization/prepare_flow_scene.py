"""Extract exact fluid geometry and trace saved depth-averaged fluxes for display."""
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
receipt = json.loads((OUT / "SOURCE.json").read_text(encoding="utf-8"))
source = Path(receipt["source"])
if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["source_sha256"]:
    raise RuntimeError("Source changed")
with np.load(source, allow_pickle=False) as archive:
    d = archive["binary"].copy()
    qx, qy, qin, qout = (archive[k].copy() for k in ("qx", "qy", "qin", "qout"))
dx, dy, height = 75.4 / 64, 72 / 64, 1e-3
vx = np.zeros((65, 64)); vy = np.zeros((64, 65))
vx[1:-1] = qx / (dy * 1e-3 * height)
vy[:, 1:-1] = qy / (dx * 1e-3 * height)
vx[0, 24:40] = qin / (dy * 1e-3 * height)
vx[-1, 24:40] = qout / (dy * 1e-3 * height)


def velocity(point):
    x, y = point
    if not (0 <= x < 75.4 and 0 <= y < 72):
        return None
    i, j = int(x / dx), int(y / dy)
    if not d[i, j]:
        return None
    a, b = x / dx - i, y / dy - j
    return np.array([(1-a)*vx[i,j]+a*vx[i+1,j],
                     (1-b)*vy[i,j]+b*vy[i,j+1]])


paths = []
for j in range(24, 40):
    if not d[0, j] or vx[0, j] <= 0:
        continue
    p = np.array([0.015, (j + 0.5) * dy])
    line = [p.copy()]
    for _ in range(3000):
        v = velocity(p)
        if v is None or np.linalg.norm(v) < 1e-12:
            break
        direction = v / np.linalg.norm(v)
        mid = velocity(p + .07 * direction)
        if mid is not None and np.linalg.norm(mid) > 1e-12:
            direction = mid / np.linalg.norm(mid)
        nxt = p + .14 * direction
        if nxt[0] >= 75.4:
            end = p + (75.4-p[0]) / direction[0] * direction
            line.append(end)
            break
        if velocity(nxt) is None:
            break
        line.append(nxt.copy()); p = nxt
    path = np.asarray(line)
    if path[-1, 0] >= 75.39:
        paths.append(path)
if len(paths) < 4:
    raise RuntimeError(f"Only {len(paths)} full inlet-outlet streamlines; inspect field")
selected = [paths[i] for i in np.unique(np.linspace(0, len(paths)-1, 8).astype(int))]
for path in selected:
    for p in path[1:-1]:
        if velocity(p) is None:
            raise RuntimeError("Displayed streamline leaves water cells")
selection = Path("artifacts/inputs/development_tasks.json")
if hashlib.sha256(selection.read_bytes()).hexdigest() != receipt["selection_sha256"]:
    raise RuntimeError("Selection changed")
task = json.loads(selection.read_text())["tasks"][receipt["task_index"]]
scene = {
    "binary": d.astype(int).tolist(), "chiplets": task["chiplets"],
    "paths_mm": [p.tolist() for p in selected], "dx_mm": dx, "dy_mm": dy,
    "actual_layer_z_mm": [0, .5, .6, 1.6, 2.6],
    "flow_ml_s": float(qin.sum()*1e6), "source": str(source),
    "source_sha256": receipt["source_sha256"],
    "streamline_method": "piecewise face-flux interpolation, normalized RK2, 0.14mm step; stop outside binary water",
    "flow_is_depth_averaged": True, "new_3d_CFD_solve": False,
    "streamline_draw_height_mm": 2.66,
    "external_hoses_or_pump_in_model": False,
}
with (OUT / "flow_scene.json").open("x", encoding="utf-8") as handle:
    json.dump(scene, handle)
print(json.dumps({"valid_paths":len(paths),"display_paths":len(selected),
                  "water_cells":int(d.sum()),"flow_ml_s":scene["flow_ml_s"]}))
