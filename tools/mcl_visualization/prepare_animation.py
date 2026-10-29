"""Package sealed steady-state results for the explanatory animation; no solves."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ANIMATION_001"


def prepare():
    receipt = json.loads((ROOT / "SOURCE.json").read_text(encoding="utf-8"))
    scene = json.loads((ROOT / "flow_scene.json").read_text(encoding="utf-8"))
    source = Path(receipt["source"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt["source_sha256"]
    with np.load(source, allow_pickle=False) as data:
        mask = data["binary"].copy()
        temperature = data["temperature"].copy()
        qx, qy, qin, qout = [data[k].copy() for k in ("qx", "qy", "qin", "qout")]
        assert str(data["temperature_units"]) == "K above inlet"
    assert np.array_equal(mask, scene["binary"])
    assert mask.shape == (64, 64) and int(mask.sum()) == 1024
    assert temperature.shape == (3, 4, 64, 64) and np.isfinite(temperature).all()
    case = int(temperature[:, 0].max(axis=(1, 2)).argmax())
    field = temperature[case]
    dx, dy = scene["dx_mm"], scene["dy_mm"]
    vx = np.zeros((65, 64)); vy = np.zeros((64, 65))
    vx[1:-1] = qx / (dy * 1e-3 * 1e-3)
    vy[:, 1:-1] = qy / (dx * 1e-3 * 1e-3)
    vx[0, 24:40] = qin / (dy * 1e-3 * 1e-3)
    vx[-1, 24:40] = qout / (dy * 1e-3 * 1e-3)

    def speed(point):
        x = float(np.clip(point[0], .0001, 75.4-.0001))
        y = float(np.clip(point[1], .0001, 72-.0001))
        i, j = int(x/dx), int(y/dy)
        assert mask[i, j], (x, y, i, j)
        a, b = x/dx-i, y/dy-j
        velocity = [(1-a)*vx[i, j]+a*vx[i+1, j],
                    (1-b)*vy[i, j]+b*vy[i, j+1]]
        return float(np.linalg.norm(velocity))

    paths = []
    for raw in scene["paths_mm"]:
        points = np.asarray(raw)
        speeds = np.array([speed(p) for p in points])
        assert np.all(speeds > 1e-10)
        ds = np.linalg.norm(np.diff(points, axis=0), axis=1)*1e-3
        travel = np.r_[0., np.cumsum(2*ds/(speeds[:-1]+speeds[1:]))]
        assert np.all(np.diff(travel) > 0)
        paths.append({"xy_mm": points.tolist(), "travel_s": travel.tolist()})
    # All trajectories use one display factor; no independent path time warping.
    slow_factor = float(np.median([p["travel_s"][-1] for p in paths])/9.)
    assert 0 < slow_factor < 1
    payload = {
        **scene, "paths": paths, "scenario_index": case,
        "temperature_rise_K": field.tolist(), "temperature_scale": [0., 100.],
        "peak_rise_K": float(field[0].max()),
        "display_seconds_to_model_seconds": slow_factor,
        "temperature_is_steady_state": True,
        "heat_arrows_are_explanatory_not_flux_magnitudes": True,
        "layer_separation_is_display_only": True,
        "duration_s": 25, "fps": 24, "resolution": [1920, 1080],
        "additional_surface_heat_loss_in_model": True,
    }
    OUT.mkdir(exist_ok=True)
    target = OUT / "scene_data.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    report = {
        "source_sha256_verified": True, "binary_cells_verified": 1024,
        "full_paths": len(paths), "every_path_point_inside_water": True,
        "scenario_index": case, "steady_peak_rise_K": float(field[0].max()),
        "model_transit_seconds": [p["travel_s"][-1] for p in paths],
        "display_speed_factor": slow_factor, "new_physics_solve": False,
        "unopened_test_workloads_accessed": False,
    }
    with (OUT / "DATA_CHECK.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
