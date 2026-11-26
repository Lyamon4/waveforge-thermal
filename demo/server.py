"""WaveForge demo: local web UI over the real model and the unchanged physics.

Neural design: the frozen solver-free model (no physics inside generation).
Shown fields and numbers: the repository's physics (64x64 fields; peak numbers
from the full 64/128 admission evaluator). MMA600 runs live in the
background exactly as in the benchmark (8 analytic starts, best, 1x600 MMA).
"""

import contextlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402


def _default_backend():
    """cuDSS on an NVIDIA GPU when its wheel is installed, else SciPy on the CPU."""
    if not torch.cuda.is_available():
        return "scipy"
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution(f"nvidia-cudss-cu{torch.version.cuda.split('.')[0]}")
        return "cudss"
    except PackageNotFoundError:
        return "scipy"


DEVICE = os.environ.get("DEMO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault(
    "WAVEFORGE_SPARSE_BACKEND", os.environ.get("DEMO_BACKEND", _default_backend())
)

from waveforge.lab.contracts import Chiplet, ChipletTask  # noqa: E402
from waveforge.lab.epyc_tasks import ANCHORS, IOD, sample_epyc_task  # noqa: E402
from waveforge.lab.mcl.baselines import uniform_parallel  # noqa: E402
from waveforge.lab.mcl.contracts import problem_from_task  # noqa: E402
from waveforge.lab.mcl.dual_profile import solve_dual_profile  # noqa: E402
from waveforge.lab.mcl.locked_channels import canonical_frame  # noqa: E402
from waveforge.lab.mcl.locked_evaluation import ranking  # noqa: E402
from waveforge.lab.mcl.ribbed_evaluation import (  # noqa: E402
    RibbedEvaluator,
    analytic_reference_starts,
    optimize_physical,
)
from waveforge.lab.mcl.solver_free_v2 import SessionV2  # noqa: E402
from waveforge.lab.mcl.stage3_training import worker_init  # noqa: E402
from waveforge.lab.mcl.stage5_evaluation import PUMP_W  # noqa: E402
from waveforge.lab.mcl.stage5_physics import stage5_spec  # noqa: E402

WORK = ROOT / "work"
LOCK = threading.Lock()
JOBS = {}
SESSION = None


def build_task(body):
    """Chiplet layout + three workloads: all-core, the user's load, row-biased."""
    if body.get("layout") == "random":
        seed = int(body.get("seed", 0))
        chips = sample_epyc_task(np.random.default_rng(seed)).chiplets
        ccd = sorted((c for c in chips if c.width == 8), key=lambda c: (c.y >= 32, c.x))
        chips = (*ccd, IOD)
    else:
        chips = (*(Chiplet(int(x), int(y), 8, 7) for x, y in ANCHORS), IOD)
    user = [float(w) for w in body["powers"]]  # 8 CCD + IO, watts
    if len(user) != 9 or min(user) < 0 or max(user) > 400:
        raise ValueError("nine non-negative chiplet powers (W) required")
    row = [58.125] * 4 + [19.375] * 4
    powers = ([38.75] * 8 + [50.0], user, [*row, 50.0])
    return ChipletTask(chips, powers)


def fields(task, design):
    """Unchanged physics on the 64x64 grid, both pump regimes, for display."""
    problem = problem_from_task(task, stage5_spec())
    nominal, power = solve_dual_profile(
        np.asarray(design, dtype=float), problem, pump_power_w=PUMP_W
    )

    def pack(result):
        t = result.temperature  # [scenario, layer, x, y] rise above 25 C inlet
        return dict(
            layers=np.round(
                t + 25, 1
            ).tolist(),  # [scenario][si, tim, base, channel][x][y]
            qx=np.round(result.qx * 1e6, 3).tolist(),  # ml/s through x faces
            qy=np.round(result.qy * 1e6, 3).tolist(),
            flow_ml_s=float(result.flow_ml_s),
            peaks_c=(result.scenario_peaks + 25).round(2).tolist(),
        )

    return dict(
        nominal=pack(nominal),
        power=dict(
            peaks_c=(power.scenario_peaks + 25).round(2).tolist(),
            flow_ml_s=float(power.flow_ml_s),
        ),
    )


def official(task, design_original):
    """Full admission evaluator (64 + 128 grids, both regimes), as in all results."""
    frame = canonical_frame(task)
    ev = RibbedEvaluator(task, WORK / f"eval_{uuid.uuid4().hex[:8]}")
    row = ev.evaluate(frame.to_original(np.asarray(design_original, dtype=bool)))
    return dict(
        valid=bool(row["valid"]),
        nominal_c=row["modes"]["128"]["nominal"]["silicon_peak_c"],
        power_c=row["modes"]["128"]["power"]["silicon_peak_c"],
    )


def package(task, design, seconds, method):
    d = np.asarray(design, dtype=bool)
    return dict(
        method=method,
        seconds=seconds,
        design=d.astype(int).tolist(),
        official=official(task, d),
        fields=fields(task, d),
    )


def neural(body):
    task = build_task(body)
    with LOCK:  # one GPU graph replay at a time
        runs = {}
        for name, infer in (("v2", SESSION.infer), ("v1", SESSION.v1.infer)):
            infer(task)  # warm the exact task shape
            outs = [infer(task) for _ in range(5)]
            runs[name] = (
                outs[-1]["binary"],
                float(np.median([o["total_ms"] for o in outs])),
            )
    chips = [dict(x=c.x, y=c.y, w=c.width, h=c.height) for c in task.chiplets]
    jobs = dict(
        wf=(runs["v2"][0], runs["v2"][1] / 1000, "WaveForge"),
        wf1=(runs["v1"][0], runs["v1"][1] / 1000, "WaveForge v1"),
        straight=(uniform_parallel(), 0.0, "Прямые каналы"),
    )
    with ThreadPoolExecutor(3) as pool:  # physics for display, independent designs
        done = dict(
            zip(jobs, pool.map(lambda j: package(task, *j), jobs.values()), strict=True)
        )
    result = dict(
        chips=chips,
        powers=[list(map(float, p)) for p in task.powers_w],
        **done,
        in_training_range=bool(
            abs(sum(task.powers_w[1][:8]) - 310) < 1
            and max(task.powers_w[1][:8]) <= 80
            and abs(task.powers_w[1][8] - 50) < 1
        ),
    )
    return result


def classical(job, body, kind):
    try:
        worker_init()
        task = build_task(body)
        folder = WORK / f"{kind}_{job}"
        folder.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        ev = RibbedEvaluator(task, folder)
        starts = [(q, h, ev.parameters(q, h)) for q, h in analytic_reference_starts(8)]
        q, h, _ = min(starts, key=lambda x: ranking(x[2]))
        JOBS[job]["phase"] = "optimizing"
        result = optimize_physical(ev, q, h, kind=kind, steps=600, checkpoint_every=50)
        uniform = ev.evaluate(uniform_parallel())
        selected = min((uniform, result["raw"]), key=ranking)
        seconds = time.perf_counter() - started
        design = ev.frame.to_original(ev.designs[selected["binary_sha256"]])
        JOBS[job].update(
            state="done", result=package(task, design, seconds, kind + "600")
        )
    except Exception as exc:  # surfaced to the UI
        traceback.print_exc()
        JOBS[job].update(state="error", error=str(exc))


def progress(job):
    info = JOBS[job]
    path = WORK / f"{info['kind']}_{job}" / "progress.json"
    steps = 0
    if path.exists():
        with contextlib.suppress(json.JSONDecodeError, KeyError):
            steps = json.loads(path.read_text())["objective_evaluations"]
    return dict(
        state=info["state"],
        steps=steps,
        elapsed=time.time() - info["t0"],
        kind=info["kind"],
        error=info.get("error"),
        result=info.get("result") if info["state"] == "done" else None,
    )


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT / "static"), **k)

    def log_message(self, *a):
        pass

    def reply(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        try:
            if self.path == "/api/design":
                return self.reply(200, neural(self.body()))
            if self.path == "/api/mma":
                body = self.body()
                kind = "MMA"
                job = uuid.uuid4().hex[:10]
                JOBS[job] = dict(
                    state="running", phase="starts", kind=kind, t0=time.time()
                )
                threading.Thread(
                    target=classical, args=(job, body, kind), daemon=True
                ).start()
                return self.reply(200, dict(job=job))
            self.reply(404, dict(error="unknown endpoint"))
        except Exception as exc:
            traceback.print_exc()
            self.reply(400, dict(error=str(exc)))

    def do_GET(self):
        if self.path.startswith("/api/job/"):
            job = self.path.rsplit("/", 1)[-1]
            return (
                self.reply(200, progress(job)) if job in JOBS else self.reply(404, {})
            )
        return super().do_GET()


def main():
    global SESSION
    worker_init()
    shutil.rmtree(WORK, ignore_errors=True)  # scratch evaluator folders only
    WORK.mkdir(exist_ok=True)
    SESSION = SessionV2(device=DEVICE)
    warm = build_task(dict(powers=[38.75] * 8 + [50]))
    for _ in range(5):
        SESSION.infer(warm)
        SESSION.v1.infer(warm)
    port = int(os.environ.get("PORT", 8765))
    print(
        f"WaveForge demo on http://localhost:{port}  (network: {DEVICE}, "
        f"physics: {os.environ['WAVEFORGE_SPARSE_BACKEND']})",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
