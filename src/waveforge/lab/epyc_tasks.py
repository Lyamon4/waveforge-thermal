"""Nine-region procedural tasks, sealed workloads, geometry-only fast exclusions."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .contracts import Chiplet, ChipletTask, _integer, content_hash
from .epyc_protocol import CONFIG_SHA256, load_protocol, validate_protocol
from .journal import _exclusive_json

SPLITS = ("dev", "validation", "test_id", "test_ood")
ANCHORS = np.array([(x, y) for y in (10, 47) for x in (4, 20, 36, 52)])
IOD = Chiplet(26, 26, 12, 12)


def _geometry_slots(task):
    """Unique anchor assignment: even OOD jitter5 cannot reach another slot."""
    coordinates = np.empty((8, 2), dtype=np.int64)
    indices, occupied, iod = [], set(), None
    for index, chip in enumerate(task.chiplets):
        if chip == IOD:
            if iod is not None:
                raise ValueError("duplicate I/O zone")
            iod = index
            continue
        if (chip.width, chip.height) != (8, 7):
            raise ValueError("unexpected CCD shape")
        distance = np.max(np.abs(ANCHORS - (chip.x, chip.y)), axis=1)
        slot = int(distance.argmin())
        if distance[slot] > 5 or slot in occupied:
            raise ValueError("CCD is outside unique registered slots")
        occupied.add(slot)
        coordinates[slot] = chip.x, chip.y
        indices.append((slot, index))
    if len(occupied) != 8 or iod is None or len(task.chiplets) != 9:
        raise ValueError("exactly eight CCD and one I/O zone required")
    for i, a in enumerate(task.chiplets):
        if min(a.x, a.y, 64 - a.x - a.width, 64 - a.y - a.height) < 2:
            raise ValueError("edge margin violated")
        for b in task.chiplets[:i]:
            if not (
                a.x >= b.x + b.width + 2
                or b.x >= a.x + a.width + 2
                or a.y >= b.y + b.height + 2
                or b.y >= a.y + a.height + 2
            ):
                raise ValueError("inter-region gap violated")
    return coordinates, [index for _, index in sorted(indices)], iod


def validate_epyc_task(task: ChipletTask, family: str) -> None:
    if family not in ("id", "ood_geometry", "ood_power"):
        raise ValueError("unknown EPYC task family")
    coordinates, ccd_indices, iod = _geometry_slots(task)
    span = int(np.max(np.abs(coordinates - ANCHORS)))
    if (family == "ood_geometry" and not 2 < span <= 5) or (
        family != "ood_geometry" and span > 2
    ):
        raise ValueError("geometry does not belong to declared family")
    powers = np.asarray(task.powers_w)
    if powers.shape != (3, 9) or not np.allclose(
        powers.sum(axis=1), 360, rtol=0, atol=1e-10
    ):
        raise ValueError("three360W workloads required")
    ccd = powers[:, ccd_indices]
    if not np.array_equal(powers[:, iod], [50] * 3) or not np.all(ccd[0] == 38.75):
        raise ValueError("uniform scenario or I/O power changed")
    expected = np.array([58.125] * 4 + [19.375] * 4)
    if not (np.array_equal(ccd[2], expected) or np.array_equal(ccd[2], expected[::-1])):
        raise ValueError("row-load correspondence changed")
    if family == "ood_power":
        if not np.allclose(
            np.sort(ccd[1]), [110 / 6] * 6 + [100] * 2, rtol=0, atol=1e-12
        ):
            raise ValueError("invalid OOD power pattern")
    elif np.any(ccd[1] <= 0) or np.any(ccd[1] > 80):
        raise ValueError("ID power cap violated")


def sample_epyc_task(rng: np.random.Generator, family: str = "id") -> ChipletTask:
    if family not in ("id", "ood_geometry", "ood_power"):
        raise ValueError("unknown EPYC task family")
    jitter = 5 if family == "ood_geometry" else 2
    for _ in range(10000):
        offsets = rng.integers(-jitter, jitter + 1, (8, 2))
        if family == "ood_geometry" and np.max(np.abs(offsets)) <= 2:
            continue
        xy = ANCHORS + offsets
        if np.any(xy < 2) or np.any(xy + np.array([8, 7]) > 62):
            continue
        chips = (*(Chiplet(int(x), int(y), 8, 7) for x, y in xy), IOD)
        try:
            geometry_task = ChipletTask(chips, ((0.0,) * 9,))
            _geometry_slots(geometry_task)
        except ValueError:
            continue
        break
    else:
        raise RuntimeError("geometry rejection exhausted; distribution unchanged")
    if family == "ood_power":
        random_load = np.full(8, 110 / 6)
        random_load[rng.choice(8, 2, replace=False)] = 100
    else:
        for _ in range(10000):
            random_load = rng.dirichlet(np.full(8, 2.0)) * 310
            if random_load.max() <= 80:
                break
        else:
            raise RuntimeError("power rejection exhausted")
    row_load = [58.125] * 4 + [19.375] * 4
    if rng.integers(0, 2):
        row_load.reverse()
    task = ChipletTask(
        chips,
        (
            tuple([38.75] * 8 + [50]),
            (*random_load, 50),
            (*row_load, 50),
        ),
    )
    canonical = task.canonical()
    task = _decode_task(canonical)
    validate_epyc_task(task, family)
    return task


def _decode_task(row):
    return ChipletTask(tuple(Chiplet(**c) for c in row["chiplets"]), row["powers_w"])


def demo_geometry_hash() -> str:
    chips = (*(Chiplet(int(x), int(y), 8, 7) for x, y in ANCHORS), IOD)
    return ChipletTask(chips, ((0.0,) * 9,)).geometry_fingerprint()


class GeometryIndex:
    """Vectorized equivalent of bipartite matching for these separated slots."""

    def __init__(self, tasks: tuple[ChipletTask, ...]):
        self.coordinates = np.array(
            [_geometry_slots(t)[0] for t in tasks], dtype=np.int64
        ).reshape(-1, 8, 2)

    def contains(self, task: ChipletTask) -> bool:
        xy = _geometry_slots(task)[0]
        return bool(np.any(np.max(np.abs(self.coordinates - xy), axis=(1, 2)) <= 2))


def create_registry(path: Path, protocol: dict) -> dict:
    protocol = validate_protocol(protocol)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    chosen, payloads, geometries = [], {}, {}
    split_spec = protocol["splits"]
    for name in SPLITS:
        rng = np.random.default_rng(split_spec[f"{name}_seed"])
        rows = []
        for i in range(split_spec[f"{name}_count"]):
            family = (
                ("ood_geometry" if i < 16 else "ood_power")
                if name == "test_ood"
                else "id"
            )
            index = GeometryIndex(tuple(chosen))
            for _ in range(10000):
                task = sample_epyc_task(rng, family)
                if (
                    task.geometry_fingerprint() != demo_geometry_hash()
                    and not index.contains(task)
                ):
                    chosen.append(task)
                    rows.append(task.canonical())
                    break
            else:
                raise RuntimeError("registered near-duplicate rejection exhausted")
        payloads[name] = json.dumps(
            rows, sort_keys=True, indent=2, allow_nan=False
        ).encode()
        geometries[name] = [row["chiplets"] for row in rows]
    geometry_payload = {"schema": "epyc-geometry-index-v1", "splits": geometries}
    manifest = {
        "schema": "epyc-native-splits-v1",
        "protocol_sha256": CONFIG_SHA256,
        "counts": {n: split_spec[f"{n}_count"] for n in SPLITS},
        "seeds": {n: split_spec[f"{n}_seed"] for n in SPLITS},
        "sha256": {n: hashlib.sha256(data).hexdigest() for n, data in payloads.items()},
        "geometry_content_sha256": content_hash(geometry_payload),
        "demo_exact_geometry_sha256": demo_geometry_hash(),
    }
    path.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (path / f"{name}.json").open("xb") as handle:
            handle.write(data)
    _exclusive_json(path / "geometry_exclusions.json", geometry_payload)
    _exclusive_json(path / "manifest.json", manifest)
    return manifest


def validate_registry(path: Path) -> dict:
    path = Path(path)
    data = (path / "manifest.json").read_bytes()
    manifest = json.loads(data)
    spec = load_protocol()["splits"]
    if (
        manifest.get("schema") != "epyc-native-splits-v1"
        or manifest.get("protocol_sha256") != CONFIG_SHA256
        or manifest.get("counts") != {n: spec[f"{n}_count"] for n in SPLITS}
        or manifest.get("seeds") != {n: spec[f"{n}_seed"] for n in SPLITS}
        or manifest.get("demo_exact_geometry_sha256") != demo_geometry_hash()
        or set(manifest.get("sha256", {})) != set(SPLITS)
    ):
        raise ValueError("registry contract mismatch")
    for n in SPLITS:
        # Integrity only: NEVER deserialize test powers here.
        if (
            hashlib.sha256((path / f"{n}.json").read_bytes()).hexdigest()
            != manifest["sha256"][n]
        ):
            raise ValueError("registry workload hash mismatch")
    geometry = json.loads((path / "geometry_exclusions.json").read_bytes())
    if content_hash(geometry) != manifest["geometry_content_sha256"]:
        raise ValueError("geometry index hash mismatch")
    if geometry.get("schema") != "epyc-geometry-index-v1" or set(
        geometry["splits"]
    ) != set(SPLITS):
        raise ValueError("geometry index membership mismatch")
    for n in SPLITS:
        if len(geometry["splits"][n]) != manifest["counts"][n]:
            raise ValueError("geometry index count mismatch")
    return dict(manifest, registry_sha256=hashlib.sha256(data).hexdigest())


def load_epyc_split(
    path: Path, name: str, *, allow_test: bool = False
) -> list[ChipletTask]:
    if name not in SPLITS:
        raise ValueError("unknown split")
    if name.startswith("test_") and allow_test is not True:
        raise PermissionError("test workloads remain sealed until frozen evaluation")
    manifest = validate_registry(path)
    rows = json.loads((Path(path) / f"{name}.json").read_bytes())
    if len(rows) != manifest["counts"][name]:
        raise ValueError("registered count mismatch")
    tasks = [_decode_task(row) for row in rows]
    for i, task in enumerate(tasks):
        validate_epyc_task(
            task,
            ("ood_geometry" if i < 16 else "ood_power") if name == "test_ood" else "id",
        )
    return tasks


class EpycTaskStream:
    def __init__(self, seed: int, registry: Path):
        _integer(seed, "seed", minimum=0)
        registry = Path(registry)
        manifest = validate_registry(registry)
        geometry = json.loads((registry / "geometry_exclusions.json").read_bytes())
        if content_hash(geometry) != manifest["geometry_content_sha256"]:
            raise ValueError("geometry index hash mismatch")
        if geometry.get("schema") != "epyc-geometry-index-v1" or set(
            geometry["splits"]
        ) != set(SPLITS):
            raise ValueError("geometry index membership mismatch")
        tasks = []
        for name in SPLITS:
            rows = geometry["splits"][name]
            if len(rows) != manifest["counts"][name]:
                raise ValueError("geometry membership count mismatch")
            tasks.extend(
                ChipletTask(tuple(Chiplet(**c) for c in row), ((0.0,) * 9,))
                for row in rows
            )
        self.exclusions = GeometryIndex(tuple(tasks))
        self.exclusion_sha256 = content_hash(
            {
                "registry": manifest["registry_sha256"],
                "geometry": manifest["geometry_content_sha256"],
            }
        )
        self.protocol_sha256 = CONFIG_SHA256
        self.seed, self.rng = seed, np.random.default_rng(seed)
        self.exposures, self.rejected = 0, 0
        self.accepted_task_hashes = []

    def next_batch(self) -> tuple[ChipletTask, ...]:
        tasks = []
        for _ in range(4):
            for _attempt in range(10000):
                task = sample_epyc_task(self.rng)
                if (
                    task.geometry_fingerprint() != demo_geometry_hash()
                    and not self.exclusions.contains(task)
                ):
                    tasks.append(task)
                    break
                self.rejected += 1
            else:
                raise RuntimeError("training holdout rejection exhausted")
        self.exposures += 4
        self.accepted_task_hashes.extend(t.fingerprint() for t in tasks)
        return tuple(tasks)

    def state_dict(self) -> dict:
        return copy.deepcopy(
            {
                "seed": self.seed,
                "exclusion_sha256": self.exclusion_sha256,
                "protocol_sha256": self.protocol_sha256,
                "exposures": self.exposures,
                "rejected": self.rejected,
                "accepted_task_hashes": self.accepted_task_hashes,
                "rng_state": self.rng.bit_generator.state,
            }
        )

    def load_state_dict(self, state: dict) -> None:
        if any(
            state[k] != getattr(self, k)
            for k in ("seed", "protocol_sha256", "exclusion_sha256")
        ):
            raise ValueError("resume identity mismatch")
        for key in ("exposures", "rejected"):
            _integer(state[key], key, minimum=0)
        hashes = state["accepted_task_hashes"]
        if (
            state["exposures"] % 4
            or len(hashes) != state["exposures"]
            or any(
                not isinstance(h, str) or re.fullmatch("[0-9a-f]{64}", h) is None
                for h in hashes
            )
        ):
            raise ValueError("incomplete or malformed exposure history")
        rng = np.random.default_rng(self.seed)
        rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self.rng, self.exposures, self.rejected = (
            rng,
            state["exposures"],
            state["rejected"],
        )
        self.accepted_task_hashes = list(hashes)
