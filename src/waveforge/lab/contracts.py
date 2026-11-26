"""Immutable physical inputs. Geometry uses the explicit base-64 design lattice."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path


def _positive(value: float, name: str, *, zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} is outside its physical range")


def _integer(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def content_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Material:
    k: float
    rho: float
    cp: float

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            _positive(value, key)
            object.__setattr__(self, key, float(value))


@dataclass(frozen=True)
class StackSpec:
    width_m: float = 0.0754
    height_m: float = 0.072
    silicon_m: float = 0.0005
    tim_m: float = 0.0001
    spreader_m: float = 0.002
    h_w_m2k: float = 5000.0
    ambient_c: float = 25.0
    contact_m2k_w: float = 2.0e-6
    copper_fraction: float = 0.25
    silicon: Material = Material(130.0, 2330.0, 700.0)
    tim: Material = Material(5.0, 2500.0, 1200.0)
    copper: Material = Material(400.0, 8960.0, 385.0)
    aluminum: Material = Material(200.0, 2700.0, 900.0)

    def __post_init__(self) -> None:
        for key in (
            "width_m",
            "height_m",
            "silicon_m",
            "tim_m",
            "spreader_m",
            "h_w_m2k",
        ):
            _positive(getattr(self, key), key)
        _positive(self.contact_m2k_w, "contact resistance", zero=True)
        if (
            isinstance(self.ambient_c, bool)
            or not isinstance(self.ambient_c, Real)
            or not math.isfinite(self.ambient_c)
            or self.ambient_c <= -273.15
        ):
            raise ValueError("ambient must be above absolute zero")
        if self.copper_fraction != 0.25:
            raise ValueError("virtual-lab-v1 requires copper fraction 0.25")
        for key in ("silicon", "tim", "copper", "aluminum"):
            if not isinstance(getattr(self, key), Material):
                raise ValueError(f"{key} must be a Material")
        for key, value in asdict(self).items():
            if isinstance(value, Real):
                object.__setattr__(self, key, 0.0 if value == 0 else float(value))

    def fingerprint(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True, order=True)
class Chiplet:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for key in ("x", "y", "width", "height"):
            _integer(getattr(self, key), key, minimum=0 if key in ("x", "y") else 1)
        if self.x + self.width > 64 or self.y + self.height > 64:
            raise ValueError("chiplet exceeds base-64 domain")


@dataclass(frozen=True)
class ChipletTask:
    chiplets: tuple[Chiplet, ...]
    powers_w: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "chiplets", tuple(self.chiplets))
        object.__setattr__(self, "powers_w", tuple(tuple(row) for row in self.powers_w))
        if not self.chiplets or not self.powers_w:
            raise ValueError("task requires chiplets and workloads")
        for index, first in enumerate(self.chiplets):
            if not isinstance(first, Chiplet):
                raise ValueError("chiplets must contain Chiplet instances")
            for second in self.chiplets[:index]:
                if (
                    first.x < second.x + second.width
                    and second.x < first.x + first.width
                    and first.y < second.y + second.height
                    and second.y < first.y + first.height
                ):
                    raise ValueError("chiplet footprints overlap")
        for row in self.powers_w:
            if len(row) != len(self.chiplets):
                raise ValueError("one power per chiplet is required")
            for value in row:
                _positive(value, "power", zero=True)
        object.__setattr__(
            self,
            "powers_w",
            tuple(
                tuple(0.0 if value == 0 else float(value) for value in row)
                for row in self.powers_w
            ),
        )

    def canonical(self) -> dict:
        order = sorted(range(len(self.chiplets)), key=self.chiplets.__getitem__)
        return {
            "chiplets": [asdict(self.chiplets[i]) for i in order],
            "powers_w": [[float(row[i]) for i in order] for row in self.powers_w],
        }

    def fingerprint(self) -> str:
        return content_hash(self.canonical())

    def geometry_fingerprint(self) -> str:
        return content_hash(self.canonical()["chiplets"])


def save_stack(spec: StackSpec, path: Path) -> None:
    """Exclusive create: never overwrite an existing protocol accidentally."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(asdict(spec), handle, indent=2, allow_nan=False)


def load_stack(path: Path) -> StackSpec:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    for name in ("silicon", "tim", "copper", "aluminum"):
        if name in payload:
            payload[name] = Material(**payload[name])
    return StackSpec(**payload)
