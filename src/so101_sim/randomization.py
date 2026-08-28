"""Semantic randomization engine.

Draws a fresh episode specification -- which shapes exist, what colour they
are, where they sit, and which (object, zone) pair the instruction refers to --
from a single integer seed, so any episode is exactly reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .constants import (
    CUBE_HALF,
    MIN_ENTITY_SEPARATION,
    OBJECT_COLORS,
    OBJECT_SHAPES,
    WORKSPACE_R_MAX,
    WORKSPACE_R_MIN,
    WORKSPACE_YAW,
    ZONE_COLORS,
    ZONE_RADIUS,
    ZONE_SHAPES,
)


@dataclass
class Entity:
    name: str
    color: str
    shape: str
    rgba: tuple[float, float, float, float]
    pos: tuple[float, float]
    yaw: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass
class EpisodeSpec:
    seed: int
    objects: list[Entity]
    zones: list[Entity]
    target_object_idx: int
    target_zone_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_object(self) -> Entity:
        return self.objects[self.target_object_idx]

    @property
    def target_zone(self) -> Entity:
        return self.zones[self.target_zone_idx]

    @property
    def instruction(self) -> str:
        return f"take the {self.target_object.label} and place it in the {self.target_zone.label}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["instruction"] = self.instruction
        return d


_MAX_LAYOUT_RESTARTS = 50


class _PlacementFailure(RuntimeError):
    """Raised internally when a greedy layout pass cannot place every entity."""


def _sample_positions(rng: np.random.Generator, count: int, existing: list[tuple[float, float]]):
    """Rejection-sample xy positions in the reachable annulus, spaced apart."""
    placed: list[tuple[float, float]] = []
    for _ in range(count):
        for _attempt in range(500):
            radius = rng.uniform(WORKSPACE_R_MIN, WORKSPACE_R_MAX)
            yaw = rng.uniform(-WORKSPACE_YAW, WORKSPACE_YAW)
            candidate = (float(radius * np.cos(yaw)), float(radius * np.sin(yaw)))
            others = existing + placed
            if all(
                np.hypot(candidate[0] - o[0], candidate[1] - o[1]) >= MIN_ENTITY_SEPARATION
                for o in others
            ):
                placed.append(candidate)
                break
        else:
            raise _PlacementFailure
    return placed


def sample_episode(seed: int) -> EpisodeSpec:
    """Build a fully specified, reproducible episode from `seed`."""
    rng = np.random.default_rng(seed)

    n_objects = int(rng.integers(2, 4))   # 2 or 3
    n_zones = int(rng.integers(2, 4))     # 2 or 3

    obj_colors = [str(c) for c in rng.choice(list(OBJECT_COLORS), size=n_objects, replace=False)]
    zone_colors = [str(c) for c in rng.choice(list(ZONE_COLORS), size=n_zones, replace=False)]

    # Zones are placed first: they are larger footprints and harder to fit.
    # The annulus is tight enough that a single greedy pass occasionally paints
    # itself into a corner, so restart the whole layout rather than dropping the
    # seed. Draws continue on the same stream, so this stays reproducible.
    for _restart in range(_MAX_LAYOUT_RESTARTS):
        try:
            zone_xy = _sample_positions(rng, n_zones, [])
            obj_xy = _sample_positions(rng, n_objects, zone_xy)
            break
        except _PlacementFailure:
            continue
    else:
        raise RuntimeError(
            f"seed {seed}: could not lay out {n_objects} objects and {n_zones} zones "
            f"in the workspace annulus after {_MAX_LAYOUT_RESTARTS} restarts"
        )

    zones = [
        Entity(
            name=f"zone_{i}",
            color=c,
            shape=str(rng.choice(ZONE_SHAPES)),
            rgba=ZONE_COLORS[c],
            pos=zone_xy[i],
        )
        for i, c in enumerate(zone_colors)
    ]
    objects = [
        Entity(
            name=f"object_{i}",
            color=c,
            shape=str(rng.choice(OBJECT_SHAPES)),
            rgba=OBJECT_COLORS[c],
            pos=obj_xy[i],
            yaw=float(rng.uniform(-np.pi / 4, np.pi / 4)),
        )
        for i, c in enumerate(obj_colors)
    ]

    return EpisodeSpec(
        seed=seed,
        objects=objects,
        zones=zones,
        target_object_idx=int(rng.integers(0, n_objects)),
        target_zone_idx=int(rng.integers(0, n_zones)),
        metadata={
            "cube_half": CUBE_HALF,
            "zone_radius": ZONE_RADIUS,
            "n_objects": n_objects,
            "n_zones": n_zones,
        },
    )
