"""Line-of-sight sensing of other vehicles.

A vehicle sees a neighbour when it is within ``max_range`` and no obstacle
body lies on the straight segment between them. On a convex obstacle that
makes neighbours drop out of sight when the curve turns far enough for the
body to come between the two vehicles.

Vehicles are points, and a neighbour never hides another one. This is pure
geometry: it reads positions and obstacles, never time or control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from circumnav.models.obstacles import Obstacle


@dataclass(frozen=True)
class NeighbourSensor:
    """Range-limited, occludable line of sight between two vehicles."""

    max_range: float
    check_occlusion: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_range) or self.max_range <= 0.0:
            raise ValueError("max_range must be positive")

    def in_range(
        self, observer: NDArray[np.float64], target: NDArray[np.float64]
    ) -> bool:
        return float(np.linalg.norm(target - observer)) <= self.max_range

    def occluded(
        self,
        observer: NDArray[np.float64],
        target: NDArray[np.float64],
        obstacles: Sequence[Obstacle],
    ) -> bool:
        """Whether an obstacle body crosses the segment ``observer -> target``."""

        if not self.check_occlusion:
            return False
        return any(
            obstacle.first_intersection(observer, target) is not None
            for obstacle in obstacles
        )

    def sees(
        self,
        observer: NDArray[np.float64],
        target: NDArray[np.float64],
        obstacles: Sequence[Obstacle],
    ) -> bool:
        return self.in_range(observer, target) and not self.occluded(
            observer, target, obstacles
        )

    def visible_neighbours(
        self,
        index: int,
        positions: NDArray[np.float64],
        obstacles: Sequence[Obstacle],
    ) -> tuple[int, ...]:
        """Indices of the vehicles that vehicle ``index`` currently sees."""

        return tuple(
            other
            for other in range(positions.shape[0])
            if other != index
            and self.sees(positions[index], positions[other], obstacles)
        )
