"""Time histories produced by simulation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulationResult:
    """Aligned state and input samples from a completed simulation."""

    time: NDArray[np.float64]
    state: NDArray[np.float64]
    requested_control: NDArray[np.float64]
    applied_control: NDArray[np.float64]
    saturated: NDArray[np.bool_]

    def __post_init__(self) -> None:
        sample_count = self.time.shape[0]
        expected_shapes = {
            "state": (sample_count, 3),
            "requested_control": (sample_count, 2),
            "applied_control": (sample_count, 2),
            "saturated": (sample_count,),
        }
        for name, expected in expected_shapes.items():
            actual = getattr(self, name).shape
            if actual != expected:
                raise ValueError(f"{name} has shape {actual}, expected {expected}")

    @property
    def x(self) -> NDArray[np.float64]:
        return self.state[:, 0]

    @property
    def y(self) -> NDArray[np.float64]:
        return self.state[:, 1]

    @property
    def heading(self) -> NDArray[np.float64]:
        return self.state[:, 2]

    def save_npz(self, path: str | Path) -> None:
        """Save numeric histories in a compact, lossless format."""

        np.savez_compressed(
            Path(path),
            time=self.time,
            state=self.state,
            requested_control=self.requested_control,
            applied_control=self.applied_control,
            saturated=self.saturated,
        )



@dataclass(frozen=True)
class FleetResult:
    """Results of several vehicles simulated on one common time grid."""

    results: tuple[SimulationResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("a fleet needs at least one result")
        reference = self.results[0].time
        for result in self.results[1:]:
            if not np.array_equal(result.time, reference):
                raise ValueError("all fleet results must share one time grid")

    @property
    def robot_count(self) -> int:
        return len(self.results)

    @property
    def time(self) -> NDArray[np.float64]:
        return self.results[0].time

    @property
    def positions(self) -> NDArray[np.float64]:
        """Planar positions, shape ``(robot_count, samples, 2)``."""

        return np.stack([result.state[:, :2] for result in self.results])
