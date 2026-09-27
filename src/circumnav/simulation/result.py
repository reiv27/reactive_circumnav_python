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

