"""Common controller interface."""

from __future__ import annotations

from typing import Protocol

from circumnav.models.dubins import DubinsCommand, DubinsState


class Controller(Protocol):
    """A sampled-data state-feedback controller."""

    def reset(self, initial_state: DubinsState) -> None:
        """Reset internal controller state before a new experiment."""

    def compute(self, time: float, state: DubinsState) -> DubinsCommand:
        """Compute a requested command from the current state."""

