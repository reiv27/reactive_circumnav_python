"""Open-loop controllers useful for plant verification."""

from __future__ import annotations

from circumnav.models.dubins import DubinsCommand, DubinsState


class ConstantController:
    """Return the same command at every controller update."""

    def __init__(self, command: DubinsCommand) -> None:
        self.command = command

    def reset(self, initial_state: DubinsState) -> None:
        del initial_state

    def compute(self, time: float, state: DubinsState) -> DubinsCommand:
        del time, state
        return self.command

