"""Sensing delay: wrap a controller so it perceives a stale state.

This models the ``tau_s`` insertion point (README on delay, Section 8): the
measurement the controller acts on is ``x(t - tau_s)``, not ``x(t)``. It is
*not* equivalent to delaying the command after computing it (the ``tau_c`` /
``tau_a`` case, which reduces to a pure delay on the requested-to-applied
path and belongs on the actuator instead): here the state itself is stale, so
a hybrid, nonlinear controller such as :class:`ReactiveCircumnavController`
also detects mode switches (``C``/``G``) on outdated geometry, which a pure
actuation delay cannot reproduce.

The wrapper delays only ``state``. ``time`` is passed through unchanged, so
control updates still happen on the real clock and diagnostic logs (e.g.
``ReactiveCircumnavController.log``) stay timestamped correctly; only the
*content* the controller reasons about is old.
"""

from __future__ import annotations

from collections import deque

from circumnav.controllers.base import Controller
from circumnav.models.dubins import DubinsCommand, DubinsState


class DelayedController:
    """Feed ``inner`` the state from ``delay_steps`` control updates ago.

    Every attribute other than ``reset``/``compute`` is forwarded to
    ``inner``, so diagnostics (``.log``, ``.mode``, ``.switch_count``,
    ``.config``, ...) keep working exactly as for the unwrapped controller.
    """

    def __init__(self, inner: Controller, delay_steps: int) -> None:
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self.inner = inner
        self.delay_steps = delay_steps
        self._history: deque[DubinsState] = deque()

    def reset(self, initial_state: DubinsState) -> None:
        self.inner.reset(initial_state)
        self._history.clear()

    def compute(self, time: float, state: DubinsState) -> DubinsCommand:
        self._history.append(state)
        if len(self._history) > self.delay_steps:
            delayed_state = self._history.popleft()
        else:
            delayed_state = self._history[0]
        return self.inner.compute(time, delayed_state)

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)
