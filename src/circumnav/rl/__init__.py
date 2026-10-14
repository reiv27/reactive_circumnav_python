"""Reinforcement-learning experiments around the reactive circumnavigation task.

See ``docs/RL_plan.md`` for why this exists and what it is not: a bid to beat
the model-based ``PredictorController`` in the nominal case (it structurally
cannot, since the Dubins model and delay are exactly known there), but a
sandbox for the case where they are not.
"""

from __future__ import annotations

from circumnav.rl.env import CircumnavEnv, CircumnavEnvConfig

__all__ = ["CircumnavEnv", "CircumnavEnvConfig"]
