from dataclasses import replace

import numpy as np
import pytest

from circumnav.controllers.delayed import DelayedController
from circumnav.controllers.predictor import PredictorController
from circumnav.models.actuator import DelayedActuator, IdealActuator, LagActuator
from circumnav.models.dubins import (
    DubinsCommand,
    DubinsLimits,
    DubinsModel,
    DubinsState,
)
from circumnav.models.obstacles import CircleObstacle
from circumnav.scenarios.reactive import ReactiveCircumnavScenario
from circumnav.simulation.engine import SimulationConfig, Simulator


class ScriptedController:
    """Returns commands from a fixed script, independent of the state given.

    Records every state it was actually called with, so the test can check
    exactly what PredictorController reconstructed.
    """

    def __init__(self, commands: list[DubinsCommand]) -> None:
        self.commands = commands
        self.seen_states: list[DubinsState] = []
        self._calls = 0

    def reset(self, initial_state: DubinsState) -> None:
        del initial_state
        self.seen_states = []
        self._calls = 0

    def compute(self, time: float, state: DubinsState) -> DubinsCommand:
        del time
        self.seen_states.append(state)
        command = self.commands[self._calls]
        self._calls += 1
        return command


def _ground_truth_trajectory(
    initial: DubinsState, commands: list[DubinsCommand], control_period: float
) -> list[DubinsState]:
    model = DubinsModel()
    states = [initial]
    for command in commands:
        states.append(model.propagate(states[-1], command, control_period))
    return states


def test_zero_delay_reconstructs_the_current_state():
    commands = [DubinsCommand(1.0, 0.3), DubinsCommand(1.0, -0.4), DubinsCommand(1.0, 0.1)]
    initial = DubinsState(0.0, 0.0, 0.0)
    true_states = _ground_truth_trajectory(initial, commands, control_period=0.1)

    inner = ScriptedController(commands)
    predictor = PredictorController(inner, delay_steps=0, control_period=0.1)
    predictor.reset(initial)

    for index, command in enumerate(commands):
        del command
        predictor.compute(float(index) * 0.1, true_states[index])

    for seen, expected in zip(inner.seen_states, true_states[:-1]):
        assert seen == expected


def test_delayed_measurement_is_reconstructed_to_the_exact_current_state():
    # A varied command script so the replay is non-trivial (changing radius
    # and direction every control period).
    commands = [
        DubinsCommand(1.0, 0.3),
        DubinsCommand(1.0, -0.5),
        DubinsCommand(1.0, 0.2),
        DubinsCommand(1.0, 0.0),
        DubinsCommand(1.0, 0.4),
        DubinsCommand(1.0, -0.1),
    ]
    control_period = 0.1
    initial = DubinsState(-3.0, 2.0, 0.7)
    true_states = _ground_truth_trajectory(initial, commands, control_period)

    inner = ScriptedController(commands)
    delay_steps = 2
    predictor = PredictorController(inner, delay_steps, control_period)
    predictor.reset(initial)

    for index in range(len(commands)):
        predictor.compute(float(index) * control_period, true_states[index])

    # Every call must see a reconstructed estimate equal to the TRUE current
    # state at that call -- i.e. the sensing delay has been fully undone.
    assert len(inner.seen_states) == len(commands)
    for call_index, (seen, expected) in enumerate(zip(inner.seen_states, true_states)):
        assert seen.x == pytest.approx(expected.x, abs=1e-9), call_index
        assert seen.y == pytest.approx(expected.y, abs=1e-9), call_index
        assert seen.heading == pytest.approx(expected.heading, abs=1e-9), call_index


def test_time_argument_is_never_delayed():
    commands = [DubinsCommand(1.0, 0.1)] * 4
    inner = ScriptedController(commands)
    predictor = PredictorController(inner, delay_steps=2, control_period=0.02)
    predictor.reset(DubinsState(0.0, 0.0, 0.0))

    seen_times = []
    original_compute = inner.compute

    def spy(time, state):
        seen_times.append(time)
        return original_compute(time, state)

    inner.compute = spy
    state = DubinsState(0.0, 0.0, 0.0)
    for index in range(4):
        predictor.compute(float(index) * 0.02, state)

    assert seen_times == pytest.approx([0.0, 0.02, 0.04, 0.06])


def test_reset_clears_history():
    commands = [DubinsCommand(1.0, 0.2)] * 5
    inner = ScriptedController(commands)
    predictor = PredictorController(inner, delay_steps=2, control_period=0.1)
    predictor.reset(DubinsState(0.0, 0.0, 0.0))
    for index in range(3):
        predictor.compute(float(index) * 0.1, DubinsState(float(index), 0.0, 0.0))

    predictor.reset(DubinsState(9.0, 9.0, 0.0))
    fresh_state = DubinsState(9.0, 9.0, 0.0)
    predictor.compute(0.0, fresh_state)

    # Right after reset the history is empty, so the very next call must see
    # exactly the state it was given, unmodified by any stale replay.
    assert inner.seen_states[-1] == fresh_state


def test_negative_delay_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        PredictorController(ScriptedController([]), delay_steps=-1, control_period=0.02)


def test_non_positive_control_period_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        PredictorController(ScriptedController([]), delay_steps=1, control_period=0.0)


def test_attribute_access_is_forwarded_to_the_inner_controller():
    obstacles = (CircleObstacle(center=[0.0, 0.0], radius=4.0, obstacle_id=0),)
    scenario = ReactiveCircumnavScenario(duration=2.0, obstacles=obstacles)
    inner = scenario.build_controller()
    predictor = PredictorController(inner, delay_steps=1, control_period=scenario.control_period)

    predictor.reset(DubinsState(-12.0, 0.0, 0.0))
    predictor.compute(0.0, DubinsState(-12.0, 0.0, 0.0))

    assert predictor.switch_count == inner.switch_count
    assert predictor.log is inner.log
    assert predictor.config.safety_distance == scenario.safety_distance


def test_scenario_uses_predictor_only_when_compensation_is_requested():
    base = ReactiveCircumnavScenario(
        duration=2.0, sensing_delay=0.02 * 3, control_period=0.02
    )

    _, uncompensated_controller = base.run()
    assert isinstance(uncompensated_controller, DelayedController)

    compensated = replace(base, sensing_compensation=True)
    _, compensated_controller = compensated.run()
    assert isinstance(compensated_controller, PredictorController)


def test_predictor_reproduces_the_sensing_delay_free_baseline():
    """The central acceptance test for stage 1a.

    With actuation_delay=0 and actuator_time_constant=0 the predictor's
    reconstruction is exact, so a run with sensing_delay>0 and
    sensing_compensation=True must reproduce the sensing_delay=0 baseline
    almost bit-for-bit -- not just "improved metrics".
    """

    baseline_scenario = ReactiveCircumnavScenario(duration=20.0)
    baseline_result, baseline_controller = baseline_scenario.run()

    compensated_scenario = replace(
        baseline_scenario,
        sensing_delay=baseline_scenario.control_period * 5,
        sensing_compensation=True,
    )
    compensated_result, compensated_controller = compensated_scenario.run()

    assert isinstance(compensated_controller, PredictorController)
    np.testing.assert_allclose(
        compensated_result.state, baseline_result.state, atol=1e-8
    )
    np.testing.assert_allclose(
        compensated_result.applied_control, baseline_result.applied_control, atol=1e-8
    )

    baseline_d_r = np.array(
        [entry.equidistant_deviation for entry in baseline_controller.log]
    )
    compensated_d_r = np.array(
        [entry.equidistant_deviation for entry in compensated_controller.log]
    )
    np.testing.assert_allclose(compensated_d_r, baseline_d_r, atol=1e-8)
    assert compensated_controller.switch_count == baseline_controller.switch_count


# --- stage 1b: backward reconstruction stays exact with a non-ideal shadow
# actuator (actuation delay and/or lag), as long as sensing is the only
# channel being compensated. This does NOT yet compensate tau_a/lag
# themselves (see controllers/predictor.py's module docstring).


def _generous_limits() -> DubinsLimits:
    return DubinsLimits(min_speed=0.0, max_speed=10.0, max_yaw_rate=10.0)


def test_predictor_reconstructs_exactly_under_joint_sensing_and_actuation_delay():
    control_period = 0.1
    integration_step = 0.02
    actuation_delay = 0.04
    delay_steps = 2
    commands = [
        DubinsCommand(1.0, 0.3),
        DubinsCommand(1.0, -0.5),
        DubinsCommand(1.0, 0.2),
        DubinsCommand(1.0, 0.0),
        DubinsCommand(1.0, 0.4),
    ]
    initial = DubinsState(-2.0, 1.0, 0.4)
    cruise = DubinsCommand(1.0, 0.0)

    def build_actuator() -> DelayedActuator:
        return DelayedActuator(
            IdealActuator(_generous_limits()), actuation_delay, integration_step, cruise
        )

    # Ground truth: drive a real (non-shadow) instance of the same actuator
    # chain directly.
    reference_actuator = build_actuator()
    model = DubinsModel()
    substeps = round(control_period / integration_step)
    true_states = [initial]
    state = initial
    for command in commands:
        for _ in range(substeps):
            applied = reference_actuator.apply(command, integration_step).applied
            state = model.propagate(state, applied, integration_step)
        true_states.append(state)

    inner = ScriptedController(commands)
    predictor = PredictorController(
        inner,
        delay_steps=delay_steps,
        control_period=control_period,
        integration_step=integration_step,
        actuator_factory=build_actuator,
    )
    predictor.reset(initial)
    for index in range(len(commands)):
        predictor.compute(float(index) * control_period, true_states[index])

    assert len(inner.seen_states) == len(commands)
    for call_index, (seen, expected) in enumerate(zip(inner.seen_states, true_states)):
        assert seen.x == pytest.approx(expected.x, abs=1e-9), call_index
        assert seen.y == pytest.approx(expected.y, abs=1e-9), call_index
        assert seen.heading == pytest.approx(expected.heading, abs=1e-9), call_index


def test_predictor_reconstructs_exactly_under_joint_sensing_delay_and_lag():
    control_period = 0.1
    integration_step = 0.02
    time_constant = 0.05
    delay_steps = 2
    commands = [
        DubinsCommand(1.0, 0.3),
        DubinsCommand(1.0, -0.5),
        DubinsCommand(1.0, 0.2),
        DubinsCommand(1.0, 0.0),
        DubinsCommand(1.0, 0.4),
    ]
    initial = DubinsState(-2.0, 1.0, 0.4)
    cruise = DubinsCommand(1.0, 0.0)

    def build_actuator() -> LagActuator:
        return LagActuator(IdealActuator(_generous_limits()), time_constant, cruise)

    reference_actuator = build_actuator()
    model = DubinsModel()
    substeps = round(control_period / integration_step)
    true_states = [initial]
    state = initial
    for command in commands:
        for _ in range(substeps):
            applied = reference_actuator.apply(command, integration_step).applied
            state = model.propagate(state, applied, integration_step)
        true_states.append(state)

    inner = ScriptedController(commands)
    predictor = PredictorController(
        inner,
        delay_steps=delay_steps,
        control_period=control_period,
        integration_step=integration_step,
        actuator_factory=build_actuator,
    )
    predictor.reset(initial)
    for index in range(len(commands)):
        predictor.compute(float(index) * control_period, true_states[index])

    for call_index, (seen, expected) in enumerate(zip(inner.seen_states, true_states)):
        assert seen.x == pytest.approx(expected.x, abs=1e-9), call_index
        assert seen.y == pytest.approx(expected.y, abs=1e-9), call_index
        assert seen.heading == pytest.approx(expected.heading, abs=1e-9), call_index


def test_scenario_predictor_matches_the_same_actuation_delay_without_sensing_delay():
    """The predictor must leave tau_a's own (uncompensated) effect untouched.

    Comparing against the sensing-delay-free baseline (as in stage 1a) would
    be the wrong target here: tau_a is still present and NOT compensated by
    this controller, so the fair reference is a run with the same tau_a and
    zero sensing delay.
    """

    reference_scenario = ReactiveCircumnavScenario(duration=20.0, actuation_delay=0.04)
    reference_result, _ = reference_scenario.run()

    joint_scenario = replace(
        reference_scenario,
        sensing_delay=reference_scenario.control_period * 3,
        sensing_compensation=True,
    )
    joint_result, joint_controller = joint_scenario.run()

    assert isinstance(joint_controller, PredictorController)
    np.testing.assert_allclose(
        joint_result.state, reference_result.state, atol=1e-7
    )
    np.testing.assert_allclose(
        joint_result.applied_control, reference_result.applied_control, atol=1e-7
    )


def test_scenario_predictor_matches_the_same_lag_without_sensing_delay():
    reference_scenario = ReactiveCircumnavScenario(
        duration=20.0, actuator_time_constant=0.05
    )
    reference_result, _ = reference_scenario.run()

    joint_scenario = replace(
        reference_scenario,
        sensing_delay=reference_scenario.control_period * 3,
        sensing_compensation=True,
    )
    joint_result, joint_controller = joint_scenario.run()

    assert isinstance(joint_controller, PredictorController)
    np.testing.assert_allclose(
        joint_result.state, reference_result.state, atol=1e-7
    )


def test_scenario_predictor_matches_the_same_actuation_delay_and_lag_together():
    reference_scenario = ReactiveCircumnavScenario(
        duration=20.0, actuation_delay=0.04, actuator_time_constant=0.05
    )
    reference_result, _ = reference_scenario.run()

    joint_scenario = replace(
        reference_scenario,
        sensing_delay=reference_scenario.control_period * 3,
        sensing_compensation=True,
    )
    joint_result, joint_controller = joint_scenario.run()

    assert isinstance(joint_controller, PredictorController)
    np.testing.assert_allclose(
        joint_result.state, reference_result.state, atol=1e-7
    )


def test_delay_steps_zero_is_a_no_op_even_with_an_actuator_factory():
    """Backward reconstruction alone cannot fix a sensing-delay-free loop.

    With delay_steps=0 there is nothing to reconstruct, so the predictor must
    leave actuation delay/lag's own effect completely untouched -- this is
    the documented scope boundary (tau_a/lag compensation is a separate,
    not-yet-implemented increment).
    """

    scenario = ReactiveCircumnavScenario(duration=10.0, actuation_delay=0.04)
    uncompensated_result, _ = scenario.run()

    inner = scenario.build_controller()
    predictor = PredictorController(
        inner,
        delay_steps=0,
        control_period=scenario.control_period,
        integration_step=scenario.integration_step,
        actuator_factory=scenario.build_actuator,
    )
    simulator = Simulator(
        model=DubinsModel(),
        actuator=scenario.build_actuator(),
        config=SimulationConfig(
            duration=scenario.duration,
            integration_step=scenario.integration_step,
            control_period=scenario.control_period,
        ),
    )
    predictor_result = simulator.run(
        initial_state=DubinsState(
            scenario.initial_x, scenario.initial_y, scenario.initial_heading
        ),
        controller=predictor,
    )

    np.testing.assert_allclose(
        predictor_result.state, uncompensated_result.state, atol=1e-9
    )


def test_uncompensated_delay_actually_diverges_from_the_baseline():
    """Sanity check that the delay used above is not trivially harmless."""

    baseline_scenario = ReactiveCircumnavScenario(duration=20.0)
    baseline_result, _ = baseline_scenario.run()

    uncompensated_scenario = replace(
        baseline_scenario,
        sensing_delay=baseline_scenario.control_period * 5,
        sensing_compensation=False,
    )
    uncompensated_result, _ = uncompensated_scenario.run()

    assert not np.allclose(
        uncompensated_result.state, baseline_result.state, atol=1e-3
    )
