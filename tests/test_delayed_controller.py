import pytest

from circumnav.controllers.delayed import DelayedController
from circumnav.controllers.reactive import CircumnavMode
from circumnav.models.dubins import DubinsCommand, DubinsState
from circumnav.models.obstacles import CircleObstacle
from circumnav.scenarios.reactive import ReactiveCircumnavScenario


class RecordingController:
    """A minimal Controller that just remembers every state it was given."""

    def __init__(self) -> None:
        self.seen_states: list[DubinsState] = []

    def reset(self, initial_state: DubinsState) -> None:
        del initial_state
        self.seen_states = []

    def compute(self, time: float, state: DubinsState) -> DubinsCommand:
        del time
        self.seen_states.append(state)
        return DubinsCommand(speed=1.0, yaw_rate=0.0)


def make_states(n):
    return [DubinsState(float(i), 0.0, 0.0) for i in range(n)]


def test_zero_delay_passes_the_current_state_through():
    inner = RecordingController()
    delayed = DelayedController(inner, delay_steps=0)
    delayed.reset(DubinsState(0.0, 0.0, 0.0))

    states = make_states(5)
    for i, state in enumerate(states):
        delayed.compute(float(i), state)

    assert inner.seen_states == states


def test_delay_steps_shifts_the_state_by_that_many_updates():
    inner = RecordingController()
    delayed = DelayedController(inner, delay_steps=3)
    delayed.reset(DubinsState(0.0, 0.0, 0.0))

    states = make_states(8)
    for i, state in enumerate(states):
        delayed.compute(float(i), state)

    # The first 3 updates see the earliest available state (no history yet);
    # from the 4th update onward they see the state from 3 updates back.
    expected = [states[0], states[0], states[0]] + states[:5]
    assert inner.seen_states == expected


def test_time_argument_is_never_delayed():
    inner = RecordingController()
    delayed = DelayedController(inner, delay_steps=2)
    delayed.reset(DubinsState(0.0, 0.0, 0.0))

    seen_times = []
    original_compute = inner.compute

    def spy(time, state):
        seen_times.append(time)
        return original_compute(time, state)

    inner.compute = spy
    for i, state in enumerate(make_states(4)):
        delayed.compute(float(i) * 0.02, state)

    assert seen_times == pytest.approx([0.0, 0.02, 0.04, 0.06])


def test_reset_clears_history():
    inner = RecordingController()
    delayed = DelayedController(inner, delay_steps=2)
    delayed.reset(DubinsState(0.0, 0.0, 0.0))
    for state in make_states(5):
        delayed.compute(0.0, state)

    delayed.reset(DubinsState(9.0, 9.0, 0.0))
    fresh_state = DubinsState(9.0, 9.0, 0.0)
    delayed.compute(0.0, fresh_state)

    assert inner.seen_states[-1] is fresh_state


def test_negative_delay_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        DelayedController(RecordingController(), delay_steps=-1)


def test_attribute_access_is_forwarded_to_the_inner_controller():
    obstacles = (CircleObstacle(center=[0.0, 0.0], radius=4.0, obstacle_id=0),)
    scenario = ReactiveCircumnavScenario(duration=2.0, obstacles=obstacles)
    inner = scenario.build_controller()
    delayed = DelayedController(inner, delay_steps=1)

    delayed.reset(DubinsState(-12.0, 0.0, 0.0))
    delayed.compute(0.0, DubinsState(-12.0, 0.0, 0.0))

    # These are all defined on the inner controller, not on DelayedController.
    assert delayed.mode is CircumnavMode.APPROACH
    assert delayed.switch_count == inner.switch_count
    assert delayed.log is inner.log
    assert delayed.config.safety_distance == scenario.safety_distance


def test_scenario_wraps_the_controller_when_sensing_delay_is_set():
    scenario = ReactiveCircumnavScenario(
        duration=2.0,
        sensing_delay=0.02 * 4,
        control_period=0.02,
    )

    assert scenario.sensing_delay_steps == 4

    result, controller = scenario.run()

    assert isinstance(controller, DelayedController)
    assert result.time[-1] == pytest.approx(2.0)


def test_scenario_sensing_delay_must_be_a_multiple_of_control_period():
    scenario = ReactiveCircumnavScenario(
        duration=2.0, sensing_delay=0.025, control_period=0.02
    )

    with pytest.raises(ValueError, match="integer multiple"):
        scenario.run()


def test_zero_sensing_delay_leaves_the_controller_unwrapped():
    scenario = ReactiveCircumnavScenario(duration=2.0, sensing_delay=0.0)

    result, controller = scenario.run()

    assert not isinstance(controller, DelayedController)
    assert result.time[-1] == pytest.approx(2.0)
