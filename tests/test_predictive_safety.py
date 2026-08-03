from __future__ import annotations

from dataclasses import replace

from usvlib4ros.planning import (
    Control,
    PlanningMapSnapshot,
    PrototypeReducedDynamics,
    VesselState,
)
from usvlib4ros.policy.safety_supervisor import (
    CandidateControl,
    CandidateControlGenerator,
    PredictiveSafetySupervisor,
    minimum_intervention_action,
)


def _world() -> PlanningMapSnapshot:
    return PlanningMapSnapshot.from_rows(
        (
            "............",
            "............",
            "............",
            "............",
            "............",
            "............",
        ),
        snapshot_id="map-safety-v0",
        session_id="session-safety",
        source_version=1,
        resolution=1.0,
        stamp_sim=10.0,
    )


def _state() -> VesselState:
    return VesselState(
        x=2.0,
        y=3.0,
        yaw=0.0,
        speed=0.4,
        yaw_rate=0.0,
        frame_id="map",
        stamp_sim=10.0,
    )


def test_candidate_generator_produces_five_complete_controls():
    candidates = CandidateControlGenerator().generate(
        nominal_throttle=0.5,
        nominal_rudder=0.0,
    )
    assert tuple(candidate.action for candidate in candidates) == (0, 1, 2, 3, 4)
    assert all(
        -0.1 <= candidate.control.rudder <= 0.1
        for candidate in candidates
    )
    assert all(
        0.0 <= candidate.control.throttle <= 0.1
        for candidate in candidates
    )


def test_profile_hard_actions_keep_both_full_rudder_endpoints():
    candidates = CandidateControlGenerator(
        max_throttle=0.4,
        max_abs_rudder=0.5,
        action_controls=(
            Control(0.1, -0.5),
            Control(0.4, -0.2),
            Control(0.4, 0.0),
            Control(0.4, 0.2),
            Control(0.4, 0.5),
        ),
    ).generate(0.4, 0.5)

    assert candidates[0].control == Control(0.1, -0.5)
    assert candidates[2].control == Control(0.4, 0.5)
    assert candidates[4].control == Control(0.4, 0.5)


def test_minimum_intervention_rejects_a_hard_sac_turn_when_soft_is_safe():
    candidates = tuple(
        CandidateControl(index, control)
        for index, control in enumerate(
            (
                Control(0.1, -0.5),
                Control(0.4, -0.2),
                Control(0.4, 0.0),
                Control(0.4, 0.2),
                Control(0.4, 0.5),
            )
        )
    )

    action = minimum_intervention_action(
        policy_action=4,
        safe_action_mask=(False, False, False, True, True),
        candidates=candidates,
        nominal_control=Control(0.4, 0.0),
    )

    assert action == 3


def test_minimum_intervention_keeps_sac_as_tie_breaker():
    candidates = tuple(
        CandidateControl(index, control)
        for index, control in enumerate(
            (
                Control(0.4, -0.4),
                Control(0.4, -0.2),
                Control(0.4, 0.0),
                Control(0.4, 0.2),
                Control(0.4, 0.4),
            )
        )
    )

    action = minimum_intervention_action(
        policy_action=3,
        safe_action_mask=(False, True, False, True, False),
        candidates=candidates,
        nominal_control=Control(0.4, 0.0),
    )

    assert action == 3


def test_minimum_intervention_prefers_safe_nominal_over_sac_deviation():
    candidates = CandidateControlGenerator(
        max_throttle=0.4,
        max_abs_rudder=0.5,
        action_controls=(
            Control(0.1, -0.5),
            Control(0.4, -0.2),
            Control(0.4, 0.0),
            Control(0.4, 0.2),
            Control(0.4, 0.5),
        ),
    ).generate(0.4, 0.1)

    action = minimum_intervention_action(
        policy_action=0,
        safe_action_mask=(True,) * 5,
        candidates=candidates,
        nominal_control=Control(0.4, 0.1),
    )

    assert action == 2


def test_precheck_and_final_arbitration_keep_safe_policy_action():
    world = _world()
    dynamics = PrototypeReducedDynamics()
    supervisor = PredictiveSafetySupervisor(prediction_horizon_s=0.5)
    candidates = CandidateControlGenerator().generate(0.5, 0.0)

    mask, reasons, clearances = supervisor.precheck(
        _state(), candidates, world, dynamics, now_sim=10.0
    )
    assert any(mask)
    assert len(reasons) == len(clearances) == 5

    decision = supervisor.finalize(
        policy_action=2,
        candidate_mask=mask,
        candidates=candidates,
        snapshot_id=world.snapshot_id,
        current_snapshot_id=world.snapshot_id,
        current_state=_state(),
        current_map_snapshot=world,
        dynamics=dynamics,
        now_sim=10.0,
    )
    assert decision.final_action == 2
    assert decision.stop is False
    assert decision.control == candidates[2].control


def test_clearance_recovery_only_approves_motion_back_into_strict_safe_area():
    world = PlanningMapSnapshot.from_rows(
        ("." * 50,) * 20,
        snapshot_id="map-clearance-recovery",
        session_id="session-clearance-recovery",
        source_version=1,
        resolution=0.1,
        footprint_radius=0.0,
        required_clearance=0.2,
        stamp_sim=10.0,
    )
    state = VesselState(
        x=0.15,
        y=1.0,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        frame_id="map",
        stamp_sim=10.0,
    )
    dynamics = PrototypeReducedDynamics()
    supervisor = PredictiveSafetySupervisor(prediction_horizon_s=2.0)

    assert not world.is_state_valid(state)
    assert supervisor.clearance_recovery_is_safe(
        state,
        Control(0.1, 0.0),
        world,
        dynamics,
        now_sim=10.0,
        minimum_clearance_m=0.1,
    )
    assert not supervisor.clearance_recovery_is_safe(
        replace(state, yaw=3.141592653589793),
        Control(0.1, 0.0),
        world,
        dynamics,
        now_sim=10.0,
        minimum_clearance_m=0.1,
    )


def test_precheck_retries_one_bounded_horizon_only_after_primary_deadlock(
    monkeypatch,
):
    supervisor = PredictiveSafetySupervisor(prediction_horizon_s=2.0)
    candidates = CandidateControlGenerator().generate(0.5, 0.0)
    calls = []

    def fake_precheck(*args, prediction_horizon_s=None, **kwargs):
        calls.append(prediction_horizon_s)
        if prediction_horizon_s == 2.0:
            return (False,) * 5, ("COLLISION",) * 5, (0.0,) * 5
        return (True,) * 5, ("SAFE",) * 5, (0.4,) * 5

    monkeypatch.setattr(supervisor, "precheck", fake_precheck)

    mask, reasons, clearances, horizon = (
        supervisor.precheck_with_horizon_fallback(
            _state(),
            candidates,
            _world(),
            PrototypeReducedDynamics(),
            now_sim=10.0,
            primary_horizon_s=2.0,
            fallback_horizon_s=1.1,
        )
    )

    assert calls == [2.0, 1.1]
    assert mask == (True,) * 5
    assert reasons == ("SAFE",) * 5
    assert clearances == (0.4,) * 5
    assert horizon == 1.1


def test_candidate_prefix_then_nominal_future_is_reused_by_final_recheck():
    world = PlanningMapSnapshot.from_rows(
        (
            "..........",
            "..........",
            "..........",
            "....#.....",
            "..........",
            "..........",
        ),
        snapshot_id="future-controls-map",
        session_id="future-controls-session",
        source_version=1,
        resolution=1.0,
        required_clearance=0.05,
    )
    state = VesselState(
        x=2.0,
        y=3.5,
        yaw=0.0,
        speed=1.0,
        yaw_rate=0.0,
    )
    candidates = CandidateControlGenerator(
        max_throttle=1.0,
        max_abs_rudder=0.1,
    ).generate(1.0, 0.0)
    dynamics = PrototypeReducedDynamics()
    supervisor = PredictiveSafetySupervisor(prediction_horizon_s=2.0)

    constant_mask, _, _ = supervisor.precheck(
        state,
        candidates,
        world,
        dynamics,
        now_sim=0.0,
    )
    future = ((Control(0.0, 0.0), 1.7),)
    mask, reasons, clearances = supervisor.precheck(
        state,
        candidates,
        world,
        dynamics,
        now_sim=0.0,
        candidate_prefix_s=0.3,
        nominal_future_controls=future,
    )
    decision = supervisor.finalize(
        policy_action=2,
        nominal_action=2,
        candidate_mask=mask,
        candidates=candidates,
        snapshot_id=world.snapshot_id,
        current_snapshot_id=world.snapshot_id,
        reasons=reasons,
        clearances=clearances,
        current_state=state,
        current_map_snapshot=world,
        dynamics=dynamics,
        now_sim=0.0,
        candidate_prefix_s=0.3,
        nominal_future_controls=future,
    )

    assert not constant_mask[2]
    assert mask[2]
    assert not decision.stop
    assert decision.final_action == 2


def test_stale_version_or_empty_safe_set_returns_stop_and_zero_control():
    candidates = CandidateControlGenerator().generate(0.5, 0.0)
    supervisor = PredictiveSafetySupervisor()

    stale = supervisor.finalize(
        policy_action=2,
        candidate_mask=(True,) * 5,
        candidates=candidates,
        snapshot_id="map-v1",
        current_snapshot_id="map-v2",
    )
    assert stale.stop is True
    assert stale.final_action is None
    assert stale.control == Control(0.0, 0.0)

    empty = supervisor.finalize(
        policy_action=None,
        candidate_mask=(False,) * 5,
        candidates=candidates,
        snapshot_id="map-v1",
        current_snapshot_id="map-v1",
    )
    assert empty.stop is True
    assert empty.reason == "NO_SAFE_ACTION"
    assert empty.control == Control(0.0, 0.0)


def test_final_arbitration_requires_complete_latest_context():
    world = _world()
    dynamics = PrototypeReducedDynamics()
    supervisor = PredictiveSafetySupervisor(prediction_horizon_s=0.5)
    candidates = CandidateControlGenerator().generate(0.5, 0.0)
    mask, _, _ = supervisor.precheck(_state(), candidates, world, dynamics, now_sim=10.0)

    decision = supervisor.finalize(
        policy_action=2,
        candidate_mask=mask,
        candidates=candidates,
        snapshot_id=world.snapshot_id,
        current_snapshot_id=world.snapshot_id,
    )

    assert decision.stop is True
    assert decision.reason == "LATEST_CONTEXT_REQUIRED"
    assert decision.control == Control(0.0, 0.0)


def test_final_arbitration_stops_for_malformed_precheck_reasons():
    world = _world()
    dynamics = PrototypeReducedDynamics()
    supervisor = PredictiveSafetySupervisor(prediction_horizon_s=0.5)
    candidates = CandidateControlGenerator().generate(0.5, 0.0)

    decision = supervisor.finalize(
        policy_action=2,
        candidate_mask=(True,) * 5,
        candidates=candidates,
        snapshot_id=world.snapshot_id,
        current_snapshot_id=world.snapshot_id,
        reasons=("SAFE",),
        current_state=_state(),
        current_map_snapshot=world,
        dynamics=dynamics,
        now_sim=10.0,
    )

    assert decision.stop is True
    assert decision.reason == "INVALID_INPUT"
    assert decision.control == Control(0.0, 0.0)


def test_final_arbitration_rechecks_latest_state_even_when_snapshot_id_is_unchanged():
    world = _world()
    dynamics = PrototypeReducedDynamics()
    supervisor = PredictiveSafetySupervisor(prediction_horizon_s=0.5)
    candidates = CandidateControlGenerator().generate(0.5, 0.0)

    decision = supervisor.finalize(
        policy_action=2,
        candidate_mask=(True,) * 5,
        candidates=candidates,
        snapshot_id=world.snapshot_id,
        current_snapshot_id=world.snapshot_id,
        current_state=replace(_state(), x=11.95),
        current_map_snapshot=world,
        dynamics=dynamics,
        now_sim=10.0,
    )

    assert decision.stop is True
    assert decision.reason == "LATEST_INPUT_UNSAFE"
    assert decision.control == Control(0.0, 0.0)


def test_final_arbitration_rejects_a_current_map_object_with_a_different_snapshot_id():
    world = _world()
    candidates = CandidateControlGenerator().generate(0.5, 0.0)
    decision = PredictiveSafetySupervisor(prediction_horizon_s=0.5).finalize(
        policy_action=2,
        candidate_mask=(True,) * 5,
        candidates=candidates,
        snapshot_id=world.snapshot_id,
        current_snapshot_id=world.snapshot_id,
        current_state=_state(),
        current_map_snapshot=replace(world, snapshot_id="map-safety-v1"),
        dynamics=PrototypeReducedDynamics(),
        now_sim=10.0,
    )

    assert decision.stop is True
    assert decision.reason == "CURRENT_MAP_SNAPSHOT_MISMATCH"
    assert decision.control == Control(0.0, 0.0)


def test_non_finite_state_fails_closed_before_prediction():
    bad_state = replace(_state(), x=float("nan"))
    candidates = CandidateControlGenerator().generate(0.5, 0.0)
    mask, reasons, clearances = PredictiveSafetySupervisor().precheck(
        bad_state,
        candidates,
        _world(),
        PrototypeReducedDynamics(),
        now_sim=10.0,
    )
    assert mask == (False,) * 5
    assert all(reason == "INVALID_INPUT" for reason in reasons)
    assert all(clearance == 0.0 for clearance in clearances)


def test_precheck_fails_closed_for_an_invalid_execution_horizon_override():
    candidates = CandidateControlGenerator().generate(0.5, 0.0)
    mask, reasons, clearances = PredictiveSafetySupervisor().precheck(
        _state(),
        candidates,
        _world(),
        PrototypeReducedDynamics(),
        now_sim=10.0,
        prediction_horizon_s="not-a-duration",
    )

    assert mask == (False,) * 5
    assert reasons == ("INVALID_INPUT",) * 5
    assert clearances == (0.0,) * 5


def test_safe_policy_action_is_not_overridden_by_safe_nominal_progress_action():
    world = _world()
    dynamics = PrototypeReducedDynamics()
    supervisor = PredictiveSafetySupervisor(prediction_horizon_s=0.5)
    candidates = CandidateControlGenerator().generate(0.5, 0.0)
    mask, reasons, clearances = supervisor.precheck(
        _state(), candidates, world, dynamics, now_sim=10.0
    )

    decision = supervisor.finalize(
        policy_action=0,
        nominal_action=2,
        candidate_mask=mask,
        candidates=candidates,
        snapshot_id=world.snapshot_id,
        current_snapshot_id=world.snapshot_id,
        reasons=reasons,
        clearances=clearances,
        current_state=_state(),
        current_map_snapshot=world,
        dynamics=dynamics,
        now_sim=10.0,
    )

    assert decision.final_action == 0
    assert decision.reason == "POLICY_ACTION_SAFE"
    assert decision.overridden is False
    assert decision.minimum_clearance == clearances[0]


def test_unsafe_policy_action_falls_back_to_safe_nominal_progress_action():
    world = _world()
    dynamics = PrototypeReducedDynamics()
    supervisor = PredictiveSafetySupervisor(prediction_horizon_s=0.5)
    candidates = CandidateControlGenerator().generate(0.5, 0.0)
    mask, reasons, clearances = supervisor.precheck(
        _state(), candidates, world, dynamics, now_sim=10.0
    )
    mask = (False,) + mask[1:]

    decision = supervisor.finalize(
        policy_action=0,
        nominal_action=2,
        candidate_mask=mask,
        candidates=candidates,
        snapshot_id=world.snapshot_id,
        current_snapshot_id=world.snapshot_id,
        reasons=reasons,
        clearances=clearances,
        current_state=_state(),
        current_map_snapshot=world,
        dynamics=dynamics,
        now_sim=10.0,
    )

    assert decision.final_action == 2
    assert decision.reason == "POLICY_ACTION_UNSAFE"
    assert decision.overridden is True
    assert decision.minimum_clearance == clearances[2]
