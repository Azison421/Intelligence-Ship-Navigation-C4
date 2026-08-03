"""Fail-closed runtime for the fixed National_Test route."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from usvlib4ros.mapping import (
    CompiledSidecarMap,
    GpsProjector,
    compass_yaw_deg_to_math_yaw_rad,
    compass_yaw_rate_degs_to_math_rad_s,
    enu_to_grid,
    fit_route_converter,
    load_sidecar_artifact,
    math_yaw_rad_to_compass_deg,
    unity_point_in_water,
)
from usvlib4ros.mapping.coordinates import AffineTransform2D
from usvlib4ros.navigation.reverse_control_calibration import (
    ReverseControlProfile,
    enable_reverse_dynamics,
    reverse_control_profile_from_dict,
)
from usvlib4ros.planning import (
    Control,
    PrototypeReducedDynamics,
    Trajectory,
    VesselState,
)
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    action_protocol_hash,
    forward_control_profile_from_dict,
    reduced_dynamics_from_profile,
)
from usvlib4ros.planning.fixed_route import (
    NARROW_ROUTE_INDEX,
    SIDECAR_PATH,
    ROUTE_GUIDANCE_VERSION,
    compile_offline_national_map,
    fixed_route_gate_region,
    fixed_route_goal_xy,
    fixed_route_guidance_hash,
    fixed_route_waypoint_reached,
    narrow_escape_released,
    plan_fixed_leg,
)
from usvlib4ros.policy.fixed_map_features import (
    braking_future_controls,
    build_fixed_map_observation,
    feedback_tracking_control,
    front_arc_laser_features,
    narrow_ingress_control,
    narrow_ingress_future_controls,
    preview_trajectory,
    reverse_tracking_control,
    tracking_future_controls,
)
from usvlib4ros.policy.recurrent_sac import (
    RecurrentDiscreteSAC,
    RecurrentHiddenState,
)
from usvlib4ros.policy.safety_supervisor import (
    CandidateControl,
    CandidateControlGenerator,
    FIXED_MAP_PREDICTION_HORIZON_S,
    MINIMUM_INTERVENTION_GATE_VERSION,
    PREDICTION_HORIZON_POLICY_VERSION,
    PredictiveSafetySupervisor,
    minimum_intervention_action,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "national_test_sac_live_v10.pt"
)
ROUTE_FIT_TOLERANCE_M = 0.05
APPROVED_TRANSFORM_TOLERANCE_M = 0.05
CONVERTER_SCALE_BAND = (0.5, 2.0)
POSE_MAX_AGE_S = 0.5
SCAN_MAX_AGE_S = 1.0
DEVICE_MAX_AGE_S = 1.0
LASER_EMERGENCY_DISTANCE_M = 0.6
CLEARANCE_RECOVERY_MINIMUM_M = 0.1
CLEARANCE_RECOVERY_MAX_SPEED_MPS = 0.15
CLEARANCE_RECOVERY_CONTROL = Control(0.1, 0.0)
PLANNING_MAX_SPEED_MPS = 0.15
PLANNING_BRAKE_CONTROL = Control(-0.4, 0.0)


@dataclass(frozen=True)
class LiveRouteContext:
    compiled_map: CompiledSidecarMap
    projector: GpsProjector
    route_version: int
    start_index: int
    fit_residual_m: float


@dataclass(frozen=True)
class RuntimeInput:
    vessel_state: VesselState
    laser_ranges: tuple[float, ...]
    laser_valid_mask: tuple[bool, ...]
    pose_age_s: float
    scan_age_s: float
    device_age_s: float
    work_model: int
    task_status: int


@dataclass(frozen=True)
class RuntimeDecision:
    reason: str
    control: Optional[Control]
    action: Optional[int]
    mission_index: int
    distance_to_goal_m: float
    advised_heading_deg: float
    safe_mask: tuple[bool, ...]
    completed: bool
    replanned: bool
    maneuver_phase: str = "NORMAL"

    @property
    def stop(self) -> bool:
        return self.control is None


def _route_points(route) -> tuple[object, ...]:
    points = tuple(getattr(route, "points", None) or ())
    if not points:
        raise ValueError("live route contains no points")
    return points


def build_live_route_context(
    route,
    pose,
    *,
    session_id: str,
) -> LiveRouteContext:
    """Bind the live route and ship pose to the approved static sidecar."""

    artifact, artifact_hash = load_sidecar_artifact(SIDECAR_PATH)
    expected_route = artifact["route"]
    points = _route_points(route)
    if str(getattr(route, "id", "")) != expected_route["route_id"]:
        raise ValueError("live route id does not match National_Test")
    if len(points) != len(expected_route["points"]):
        raise ValueError("live route point count does not match National_Test")

    anchors = artifact["gps_anchors"]
    projector = GpsProjector(
        float(anchors["latitude1"]),
        float(anchors["longitude1"]),
    )
    gps_points = tuple(
        (
            float(getattr(point, "lat")),
            float(getattr(point, "lng")),
        )
        for point in points
    )
    unity_points = tuple(
        (
            float(point["unity_position"][0]),
            float(point["unity_position"][2]),
        )
        for point in expected_route["points"]
    )
    enu_points = tuple(
        projector.gps_to_enu(lat, lng) for lat, lng in gps_points
    )
    fitted, residuals = fit_route_converter(unity_points, enu_points)
    max_residual = max(residuals)
    if max_residual > ROUTE_FIT_TOLERANCE_M:
        raise ValueError("live route affine fit exceeds tolerance")
    largest, smallest = fitted.singular_values()
    if not (
        CONVERTER_SCALE_BAND[0]
        <= smallest
        <= largest
        <= CONVERTER_SCALE_BAND[1]
    ):
        raise ValueError("live route converter scale is implausible")

    compiled = compile_offline_national_map(session_id=session_id)
    approved = AffineTransform2D(
        *json.loads(
            (
                SIDECAR_PATH.parent
                / "national_test_live_profile.json"
            ).read_text(encoding="utf-8")
        )["fitted_affine"]
    )
    approved_residual = max(
        math.hypot(
            approved.unity_to_enu(ux, uz)[0] - ex,
            approved.unity_to_enu(ux, uz)[1] - ey,
        )
        for (ux, uz), (ex, ey) in zip(unity_points, enu_points)
    )
    if approved_residual > APPROVED_TRANSFORM_TOLERANCE_M:
        raise ValueError("live route differs from the approved affine profile")
    if compiled.manifest.source_artifact_hash != artifact_hash:
        raise ValueError("compiled map and sidecar artifact hash differ")

    lat = float(getattr(pose, "lat", 0.0) or 0.0)
    lng = float(getattr(pose, "lng", 0.0) or 0.0)
    if abs(lat) < 1e-9 or abs(lng) < 1e-9:
        raise ValueError("live ship pose is unavailable")
    ship_enu = projector.gps_to_enu(lat, lng)
    unity_x, unity_z = fitted.enu_to_unity(*ship_enu)
    if not unity_point_in_water(artifact, unity_x, unity_z):
        raise ValueError("live ship pose does not lie in extracted water")

    return LiveRouteContext(
        compiled_map=compiled,
        projector=projector,
        route_version=int(getattr(route, "version", 0) or 0),
        start_index=int(getattr(route, "start_index", 0) or 0),
        fit_residual_m=max_residual,
    )


def _load_compatible_policy(
    checkpoint_path: Path,
    context: LiveRouteContext,
    *,
    require_live: bool,
) -> RecurrentDiscreteSAC:
    """Hash-check one v10 policy at its requested promotion level."""

    checkpoint = Path(checkpoint_path)
    manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("SAC checkpoint or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != (
        "national-test-sac-checkpoint-v4"
    ):
        raise ValueError("SAC checkpoint manifest schema is incompatible")
    if (
        manifest.get("offline_ready") is not True
        or (
            require_live
            and (
                manifest.get("live_ready") is not True
                or not isinstance(
                    manifest.get("unity_validation_log_hashes"),
                    list,
                )
                or len(manifest["unity_validation_log_hashes"]) < 3
            )
        )
    ):
        raise ValueError(
            "SAC checkpoint has not passed offline and Unity promotion"
        )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != manifest.get("checkpoint_sha256"):
        raise ValueError("SAC checkpoint hash does not match its manifest")

    compiled = context.compiled_map
    if not str(manifest.get("dynamics_version", "")).startswith(
        "national-test-forward-calibrated-"
    ):
        raise ValueError("SAC checkpoint dynamics_version is incompatible")
    try:
        profile = forward_control_profile_from_dict(
            manifest["forward_control_profile"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "SAC checkpoint forward control profile is invalid"
        ) from exc
    try:
        reverse_profile = reverse_control_profile_from_dict(
            manifest["reverse_control_profile"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "SAC checkpoint reverse control profile is invalid"
        ) from exc
    reduced_dynamics = enable_reverse_dynamics(
        reduced_dynamics_from_profile(profile),
        reverse_profile,
    )
    expected = {
        "route_id": compiled.manifest.route_id,
        "map_source_artifact_hash": (
            compiled.snapshot.source_artifact_hash
        ),
        "map_payload_hash": compiled.snapshot.payload_content_hash,
        "observation_schema": "local-observation-v2-reduced",
        "observation_dim": 162,
        "action_schema": "five-discrete-forward-bias-v2",
        "action_dim": 5,
        "dynamics_version": reduced_dynamics.version,
        "route_guidance_version": ROUTE_GUIDANCE_VERSION,
        "route_guidance_hash": fixed_route_guidance_hash(compiled),
        "geometry_version": compiled.snapshot.geometry_version,
        "policy_gate_version": MINIMUM_INTERVENTION_GATE_VERSION,
        "prediction_horizon_policy_version": (
            PREDICTION_HORIZON_POLICY_VERSION
        ),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"SAC checkpoint {key} is incompatible")
    hidden_dim = int(manifest.get("hidden_dim", 0))
    if hidden_dim <= 0:
        raise ValueError("SAC checkpoint hidden dimension is missing")
    action_controls = tuple(profile.action_controls)
    if manifest.get("calibration_hash") != profile.calibration_hash:
        raise ValueError("SAC checkpoint calibration hash is invalid")
    if manifest.get("action_controls") != [
        {
            "throttle": control.throttle,
            "rudder": control.rudder,
        }
        for control in action_controls
    ]:
        raise ValueError("SAC checkpoint action controls are invalid")
    if action_protocol_hash(profile) != manifest.get(
        "action_protocol_hash"
    ):
        raise ValueError("SAC checkpoint action protocol hash is invalid")
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=hidden_dim,
        seed=31,
        observation_schema=expected["observation_schema"],
    )
    policy.load_checkpoint(checkpoint)
    policy.forward_control_profile = profile
    policy.reverse_control_profile = reverse_profile
    policy.reduced_dynamics = reduced_dynamics
    return policy


def load_live_ready_policy(
    checkpoint_path: Path,
    context: LiveRouteContext,
) -> RecurrentDiscreteSAC:
    return _load_compatible_policy(
        checkpoint_path,
        context,
        require_live=True,
    )


def load_offline_ready_policy(
    checkpoint_path: Path,
    context: LiveRouteContext,
) -> RecurrentDiscreteSAC:
    """Restricted candidate loader; never used by the sample entrypoint."""

    return _load_compatible_policy(
        checkpoint_path,
        context,
        require_live=False,
    )


class LiveInputAdapter:
    """Convert atomically replaced sample GlobalData objects into fresh input."""

    def __init__(self, global_data, context: LiveRouteContext) -> None:
        self._data = global_data
        self._context = context
        self._started = time.monotonic()
        self._pose_object = None
        self._laser_object = None
        self._device_object = None
        self._pose_changed = 0.0
        self._laser_changed = 0.0
        self._device_changed = 0.0

    @staticmethod
    def _age(changed: float, now: float) -> float:
        return float("inf") if changed <= 0.0 else now - changed

    def build(self) -> RuntimeInput:
        now = time.monotonic()
        scada = self._data.scada_data
        laser = self._data.laser_data
        device = self._data.device_data
        if scada is not self._pose_object:
            self._pose_object = scada
            self._pose_changed = now
        if laser is not self._laser_object:
            self._laser_object = laser
            self._laser_changed = now
        if device is not self._device_object:
            self._device_object = device
            self._device_changed = now

        pose = getattr(scada, "pose", None)
        if pose is None:
            state = VesselState(
                x=float("nan"),
                y=float("nan"),
                yaw=float("nan"),
                speed=float("nan"),
                yaw_rate=float("nan"),
                stamp_sim=now - self._started,
                health="no-pose",
            )
        else:
            x_enu, y_enu = self._context.projector.gps_to_enu(
                float(getattr(pose, "lat", 0.0) or 0.0),
                float(getattr(pose, "lng", 0.0) or 0.0),
            )
            x, y = enu_to_grid(
                self._context.compiled_map.manifest,
                x_enu,
                y_enu,
            )
            state = VesselState(
                x=x,
                y=y,
                yaw=compass_yaw_deg_to_math_yaw_rad(
                    float(getattr(pose, "yaw", 0.0) or 0.0)
                ),
                speed=float(getattr(pose, "speed", 0.0) or 0.0),
                yaw_rate=compass_yaw_rate_degs_to_math_rad_s(
                    float(
                        getattr(pose, "rotate_speed", 0.0) or 0.0
                    )
                ),
                throttle_state=max(
                    -1.0,
                    min(
                        1.0,
                        float(
                            getattr(
                                device,
                                "throttle_percent",
                                0.0,
                            )
                            or 0.0
                        )
                        / 100.0,
                    ),
                ),
                rudder_state=max(
                    -1.0,
                    min(
                        1.0,
                        float(
                            getattr(device, "rudder_percent", 0.0)
                            or 0.0
                        )
                        / 100.0,
                    ),
                ),
                stamp_sim=now - self._started,
            )
        ranges, valid = front_arc_laser_features(
            getattr(laser, "ranges", ()) or ()
        )
        return RuntimeInput(
            vessel_state=state,
            laser_ranges=ranges,
            laser_valid_mask=valid,
            pose_age_s=self._age(self._pose_changed, now),
            scan_age_s=self._age(self._laser_changed, now),
            device_age_s=self._age(self._device_changed, now),
            work_model=int(getattr(device, "work_model", 0) or 0),
            task_status=int(getattr(device, "task_status", 0) or 0),
        )


class FixedMapControllerCore:
    """One deterministic policy/safety/planner step with no ROS writes."""

    def __init__(
        self,
        context: LiveRouteContext,
        policy: RecurrentDiscreteSAC,
        *,
        dynamics: Optional[PrototypeReducedDynamics] = None,
    ) -> None:
        self.context = context
        self.policy = policy
        self.dynamics = (
            dynamics
            or getattr(policy, "reduced_dynamics", None)
            or PrototypeReducedDynamics()
        )
        profile = getattr(policy, "forward_control_profile", None)
        self.forward_profile = (
            profile if isinstance(profile, ForwardControlProfile) else None
        )
        reverse_profile = getattr(
            policy,
            "reverse_control_profile",
            None,
        )
        self.reverse_profile = (
            reverse_profile
            if isinstance(reverse_profile, ReverseControlProfile)
            else None
        )
        self.planning_controls = (
            ()
            if self.forward_profile is None
            else (
                self.forward_profile.action_controls
                if self.reverse_profile is None
                else (
                    *self.forward_profile.action_controls,
                    self.reverse_profile.control,
                )
            )
        )
        if isinstance(profile, ForwardControlProfile):
            self.generator = CandidateControlGenerator(
                max_throttle=max(
                    control.throttle
                    for control in profile.action_controls
                ),
                max_abs_rudder=max(
                    abs(control.rudder)
                    for control in profile.action_controls
                ),
                action_controls=profile.action_controls,
            )
        else:
            self.generator = CandidateControlGenerator()
        self.supervisor = PredictiveSafetySupervisor(
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
            max_state_age_s=1.0,
        )
        point_count = len(
            self.context.compiled_map.manifest.route_points_enu
        )
        self.mission_index = max(
            0,
            min(context.start_index, point_count - 1),
        )
        self.trajectory: Optional[Trajectory] = None
        self.trajectory_index = 0
        self.hidden: Optional[RecurrentHiddenState] = None
        self.hidden_reset = True
        self.maneuver_phase = "NORMAL"
        self.planning_hold_pending = False

    def _stop(
        self,
        reason: str,
        state: VesselState,
        *,
        completed: bool = False,
    ) -> RuntimeDecision:
        return RuntimeDecision(
            reason=reason,
            control=None,
            action=None,
            mission_index=self.mission_index,
            distance_to_goal_m=self._distance(state),
            advised_heading_deg=math_yaw_rad_to_compass_deg(state.yaw),
            safe_mask=(False,) * 5,
            completed=completed,
            replanned=False,
            maneuver_phase=self.maneuver_phase,
        )

    def _goal_xy(self) -> tuple[float, float]:
        manifest = self.context.compiled_map.manifest
        return fixed_route_goal_xy(
            manifest,
            self.mission_index,
        )

    def _distance(self, state: VesselState) -> float:
        if not state.is_finite():
            return 0.0
        goal_x, goal_y = self._goal_xy()
        return math.hypot(state.x - goal_x, state.y - goal_y)

    def _narrow_ingress_control(
        self,
        state: VesselState,
    ) -> Control:
        gate_x, gate_y, _ = fixed_route_gate_region(
            self.context.compiled_map,
            NARROW_ROUTE_INDEX,
        )
        desired_yaw = math.atan2(
            gate_y - state.y,
            gate_x - state.x,
        )
        heading_error = (
            desired_yaw - state.yaw + math.pi
        ) % (2.0 * math.pi) - math.pi
        return narrow_ingress_control(
            throttle=self.forward_profile.minimum_steerage_throttle,
            heading_error=heading_error,
            rudder_yaw_sign=self.dynamics.rudder_yaw_sign,
        )

    def _planning_transition_decision(
        self,
        state: VesselState,
        *,
        hold_reason: str = "PLANNING_HOLD",
    ) -> RuntimeDecision:
        if state.speed <= PLANNING_MAX_SPEED_MPS:
            self.planning_hold_pending = False
            return self._stop(hold_reason, state)
        candidates = tuple(
            CandidateControl(
                action=index,
                control=PLANNING_BRAKE_CONTROL,
            )
            for index in range(5)
        )
        future = braking_future_controls(PLANNING_BRAKE_CONTROL)
        mask, reasons, clearances = self.supervisor.precheck(
            state,
            candidates,
            self.context.compiled_map.snapshot,
            self.dynamics,
            now_sim=state.stamp_sim,
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
            candidate_prefix_s=0.3,
            nominal_future_controls=future,
        )
        if not any(mask):
            return self._stop("PLANNING_BRAKE_UNSAFE", state)
        decision = self.supervisor.finalize(
            policy_action=2,
            nominal_action=2,
            candidate_mask=mask,
            candidates=candidates,
            snapshot_id=self.context.compiled_map.snapshot.snapshot_id,
            current_snapshot_id=(
                self.context.compiled_map.snapshot.snapshot_id
            ),
            reasons=reasons,
            clearances=clearances,
            current_state=state,
            current_map_snapshot=self.context.compiled_map.snapshot,
            dynamics=self.dynamics,
            now_sim=state.stamp_sim,
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
            candidate_prefix_s=0.3,
            nominal_future_controls=future,
        )
        if decision.stop or decision.final_action is None:
            return self._stop("PLANNING_BRAKE_UNSAFE", state)
        return RuntimeDecision(
            reason="PLANNING_BRAKE",
            control=decision.control,
            action=decision.final_action,
            mission_index=self.mission_index,
            distance_to_goal_m=self._distance(state),
            advised_heading_deg=math_yaw_rad_to_compass_deg(state.yaw),
            safe_mask=decision.candidate_mask,
            completed=False,
            replanned=False,
            maneuver_phase=self.maneuver_phase,
        )

    def _clearance_recovery_decision(
        self,
        sample: RuntimeInput,
    ) -> Optional[RuntimeDecision]:
        state = sample.vessel_state
        valid_ranges = tuple(
            value
            for value, valid in zip(
                sample.laser_ranges,
                sample.laser_valid_mask,
            )
            if valid
        )
        if (
            not valid_ranges
            or min(valid_ranges) <= LASER_EMERGENCY_DISTANCE_M
            or abs(state.speed) > CLEARANCE_RECOVERY_MAX_SPEED_MPS
        ):
            return None
        control = (
            Control(
                self.forward_profile.minimum_steerage_throttle,
                0.0,
            )
            if self.forward_profile is not None
            else CLEARANCE_RECOVERY_CONTROL
        )
        if not self.supervisor.clearance_recovery_is_safe(
            state,
            control,
            self.context.compiled_map.snapshot,
            self.dynamics,
            now_sim=state.stamp_sim,
            minimum_clearance_m=CLEARANCE_RECOVERY_MINIMUM_M,
        ):
            return None
        return RuntimeDecision(
            reason="CLEARANCE_RECOVERY",
            control=control,
            action=2,
            mission_index=self.mission_index,
            distance_to_goal_m=self._distance(state),
            advised_heading_deg=math_yaw_rad_to_compass_deg(state.yaw),
            safe_mask=(False, False, True, False, False),
            completed=False,
            replanned=False,
            maneuver_phase=self.maneuver_phase,
        )

    def _gate_distance(self, state: VesselState) -> float:
        gate_x, gate_y, _ = fixed_route_gate_region(
            self.context.compiled_map,
            self.mission_index,
        )
        return math.hypot(state.x - gate_x, state.y - gate_y)

    def _advance_reached_goals(self, state: VesselState) -> bool:
        points = self.context.compiled_map.manifest.route_points_enu
        if self.maneuver_phase == "ESCAPE_PENDING":
            return False
        while True:
            gate_x, gate_y, gate_tolerance = fixed_route_gate_region(
                self.context.compiled_map,
                self.mission_index,
            )
            reached_published_point = fixed_route_waypoint_reached(
                self.context.compiled_map,
                self.mission_index,
                state,
            )
            if not reached_published_point:
                break
            if (
                self.mission_index != NARROW_ROUTE_INDEX
                and math.hypot(
                    state.x - gate_x,
                    state.y - gate_y,
                )
                > gate_tolerance + 1e-9
            ):
                break
            if self.mission_index >= len(points) - 1:
                return True
            reached_index = self.mission_index
            self.mission_index += 1
            if reached_index == NARROW_ROUTE_INDEX:
                self.maneuver_phase = "ESCAPE_PENDING"
                return False
            self.trajectory = None
            self.trajectory_index = 0
            self.hidden = None
            self.hidden_reset = True
            self.planning_hold_pending = True
        return False

    def _complete_escape_if_reached(self, state: VesselState) -> None:
        if (
            self.maneuver_phase == "ESCAPE_PENDING"
            and narrow_escape_released(
                self.context.compiled_map,
                state,
            )
        ):
            self.maneuver_phase = "NORMAL"
            self.trajectory = None
            self.trajectory_index = 0
            self.planning_hold_pending = True

    def step(self, sample: RuntimeInput) -> RuntimeDecision:
        state = sample.vessel_state
        # The running competition build reports 1 while the start request is
        # being handled and 2 once training is active.  Only zero is inactive.
        if sample.task_status == 0:
            return self._stop("TASK_INACTIVE", state)
        if sample.work_model != 2:
            return self._stop("NOT_IN_AUTO_MODE", state)
        if sample.pose_age_s > POSE_MAX_AGE_S:
            return self._stop("POSE_STALE", state)
        if sample.scan_age_s > SCAN_MAX_AGE_S:
            return self._stop("SCAN_STALE", state)
        if sample.device_age_s > DEVICE_MAX_AGE_S:
            return self._stop("DEVICE_STALE", state)
        if (
            not self.dynamics.is_state_valid(state)
            or not self.context.compiled_map.snapshot.is_state_valid(state)
        ):
            recovery = self._clearance_recovery_decision(sample)
            if recovery is not None:
                return recovery
            return self._stop("STATE_INVALID", state)
        if any(
            valid and value <= LASER_EMERGENCY_DISTANCE_M
            for value, valid in zip(
                sample.laser_ranges,
                sample.laser_valid_mask,
            )
        ):
            return self._stop("LASER_EMERGENCY_STOP", state)
        self._complete_escape_if_reached(state)
        if self._advance_reached_goals(state):
            return self._stop("MISSION_DONE", state, completed=True)
        if self.trajectory is None and self.planning_hold_pending:
            return self._planning_transition_decision(state)

        replanned = False
        if self.trajectory is None:
            planning_index = (
                NARROW_ROUTE_INDEX
                if self.maneuver_phase == "ESCAPE_PENDING"
                else self.mission_index
            )
            try:
                self.trajectory = plan_fixed_leg(
                    self.context.compiled_map,
                    start_state=state,
                    mission_index=planning_index,
                    dynamics=self.dynamics,
                    forward_action_controls=self.planning_controls,
                    narrow_visit_completed=(
                        self.maneuver_phase == "ESCAPE_PENDING"
                    ),
                )
            except RuntimeError:
                return self._stop("PLANNING_DEFERRED", state)
            self.trajectory_index = 0
            replanned = True
        preview = preview_trajectory(
            state,
            self.trajectory,
            self.trajectory_index,
            allow_reverse_branch_progress=(
                self.maneuver_phase == "ESCAPE_PENDING"
            ),
        )
        if (
            self.maneuver_phase != "ESCAPE_PENDING"
            and (
                preview.cross_track_error_m > 0.8
                or (
                preview.state_index
                >= len(self.trajectory.states) - 2
                and self._gate_distance(state)
                > fixed_route_gate_region(
                    self.context.compiled_map,
                    self.mission_index,
                )[2]
                )
            )
        ):
            self.trajectory = None
            self.trajectory_index = 0
            self.hidden = None
            self.hidden_reset = True
            self.planning_hold_pending = True
            return self._planning_transition_decision(
                state,
                hold_reason="REPLANNING_HOLD",
            )
        self.trajectory_index = preview.state_index
        planned_nominal = self.trajectory.controls[
            preview.nominal_control_index
        ]
        ingress_recovery = (
            planned_nominal.throttle < 0.0
            and self.maneuver_phase != "ESCAPE_PENDING"
            and self.mission_index == NARROW_ROUTE_INDEX
        )
        if ingress_recovery:
            nominal = self._narrow_ingress_control(state)
        elif planned_nominal.throttle < 0.0:
            nominal = reverse_tracking_control(
                preview,
                planned_nominal,
                self.dynamics,
                yaw_rate=state.yaw_rate,
            )
        else:
            nominal = feedback_tracking_control(
                preview,
                planned_nominal,
                self.dynamics,
                yaw_rate=state.yaw_rate,
                speed=state.speed,
                clearance_m=(
                    self.context.compiled_map.snapshot.clearance_at(
                        state
                    )
                ),
            )
        remaining_horizon = (
            FIXED_MAP_PREDICTION_HORIZON_S - 0.3
        )
        skip = 0.3
        nominal_future_controls = []
        for control, duration in zip(
            self.trajectory.controls[preview.nominal_control_index :],
            self.trajectory.durations[preview.nominal_control_index :],
        ):
            if remaining_horizon <= 1e-12:
                break
            available = float(duration)
            if skip > 1e-12:
                removed = min(skip, available)
                available -= removed
                skip -= removed
            if available <= 1e-12:
                continue
            applied = min(available, remaining_horizon)
            nominal_future_controls.append((control, applied))
            remaining_horizon -= applied
        if remaining_horizon > 1e-12:
            nominal_future_controls.append(
                (self.trajectory.controls[-1], remaining_horizon)
            )
        reverse_nominal = nominal.throttle < 0.0
        overspeed_braking = (
            planned_nominal.throttle >= 0.0
            and reverse_nominal
            and not ingress_recovery
        )
        nominal_future_controls = (
            narrow_ingress_future_controls(
                nominal,
                tuple(nominal_future_controls),
            )
            if ingress_recovery
            else braking_future_controls(nominal)
            if overspeed_braking
            else tracking_future_controls(
                nominal,
                tuple(nominal_future_controls),
            )
            if reverse_nominal
            else tracking_future_controls(
                nominal,
                tuple(nominal_future_controls),
            )
        )
        candidates = (
            tuple(
                CandidateControl(action=index, control=nominal)
                for index in range(5)
            )
            if reverse_nominal or ingress_recovery
            else self.generator.generate(
                nominal.throttle,
                nominal.rudder,
            )
        )
        if reverse_nominal and not ingress_recovery:
            (
                safe_mask,
                reasons,
                clearances,
                safety_horizon,
            ) = self.supervisor.precheck_with_horizon_fallback(
                state,
                candidates,
                self.context.compiled_map.snapshot,
                self.dynamics,
                now_sim=state.stamp_sim,
                candidate_prefix_s=0.3,
                nominal_future_controls=nominal_future_controls,
            )
        elif ingress_recovery:
            (
                safe_mask,
                reasons,
                clearances,
            ) = self.supervisor.precheck(
                state,
                candidates,
                self.context.compiled_map.snapshot,
                self.dynamics,
                now_sim=state.stamp_sim,
                prediction_horizon_s=(
                    FIXED_MAP_PREDICTION_HORIZON_S
                ),
                candidate_prefix_s=0.3,
                nominal_future_controls=nominal_future_controls,
            )
            safety_horizon = FIXED_MAP_PREDICTION_HORIZON_S
        else:
            (
                safe_mask,
                reasons,
                clearances,
                safety_horizon,
            ) = self.supervisor.precheck_with_horizon_fallback(
                state,
                candidates,
                self.context.compiled_map.snapshot,
                self.dynamics,
                now_sim=state.stamp_sim,
                candidate_prefix_s=0.3,
                nominal_future_controls=nominal_future_controls,
            )
        if not any(safe_mask):
            return self._stop("NO_SAFE_ACTION", state)
        if reverse_nominal or ingress_recovery:
            policy_action = 2
            next_hidden = self.hidden
        else:
            observation = build_fixed_map_observation(
                state=state,
                preview=preview,
                safe_mask=safe_mask,
                session_id=self.context.compiled_map.snapshot.session_id,
                laser_ranges=sample.laser_ranges,
                laser_valid_mask=sample.laser_valid_mask,
                scan_age_s=sample.scan_age_s,
                pose_age_s=sample.pose_age_s,
                hidden_reset=self.hidden_reset,
            )
            proposal, next_hidden = self.policy.act(
                observation,
                safe_mask,
                hidden=self.hidden,
                deterministic=True,
            )
            policy_action = minimum_intervention_action(
                policy_action=proposal.action,
                safe_action_mask=safe_mask,
                candidates=candidates,
                nominal_control=candidates[2].control,
            )
        decision = self.supervisor.finalize(
            policy_action=policy_action,
            nominal_action=2,
            candidate_mask=safe_mask,
            candidates=candidates,
            snapshot_id=self.context.compiled_map.snapshot.snapshot_id,
            current_snapshot_id=(
                self.context.compiled_map.snapshot.snapshot_id
            ),
            reasons=reasons,
            clearances=clearances,
            current_state=state,
            current_map_snapshot=self.context.compiled_map.snapshot,
            dynamics=self.dynamics,
            now_sim=state.stamp_sim,
            prediction_horizon_s=safety_horizon,
            candidate_prefix_s=0.3,
            nominal_future_controls=nominal_future_controls,
        )
        if decision.stop or decision.final_action is None:
            return self._stop(decision.reason, state)
        if not reverse_nominal and not ingress_recovery:
            self.hidden = next_hidden
            self.hidden_reset = False
        desired_yaw = state.yaw + preview.heading_error
        return RuntimeDecision(
            reason=(
                "REVERSE_ESCAPE_NOMINAL"
                if reverse_nominal and not overspeed_braking
                else "OVERSPEED_REVERSE_BRAKE"
                if overspeed_braking
                else "NARROW_INGRESS_RECOVERY"
                if ingress_recovery
                else decision.reason
            ),
            control=decision.control,
            action=decision.final_action,
            mission_index=self.mission_index,
            distance_to_goal_m=self._distance(state),
            advised_heading_deg=math_yaw_rad_to_compass_deg(desired_yaw),
            safe_mask=decision.candidate_mask,
            completed=False,
            replanned=replanned,
            maneuver_phase=self.maneuver_phase,
        )


__all__ = [
    "DEFAULT_CHECKPOINT",
    "FixedMapControllerCore",
    "LiveInputAdapter",
    "LiveRouteContext",
    "RuntimeDecision",
    "RuntimeInput",
    "build_live_route_context",
    "load_live_ready_policy",
    "load_offline_ready_policy",
]
