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
from usvlib4ros.planning import (
    Control,
    PrototypeReducedDynamics,
    Trajectory,
    VesselState,
)
from usvlib4ros.planning.fixed_route import (
    SIDECAR_PATH,
    ROUTE_GUIDANCE_VERSION,
    compile_offline_national_map,
    fixed_route_gate_region,
    fixed_route_goal_xy,
    plan_fixed_leg,
)
from usvlib4ros.policy.fixed_map_features import (
    build_fixed_map_observation,
    feedback_tracking_control,
    front_arc_laser_features,
    preview_trajectory,
)
from usvlib4ros.policy.recurrent_sac import (
    RecurrentDiscreteSAC,
    RecurrentHiddenState,
)
from usvlib4ros.policy.safety_supervisor import (
    CandidateControlGenerator,
    FIXED_MAP_PREDICTION_HORIZON_S,
    PredictiveSafetySupervisor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "national_test_sac_live_v9.pt"
)
ROUTE_FIT_TOLERANCE_M = 0.05
APPROVED_TRANSFORM_TOLERANCE_M = 0.05
CONVERTER_SCALE_BAND = (0.5, 2.0)
POSE_MAX_AGE_S = 0.5
SCAN_MAX_AGE_S = 1.0
DEVICE_MAX_AGE_S = 1.0
LASER_EMERGENCY_DISTANCE_M = 0.6


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


def load_live_ready_policy(
    checkpoint_path: Path,
    context: LiveRouteContext,
) -> RecurrentDiscreteSAC:
    """Hash-check and load only an offline-evaluated policy checkpoint."""

    checkpoint = Path(checkpoint_path)
    manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("SAC checkpoint or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != (
        "national-test-sac-checkpoint-v3"
    ):
        raise ValueError("SAC checkpoint manifest schema is incompatible")
    if manifest.get("live_ready") is not True:
        raise ValueError("SAC checkpoint has not passed policy-only evaluation")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != manifest.get("checkpoint_sha256"):
        raise ValueError("SAC checkpoint hash does not match its manifest")

    compiled = context.compiled_map
    expected = {
        "route_id": compiled.manifest.route_id,
        "map_source_artifact_hash": (
            compiled.snapshot.source_artifact_hash
        ),
        "map_payload_hash": compiled.snapshot.payload_content_hash,
        "observation_schema": "local-observation-v2-reduced",
        "observation_dim": 162,
        "action_schema": "five-discrete-rudder-v1",
        "action_dim": 5,
        "dynamics_version": PrototypeReducedDynamics().version,
        "route_guidance_version": ROUTE_GUIDANCE_VERSION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"SAC checkpoint {key} is incompatible")
    hidden_dim = int(manifest.get("hidden_dim", 0))
    if hidden_dim <= 0:
        raise ValueError("SAC checkpoint hidden dimension is missing")
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=hidden_dim,
        seed=31,
        observation_schema=expected["observation_schema"],
    )
    policy.load_checkpoint(checkpoint)
    return policy


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
        self.dynamics = dynamics or PrototypeReducedDynamics()
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

    def _gate_distance(self, state: VesselState) -> float:
        gate_x, gate_y, _ = fixed_route_gate_region(
            self.context.compiled_map,
            self.mission_index,
        )
        return math.hypot(state.x - gate_x, state.y - gate_y)

    def _advance_reached_goals(self, state: VesselState) -> bool:
        points = self.context.compiled_map.manifest.route_points_enu
        while True:
            _, _, gate_tolerance = fixed_route_gate_region(
                self.context.compiled_map,
                self.mission_index,
            )
            if self._gate_distance(state) > gate_tolerance:
                break
            if self.mission_index >= len(points) - 1:
                return True
            self.mission_index += 1
            self.trajectory = None
            self.trajectory_index = 0
            self.hidden = None
            self.hidden_reset = True
        return False

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
            return self._stop("STATE_INVALID", state)
        if any(
            valid and value < LASER_EMERGENCY_DISTANCE_M
            for value, valid in zip(
                sample.laser_ranges,
                sample.laser_valid_mask,
            )
        ):
            return self._stop("LASER_EMERGENCY_STOP", state)
        if self._advance_reached_goals(state):
            return self._stop("MISSION_DONE", state, completed=True)

        replanned = False
        if self.trajectory is None:
            self.trajectory = plan_fixed_leg(
                self.context.compiled_map,
                start_state=state,
                mission_index=self.mission_index,
                dynamics=self.dynamics,
            )
            self.trajectory_index = 0
            replanned = True
        preview = preview_trajectory(
            state,
            self.trajectory,
            self.trajectory_index,
        )
        if (
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
        ):
            self.trajectory = plan_fixed_leg(
                self.context.compiled_map,
                start_state=state,
                mission_index=self.mission_index,
                dynamics=self.dynamics,
            )
            self.trajectory_index = 0
            self.hidden = None
            self.hidden_reset = True
            preview = preview_trajectory(state, self.trajectory, 0)
            replanned = True
        self.trajectory_index = preview.state_index
        nominal = feedback_tracking_control(
            preview,
            self.trajectory.controls[preview.nominal_control_index],
            self.dynamics,
        )
        candidates = self.generator.generate(
            nominal.throttle,
            nominal.rudder,
        )
        safe_mask, reasons, clearances = self.supervisor.precheck(
            state,
            candidates,
            self.context.compiled_map.snapshot,
            self.dynamics,
            now_sim=state.stamp_sim,
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
        )
        if not any(safe_mask):
            return self._stop("NO_SAFE_ACTION", state)
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
        decision = self.supervisor.finalize(
            policy_action=proposal.action,
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
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
        )
        if decision.stop or decision.final_action is None:
            return self._stop(decision.reason, state)
        self.hidden = next_hidden
        self.hidden_reset = False
        desired_yaw = state.yaw + preview.heading_error
        return RuntimeDecision(
            reason=decision.reason,
            control=decision.control,
            action=decision.final_action,
            mission_index=self.mission_index,
            distance_to_goal_m=self._distance(state),
            advised_heading_deg=math_yaw_rad_to_compass_deg(desired_yaw),
            safe_mask=decision.candidate_mask,
            completed=False,
            replanned=replanned,
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
]
