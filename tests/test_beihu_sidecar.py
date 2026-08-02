"""Offline regression for the Beihu build-bound sidecar map pipeline.

Pure Python: no ROS, Unity or MATLAB.  These tests pin the compiler's
determinism, the coordinate transforms, the conservative distance field and
the planner's goal-relocation behaviour on the real extracted artifact.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from usvlib4ros.mapping import (
    GpsProjector,
    SidecarCompilerConfig,
    compile_beihu_sidecar,
    enu_to_grid,
    fit_affine_unity_to_enu,
    fit_unity_to_enu,
    load_sidecar_artifact,
    verify_route_points,
)
from usvlib4ros.planning import (
    CostConfig,
    GoalRegion,
    KinodynamicInformedRRTStarPlanner,
    PlannerConfig,
    PlanningRequest,
    PrototypeReducedDynamics,
    VesselState,
)

ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "usvlib4ros"
    / "mapping"
    / "data"
    / "beihu_static_world_sidecar.json"
)
PROMOTION = "operator-authorization:offline-test-fixture"


@pytest.fixture(scope="module")
def artifact():
    return load_sidecar_artifact(ARTIFACT_PATH)


@pytest.fixture(scope="module")
def compiled(artifact):
    data, artifact_hash = artifact
    config = SidecarCompilerConfig(
        transform_model="similarity",
        coverage_status="complete_prior",
        promotion_note=PROMOTION,
    )
    return compile_beihu_sidecar(
        data,
        source_artifact_hash=artifact_hash,
        session_id="test-session",
        stamp_sim=0.0,
        config=config,
    )


def test_compile_is_deterministic(artifact):
    data, artifact_hash = artifact
    first = compile_beihu_sidecar(data, source_artifact_hash=artifact_hash, session_id="s")
    second = compile_beihu_sidecar(data, source_artifact_hash=artifact_hash, session_id="s")
    assert first.snapshot.payload_content_hash == second.snapshot.payload_content_hash
    assert first.snapshot.rows == second.snapshot.rows


def test_manifest_matches_route_evidence(compiled):
    manifest = compiled.manifest
    assert manifest.route_version == 46
    assert len(manifest.route_points_enu) == 13
    assert len(manifest.buoys) == 16
    assert manifest.water_cells > 50_000
    scales = sorted(b.radius_m for b in manifest.buoys)
    assert scales[-1] > 2.0 * scales[0]  # the single 3x buoy


def test_compiled_buoy_clearance_uses_exact_circle_geometry(compiled):
    snapshot = compiled.snapshot
    manifest = compiled.manifest
    buoy = manifest.buoys[0]
    x, y = enu_to_grid(manifest, buoy.x, buoy.y)
    state = VesselState(
        x=x,
        y=y,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=snapshot.stamp_sim,
    )

    assert len(snapshot.circular_obstacles) == len(manifest.buoys)
    assert snapshot.clearance_at(state) == pytest.approx(
        -buoy.radius_m - snapshot.footprint_radius
    )


def test_default_national_map_resolution_is_point_two_metres(compiled):
    assert compiled.snapshot.resolution == pytest.approx(0.2)
    assert compiled.manifest.resolution_m == pytest.approx(0.2)


def test_candidate_coverage_is_the_default(artifact):
    data, artifact_hash = artifact
    candidate = compile_beihu_sidecar(data, source_artifact_hash=artifact_hash, session_id="s")
    assert candidate.snapshot.coverage_status == "candidate_complete_prior"
    # Promotion without an operator authorization note is rejected.
    assert not SidecarCompilerConfig(coverage_status="complete_prior").is_valid()


def test_transform_models_reconstruct_anchor(artifact):
    data, _ = artifact
    anchors = data["gps_anchors"]
    projector = GpsProjector(anchors["latitude1"], anchors["longitude1"])
    for transform in (
        fit_affine_unity_to_enu(anchors, projector),
        fit_unity_to_enu(anchors, projector, reflected=False),
        fit_unity_to_enu(anchors, projector, reflected=True),
    ):
        x, y = transform.unity_to_enu(anchors["unity_x1"], anchors["unity_z1"])
        assert math.hypot(x, y) < 1e-6  # anchor 1 is the ENU origin
        ux, uz = transform.enu_to_unity(x, y)
        assert math.hypot(ux - anchors["unity_x1"], uz - anchors["unity_z1"]) < 1e-6


def test_route_verification_detects_misaligned_points(artifact):
    data, _ = artifact
    anchors = data["gps_anchors"]
    projector = GpsProjector(anchors["latitude1"], anchors["longitude1"])
    transform = fit_unity_to_enu(anchors, projector, reflected=False)
    unity_points = [(p["unity_position"][0], p["unity_position"][2]) for p in data["route"]["points"]]
    # Synthesize GPS points from the same transform: residuals must be ~0.
    gps_points = [
        projector.enu_to_gps(*transform.unity_to_enu(ux, uz)) for ux, uz in unity_points
    ]
    ok = verify_route_points(transform, unity_points, gps_points, projector, tolerance_m=0.5)
    assert ok.passed and ok.max_residual_m < 1e-6
    # A 2 m shift must fail the same gate.
    shifted = [(ux + 2.0, uz) for ux, uz in unity_points]
    bad = verify_route_points(transform, shifted, gps_points, projector, tolerance_m=0.5)
    assert not bad.passed and bad.max_residual_m > 1.0


def test_route_verification_supports_affine_model(artifact):
    """The axis-affine candidate has no ``reflected`` field; the gate must still
    accept it (live entry iterates all three candidate models)."""

    data, _ = artifact
    anchors = data["gps_anchors"]
    projector = GpsProjector(anchors["latitude1"], anchors["longitude1"])
    transform = fit_affine_unity_to_enu(anchors, projector)
    unity_points = [(p["unity_position"][0], p["unity_position"][2]) for p in data["route"]["points"]]
    gps_points = [
        projector.enu_to_gps(*transform.unity_to_enu(ux, uz)) for ux, uz in unity_points
    ]
    ok = verify_route_points(transform, unity_points, gps_points, projector, tolerance_m=0.5)
    assert ok.passed and ok.max_residual_m < 1e-6 and ok.reflected is False


def test_fit_route_converter_recovers_transform():
    """The startup converter fit must recover a known affine map exactly."""

    from usvlib4ros.mapping import AffineTransform2D, fit_route_converter

    truth = AffineTransform2D(a=0.7, b=-0.05, c=0.02, d=1.01, tx=-350.0, ty=120.0)
    unity_points = [(-420.0 + 37.0 * (i % 4), 121.0 - 53.0 * (i // 4) + 11.0 * (i % 3)) for i in range(13)]
    enu_points = [truth.unity_to_enu(ux, uz) for ux, uz in unity_points]
    fitted, residuals = fit_route_converter(unity_points, enu_points)
    assert max(residuals) < 1e-6
    for got, want in zip(fitted.coefficients(), truth.coefficients()):
        assert got == pytest.approx(want, abs=1e-6)
    # Inverse roundtrip and scale band.
    ux, uz = fitted.enu_to_unity(*truth.unity_to_enu(-420.0, 121.0))
    assert (ux, uz) == pytest.approx((-420.0, 121.0), abs=1e-6)
    sv_max, sv_min = fitted.singular_values()
    assert 0.6 < sv_min < sv_max < 1.2
    assert fitted.reflected is False
    # Degenerate input is rejected.
    with pytest.raises(ValueError):
        fit_route_converter(unity_points[:2], enu_points[:2])


def test_route_fitted_affine_compile(artifact):
    """The compiler must accept the route-fitted model and bind it into the hash."""

    from usvlib4ros.mapping import AffineTransform2D, fit_route_converter

    data, artifact_hash = artifact
    truth = AffineTransform2D(a=0.77, b=0.01, c=-0.02, d=1.0, tx=-150.0, ty=40.0)
    unity_points = [(p["unity_position"][0], p["unity_position"][2]) for p in data["route"]["points"]]
    enu_points = [truth.unity_to_enu(ux, uz) for ux, uz in unity_points]
    fitted, _ = fit_route_converter(unity_points, enu_points)
    config = SidecarCompilerConfig(
        transform_model="route_fitted_affine",
        fitted_affine=fitted.coefficients(),
    )
    assert config.is_valid()
    first = compile_beihu_sidecar(data, source_artifact_hash=artifact_hash, session_id="s", config=config)
    second = compile_beihu_sidecar(data, source_artifact_hash=artifact_hash, session_id="s", config=config)
    assert first.snapshot.payload_content_hash == second.snapshot.payload_content_hash
    assert first.manifest.transform_model == "route_fitted_affine"
    assert first.manifest.water_cells > 50_000
    # The fitted coefficients enter the config hash: different fit -> different hash.
    other = SidecarCompilerConfig(
        transform_model="route_fitted_affine",
        fitted_affine=truth.coefficients(),
    )
    assert other.config_hash() != config.config_hash()
    # Missing coefficients are rejected.
    assert not SidecarCompilerConfig(transform_model="route_fitted_affine").is_valid()


def test_unity_point_in_water(artifact):
    """The independent ship gate geometry: route points in, far points out."""

    from usvlib4ros.mapping import unity_point_in_water

    data, _ = artifact
    wp0 = data["route"]["points"][0]["unity_position"]
    assert unity_point_in_water(data, float(wp0[0]), float(wp0[2]))
    assert not unity_point_in_water(data, float(wp0[0]) + 1000.0, float(wp0[2]) + 1000.0)


def test_distance_field_is_conservative(compiled):
    snapshot = compiled.snapshot
    field = snapshot._distance_field()
    half_diagonal = snapshot.resolution * math.sqrt(2.0) / 2.0
    hard = [
        (x, y)
        for y, row in enumerate(snapshot.rows)
        for x, marker in enumerate(row)
        if marker in "#?"
    ]
    import random

    rng = random.Random(7)
    for _ in range(200):
        cx = rng.randrange(snapshot.width)
        cy = rng.randrange(snapshot.height)
        x = (cx + rng.random()) * snapshot.resolution
        y = (cy + rng.random()) * snapshot.resolution
        exact = min(
            math.hypot(x - (hx + 0.5) * snapshot.resolution, y - (hy + 0.5) * snapshot.resolution)
            - half_diagonal
            for hx, hy in hard[:5000]
        )
        lut = field[cy, cx] * snapshot.resolution - half_diagonal
        assert lut <= exact + 1e-9


def test_planner_relocates_goal_inside_tolerance(compiled):
    snapshot = compiled.snapshot
    manifest = compiled.manifest
    dynamics = PrototypeReducedDynamics()
    cost = CostConfig()
    planner = KinodynamicInformedRRTStarPlanner()
    # Aim at the centre of a fixed buoy: occupied at the centre but surrounded
    # by free water inside a 2.5 m tolerance disk.
    buoy = manifest.buoys[0]
    gx, gy = enu_to_grid(manifest, buoy.x, buoy.y)
    start_x, start_y = gx - 4.0, gy
    start = VesselState(x=start_x, y=start_y, yaw=0.0, speed=0.0, yaw_rate=0.0, stamp_sim=0.0)
    if not snapshot.is_state_valid(start):
        pytest.skip("synthetic start not valid on this build")
    goal = GoalRegion(x=gx, y=gy, position_tolerance=2.5, speed_limit=1.2, yaw_rate_limit=1.2)
    request = PlanningRequest(
        request_id="relocate-test",
        session_id=snapshot.session_id,
        start_state=start,
        goal_region=goal,
        map_snapshot_id=snapshot.snapshot_id,
        dynamics_version=dynamics.version,
        cost_config_version=cost.version,
        time_budget_ms=5_000.0,
        seed=31,
        stamp_sim=0.0,
    )
    result = planner.plan(request, snapshot, dynamics, cost, now_sim=0.0)
    assert result.status.value != "GOAL_OCCUPIED"


def test_planner_solves_open_leg_on_real_map(compiled):
    snapshot = compiled.snapshot
    manifest = compiled.manifest
    dynamics = PrototypeReducedDynamics()
    cost = CostConfig()
    planner = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(max_nodes=2000, goal_bias=0.25, global_sample_ratio=0.3, rewire_radius=2.5, connect_tolerance=1.2)
    )
    points = manifest.route_points_enu
    # Leg 8 -> 9 is open water on the evidenced build.
    sx, sy = enu_to_grid(manifest, *points[7])
    gx, gy = enu_to_grid(manifest, *points[8])
    yaw = math.atan2(gy - sy, gx - sx)
    start = VesselState(x=sx, y=sy, yaw=yaw, speed=0.0, yaw_rate=0.0, stamp_sim=0.0)
    if not snapshot.is_state_valid(start):
        pytest.skip("leg-8 start is inside the margin on this build")
    goal = GoalRegion(x=gx, y=gy, position_tolerance=2.5, speed_limit=1.2, yaw_rate_limit=1.2)
    request = PlanningRequest(
        request_id="leg-8-9",
        session_id=snapshot.session_id,
        start_state=start,
        goal_region=goal,
        map_snapshot_id=snapshot.snapshot_id,
        dynamics_version=dynamics.version,
        cost_config_version=cost.version,
        time_budget_ms=12_000.0,
        seed=31,
        stamp_sim=0.0,
    )
    result = planner.plan(request, snapshot, dynamics, cost, now_sim=0.0)
    assert result.trajectory is not None, f"open leg failed: {result.status.value} {result.reason}"
    assert result.trajectory.validation_status == "VALID"
    assert result.trajectory.min_clearance > 0.0
