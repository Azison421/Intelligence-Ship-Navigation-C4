"""Write an offline evidence report for the three approved narrow-point gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.planning import VesselState
from usvlib4ros.planning.fixed_route import (
    NARROW_ROUTE_INDEX,
    NarrowCompositeInfeasibleError,
    compile_offline_national_map,
    fixed_route_goal_xy,
    fixed_route_planning_gate,
    plan_narrow_with_geometry_evidence,
)
from usvlib4ros.planning.forward_control_profile import (
    diagnostic_forward_control_profile,
    forward_control_profile_from_dict,
    reduced_dynamics_from_profile,
)


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "logs"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-log", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    compiled = compile_offline_national_map(
        session_id=f"narrow-forward-audit-{stamp}",
    )
    previous = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX - 1,
    )
    gate = fixed_route_planning_gate(compiled, NARROW_ROUTE_INDEX)
    start = VesselState(
        x=previous[0],
        y=previous[1],
        yaw=math.atan2(gate[1] - previous[1], gate[0] - previous[0]),
        speed=0.3,
        yaw_rate=0.0,
        stamp_sim=compiled.snapshot.stamp_sim,
    )
    calibration_status = "diagnostic_only"
    profile = diagnostic_forward_control_profile()
    if args.calibration_log is not None:
        calibration = json.loads(
            args.calibration_log.read_text(encoding="utf-8")
        )
        if (
            calibration.get("schema_version")
            != "national-test-forward-calibration-v1"
            or calibration.get("verdict") != "calibrated"
        ):
            raise ValueError("forward calibration log is not promotable")
        profile = forward_control_profile_from_dict(
            calibration["profile"]
        )
        calibration_status = "calibrated"
    dynamics = reduced_dynamics_from_profile(profile)
    result: dict[str, object] = {
        "schema_version": "national-test-narrow-feasibility-v1",
        "route_index_zero_based": NARROW_ROUTE_INDEX,
        "per_geometry_time_budget_ms": 5_000.0,
        "forward_only": True,
        "calibration_status": calibration_status,
        "calibration_hash": profile.calibration_hash,
        "contains_sensitive_connection_data": False,
    }
    exit_code = 0
    try:
        selected, trajectory, evidence = plan_narrow_with_geometry_evidence(
            compiled,
            start_state=start,
            dynamics=dynamics,
            time_budget_ms=5_000.0,
            seed=71,
            forward_action_controls=profile.action_controls,
        )
        result.update(
            {
                "verdict": "feasible",
                "selected_geometry_version": (
                    selected.snapshot.geometry_version
                ),
                "trajectory_duration_s": trajectory.times[-1],
                "evidence": [asdict(item) for item in evidence],
            }
        )
    except NarrowCompositeInfeasibleError as exc:
        result.update(
            {
                "verdict": "infeasible_under_approved_gates",
                "evidence": [asdict(item) for item in exc.evidence],
            }
        )
        exit_code = 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / f"narrow-forward-feasibility-{stamp}.json"
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(str(target))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
