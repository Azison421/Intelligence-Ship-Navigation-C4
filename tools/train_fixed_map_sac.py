"""Train the fixed-map recurrent SAC policy without ROS or Unity writes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.planning.forward_control_profile import (
    forward_control_profile_from_dict,
)
from usvlib4ros.navigation.reverse_control_calibration import (
    build_reverse_control_profile,
)
from usvlib4ros.policy.fixed_map_trainer import FixedMapSACTrainer


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "national_test_sac_live_v10.pt"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train masked recurrent SAC on the verified fixed "
            "北湖/National_Test map."
        )
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--updates-per-episode", type=int, default=512)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=2)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--calibration-log",
        type=Path,
        help="A successful output from calibrate_forward_control.py.",
    )
    parser.add_argument(
        "--reverse-calibration-log",
        type=Path,
        help="A successful output from calibrate_reverse_control.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional non-overwriting JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    forward_profile = None
    reverse_profile = None
    calibration_status = "diagnostic_only"
    reverse_calibration_status = "diagnostic_only"
    if args.calibration_log is not None:
        calibration = json.loads(
            args.calibration_log.read_text(encoding="utf-8")
        )
        if (
            calibration.get("schema_version")
            != "national-test-forward-calibration-v1"
            or calibration.get("verdict") != "calibrated"
            or not isinstance(calibration.get("profile"), dict)
        ):
            raise ValueError("forward calibration log is not promotable")
        forward_profile = forward_control_profile_from_dict(
            calibration["profile"]
        )
        calibration_status = "calibrated"
    if args.reverse_calibration_log is not None:
        reverse_bytes = args.reverse_calibration_log.read_bytes()
        reverse_calibration = json.loads(reverse_bytes.decode("utf-8"))
        evaluation = reverse_calibration.get("evaluation")
        if (
            reverse_calibration.get("schema_version")
            != "national-test-reverse-calibration-v1"
            or reverse_calibration.get("verdict") != "reverse_supported"
            or not isinstance(evaluation, dict)
            or evaluation.get("supported") is not True
        ):
            raise ValueError("reverse calibration log is not promotable")
        reverse_profile = build_reverse_control_profile(
            source_log_sha256=hashlib.sha256(
                reverse_bytes
            ).hexdigest(),
            command_throttle=(
                float(reverse_calibration["command_throttle_percent"])
                / 100.0
            ),
            baseline_signed_speed_mps=float(
                evaluation["baseline_signed_speed_mps"]
            ),
            command_signed_speed_mps=float(
                evaluation["command_signed_speed_mps"]
            ),
        )
        reverse_calibration_status = "calibrated"
    trainer = FixedMapSACTrainer(
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        forward_profile=forward_profile,
        reverse_profile=reverse_profile,
        calibration_status=calibration_status,
        reverse_calibration_status=reverse_calibration_status,
    )
    training, episodes = trainer.train(
        episodes=args.episodes,
        updates_per_episode=args.updates_per_episode,
        batch_size=args.batch_size,
        burn_in=args.burn_in,
        unroll=args.unroll,
    )
    evaluation, evaluation_episodes = trainer.evaluate(
        episodes=args.evaluation_episodes,
    )
    checkpoint, manifest = trainer.save_checkpoint(
        args.output.resolve(),
        training,
        evaluation,
    )
    report = json.dumps(
        {
            "training": asdict(training),
            "episodes": [asdict(episode) for episode in episodes],
            "evaluation": asdict(evaluation),
            "evaluation_episodes": [
                asdict(episode)
                for episode in evaluation_episodes
            ],
            "checkpoint": str(checkpoint),
            "manifest": str(manifest),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.report is not None:
        report_path = args.report.resolve()
        if report_path.exists():
            raise FileExistsError(
                f"refusing to overwrite report: {report_path}"
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
