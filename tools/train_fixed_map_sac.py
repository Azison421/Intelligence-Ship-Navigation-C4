"""Train the fixed-map recurrent SAC policy without ROS or Unity writes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from usvlib4ros.policy.fixed_map_trainer import FixedMapSACTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "national_test_sac_live_v9.pt"
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
    parser.add_argument("--evaluation-episodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=2)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    trainer = FixedMapSACTrainer(
        seed=args.seed,
        hidden_dim=args.hidden_dim,
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
    print(
        json.dumps(
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
