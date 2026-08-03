"""Promote one offline v10 candidate using three or more Unity run logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.policy.checkpoint_promotion import promote_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("unity_logs", type=Path, nargs="+")
    args = parser.parse_args()
    promoted = promote_checkpoint(args.manifest, args.unity_logs)
    print(json.dumps(promoted, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
