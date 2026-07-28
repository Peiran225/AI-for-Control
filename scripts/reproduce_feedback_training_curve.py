#!/usr/bin/env python3
"""Re-run one fixed feedback configuration while recording minibatch losses.

The selected feedback experiments predate per-epoch minibatch-loss logging.
This helper reads an immutable saved configuration, changes only the output
directory, and invokes the same training routine.  It is intended for
learning-curve reproduction and refuses to overwrite an existing directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_feedback_section5 import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    saved_args = dict(payload["args"])
    saved_args["out_dir"] = str(out_dir)
    radii = saved_args.get("test_radii", "0.05,0.10,0.20")
    if isinstance(radii, list):
        saved_args["test_radii"] = ",".join(str(value) for value in radii)

    train(Namespace(**saved_args))


if __name__ == "__main__":
    main()
