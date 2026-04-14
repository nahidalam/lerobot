#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare `slobot/aic` for ACT training.

The raw dataset stores the target action across two keys:
`action.tcp.position` and `action.tcp.orientation`.

ACT in LeRobot expects a single `action` feature, so this script creates a local,
ACT-ready LeRobot dataset with those two targets concatenated into one 7D action
vector.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import modify_features
from lerobot.datasets.io_utils import write_stats

DEFAULT_SOURCE_REPO_ID = "slobot/aic"
DEFAULT_PREPARED_REPO_ID = "slobot/aic_act"
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "datasets" / "act_ready_slobot_aic"

ACTION_POSITION_KEY = "action.tcp.position"
ACTION_ORIENTATION_KEY = "action.tcp.orientation"
COMBINED_ACTION_KEY = "action"
COMBINED_ACTION_NAMES = [
    "tcp.position.x",
    "tcp.position.y",
    "tcp.position.z",
    "tcp.orientation.x",
    "tcp.orientation.y",
    "tcp.orientation.z",
    "tcp.orientation.w",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an ACT-ready local copy of the AIC dataset.")
    parser.add_argument(
        "--source-repo-id",
        default=DEFAULT_SOURCE_REPO_ID,
        help="Source LeRobot dataset repo ID.",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Optional local source dataset root.",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Optional Hugging Face dataset revision.",
    )
    parser.add_argument(
        "--prepared-repo-id",
        default=DEFAULT_PREPARED_REPO_ID,
        help="Repo ID recorded in the prepared local dataset metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the prepared ACT-ready dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the prepared dataset even if the output directory already exists.",
    )
    return parser.parse_args()


def make_combined_action(row: dict, _episode_idx: int, _frame_in_ep: int) -> np.ndarray:
    position = np.asarray(row[ACTION_POSITION_KEY], dtype=np.float32)
    orientation = np.asarray(row[ACTION_ORIENTATION_KEY], dtype=np.float32)
    return np.concatenate([position, orientation], axis=0)


def compute_combined_action_stats(dataset: LeRobotDataset) -> dict[str, list[float]]:
    hf_dataset = dataset.hf_dataset.with_format(None)
    positions = np.asarray(hf_dataset[ACTION_POSITION_KEY], dtype=np.float32)
    orientations = np.asarray(hf_dataset[ACTION_ORIENTATION_KEY], dtype=np.float32)
    actions = np.concatenate([positions, orientations], axis=1)
    return {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    info_path = output_dir / "meta" / "info.json"

    if info_path.exists() and not args.force:
        print(f"Prepared ACT dataset already exists at: {output_dir}")
        print(f"Prepared repo_id: {args.prepared_repo_id}")
        return 0

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    source_dataset = LeRobotDataset(
        args.source_repo_id,
        root=args.source_root,
        revision=args.source_revision,
    )
    combined_action_stats = compute_combined_action_stats(source_dataset)

    prepared_dataset = modify_features(
        dataset=source_dataset,
        add_features={
            COMBINED_ACTION_KEY: (
                make_combined_action,
                {
                    "dtype": "float32",
                    "shape": (len(COMBINED_ACTION_NAMES),),
                    "names": COMBINED_ACTION_NAMES,
                },
            )
        },
        output_dir=output_dir,
        repo_id=args.prepared_repo_id,
    )

    stats = dict(prepared_dataset.meta.stats or {})
    stats[COMBINED_ACTION_KEY] = combined_action_stats
    write_stats(stats, prepared_dataset.root)

    print(f"Prepared ACT dataset root: {prepared_dataset.root}")
    print(f"Prepared ACT dataset repo_id: {prepared_dataset.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
