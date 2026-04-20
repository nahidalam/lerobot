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

The current dataset stores the target action in `action.tcp_offset`. Older
snapshots stored it across two keys: `action.tcp.position` and
`action.tcp.orientation`.

ACT in LeRobot expects a single `action` feature, so this script creates a local,
ACT-ready LeRobot dataset with that action copied or synthesized into a top-level
`action` vector.

By default it also keeps only the center camera, which reduces GPU memory enough
for ACT training to start on a single L40S.

This helper is now a legacy fallback for older dataset snapshots. If `slobot/aic`
already exposes a top-level `action` feature, prefer
`examples/training/act_aic_finetune.py`, which trains directly from the source
dataset without creating an intermediate local copy.
"""

from __future__ import annotations

import argparse
import shutil
from functools import partial
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import modify_features
from lerobot.datasets.io_utils import write_stats

DEFAULT_SOURCE_REPO_ID = "slobot/aic"
DEFAULT_PREPARED_REPO_ID = "slobot/aic_act"
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "datasets" / "act_ready_slobot_aic"
DEFAULT_KEEP_CAMERAS = ",".join(
    [
        "observation.images.center_camera",
        "observation.images.left_camera",
        "observation.images.right_camera",
    ]
)

ACTION_OFFSET_KEY = "action.tcp_offset"
ACTION_OFFSET_COMPONENT_KEYS = [
    "action.tcp_offset.linear.x",
    "action.tcp_offset.linear.y",
    "action.tcp_offset.linear.z",
    "action.tcp_offset.angular.x",
    "action.tcp_offset.angular.y",
    "action.tcp_offset.angular.z",
]
ACTION_POSITION_KEY = "action.tcp.position"
ACTION_ORIENTATION_KEY = "action.tcp.orientation"
COMBINED_ACTION_KEY = "action"
LEGACY_COMBINED_ACTION_NAMES = [
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
        "--keep-cameras",
        default=DEFAULT_KEEP_CAMERAS,
        help="Comma-separated camera feature keys to keep in the prepared dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the prepared dataset even if the output directory already exists.",
    )
    return parser.parse_args()


def resolve_action_source(dataset: LeRobotDataset) -> tuple[str, tuple[int, ...], list[str]]:
    features = dataset.meta.features
    if all(key in features for key in ACTION_OFFSET_COMPONENT_KEYS):
        return ("tcp_offset_components", (len(ACTION_OFFSET_COMPONENT_KEYS),), ACTION_OFFSET_COMPONENT_KEYS)

    if ACTION_OFFSET_KEY in features:
        source_feature = features[ACTION_OFFSET_KEY]
        return (
            ACTION_OFFSET_KEY,
            tuple(source_feature["shape"]),
            list(source_feature.get("names") or []),
        )

    if ACTION_POSITION_KEY in features and ACTION_ORIENTATION_KEY in features:
        position_shape = tuple(features[ACTION_POSITION_KEY]["shape"])
        orientation_shape = tuple(features[ACTION_ORIENTATION_KEY]["shape"])
        names = list(features[ACTION_POSITION_KEY].get("names") or []) + list(
            features[ACTION_ORIENTATION_KEY].get("names") or []
        )
        if not names:
            names = LEGACY_COMBINED_ACTION_NAMES
        return "legacy_split_tcp", (position_shape[0] + orientation_shape[0],), names

    raise SystemExit(
        "Unable to prepare ACT action data because none of the expected source keys were found. "
        f"Expected one of: {ACTION_OFFSET_KEY} or ({ACTION_POSITION_KEY}, {ACTION_ORIENTATION_KEY})."
    )


def make_combined_action(
    row: dict,
    _episode_idx: int,
    _frame_in_ep: int,
    action_source: str,
) -> np.ndarray:
    if action_source == "tcp_offset_components":
        return np.asarray([row[key] for key in ACTION_OFFSET_COMPONENT_KEYS], dtype=np.float32)

    if action_source == ACTION_OFFSET_KEY:
        return np.asarray(row[ACTION_OFFSET_KEY], dtype=np.float32)

    position = np.asarray(row[ACTION_POSITION_KEY], dtype=np.float32)
    orientation = np.asarray(row[ACTION_ORIENTATION_KEY], dtype=np.float32)
    return np.concatenate([position, orientation], axis=0)


def compute_combined_action_stats(dataset: LeRobotDataset, action_source: str) -> dict[str, list[float]]:
    hf_dataset = dataset.hf_dataset.with_format(None)
    if action_source == "tcp_offset_components":
        actions = np.stack([np.asarray(hf_dataset[key], dtype=np.float32) for key in ACTION_OFFSET_COMPONENT_KEYS], axis=1)
    elif action_source == ACTION_OFFSET_KEY:
        actions = np.asarray(hf_dataset[ACTION_OFFSET_KEY], dtype=np.float32)
    else:
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
    keep_cameras = {camera_key.strip() for camera_key in args.keep_cameras.split(",") if camera_key.strip()}

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
    action_source, action_shape, action_names = resolve_action_source(source_dataset)
    combined_action_stats = compute_combined_action_stats(source_dataset, action_source)
    remove_features = [camera_key for camera_key in source_dataset.meta.camera_keys if camera_key not in keep_cameras]

    prepared_dataset = modify_features(
        dataset=source_dataset,
        add_features={
            COMBINED_ACTION_KEY: (
                partial(make_combined_action, action_source=action_source),
                {
                    "dtype": "float32",
                    "shape": action_shape,
                    "names": action_names,
                },
            )
        },
        remove_features=remove_features or None,
        output_dir=output_dir,
        repo_id=args.prepared_repo_id,
    )

    stats = dict(prepared_dataset.meta.stats or {})
    stats[COMBINED_ACTION_KEY] = combined_action_stats
    write_stats(stats, prepared_dataset.root)

    print(f"Prepared ACT dataset root: {prepared_dataset.root}")
    print(f"Prepared ACT dataset repo_id: {prepared_dataset.repo_id}")
    print(f"Action source: {action_source} -> {COMBINED_ACTION_KEY}")
    print(f"Kept cameras: {sorted(keep_cameras)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
