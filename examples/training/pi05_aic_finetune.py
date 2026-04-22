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

"""Launch Pi0.5 finetuning for `slobot/aic`.

Pi0.5 is a VLA policy, so task conditioning comes from the LeRobot dataset's
natural-language `task` field. This helper generates a Pi0.5 config where:

1. the text instruction is read from `meta/tasks.parquet` through LeRobot's
   standard `task` field and tokenized by the Pi0.5 preprocessor;
2. `observation.state` is built by the Pi0.5 preprocessor from normalized
   `observation.tcp_offset.*` keys by default;
3. the policy action target is synthesized from `action.tcp_offset.*`;
4. all three AIC cameras are kept by default.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

import draccus

from act_aic_finetune import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_KEEP_CAMERAS,
    ROOT_DIR,
    build_policy_features,
    build_train_command,
    load_probe_sample,
    parse_keep_cameras,
    preflight_dataset_access,
    resolve_task_id_keys,
    sanitize_repo_id,
    write_train_config,
)
from lerobot.configs import DatasetConfig, FeatureType, NormalizationMode, PolicyFeature, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.pi05 import PI05Config

DEFAULT_POLICY_PATH = "lerobot/pi05_base"
DEFAULT_DATASET_REVISION = "main"
DEFAULT_PI05_TASK_ID_KEY = "none"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dataset-aware Pi0.5 finetuning config for slobot/aic and run lerobot-train."
        )
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Hugging Face dataset repo ID to train on.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional local dataset root to use instead of downloading from the Hub.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_DATASET_REVISION,
        help="Dataset revision to load. Defaults to main because slobot/aic v3.0 still points at an older schema.",
    )
    parser.add_argument(
        "--keep-cameras",
        default=DEFAULT_KEEP_CAMERAS,
        help="Comma-separated camera feature keys to expose to Pi0.5. Use 'all' to keep every camera.",
    )
    parser.add_argument(
        "--task-id-key",
        default=DEFAULT_PI05_TASK_ID_KEY,
        help=(
            "Optional task-id observation key(s) to append to observation.state. "
            "Pi0.5 uses the dataset's natural-language task instruction by default, so this defaults "
            "to 'none'. Use 'auto' to also include observation.task_id.* keys."
        ),
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Training job name. Defaults to pi05_<dataset_repo_id>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to outputs/train/<job_name> inside this repo.",
    )
    parser.add_argument(
        "--config-output",
        default=None,
        help="Optional path for the generated lerobot-train JSON config.",
    )
    parser.add_argument(
        "--policy-path",
        default=DEFAULT_POLICY_PATH,
        help="Pi0.5 checkpoint or Hub repo to fine-tune from. Use 'none' to train from scratch.",
    )
    parser.add_argument("--policy-device", default="cuda", help="Torch device to pass to Pi0.5.")
    parser.add_argument("--policy-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument(
        "--normalization",
        default="mean_std",
        choices=["mean_std", "quantiles"],
        help=(
            "AIC defaults to MEAN_STD because many snapshots do not include quantile stats. "
            "Use quantiles if the dataset stats include q01/q99."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--steps", type=int, default=3_000, help="Number of finetuning steps.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--log-freq", type=int, default=50, help="Logging frequency in steps.")
    parser.add_argument("--save-freq", type=int, default=500, help="Checkpoint save frequency in steps.")
    parser.add_argument("--seed", type=int, default=1000, help="Training seed.")
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-expert-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--push-to-hub", action="store_true", help="Push the trained policy to HF Hub.")
    parser.add_argument("--policy-repo-id", default=None, help="Destination repo ID for the trained policy.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="lerobot-pi05-aic", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional Weights & Biases entity.")
    parser.add_argument("--wandb-mode", default=None, help="Optional Weights & Biases mode.")
    parser.add_argument("--check-only", action="store_true", help="Validate and write the config, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the lerobot-train command only.")
    parser.add_argument("--skip-hf-check", action="store_true", help="Skip the Hub dataset access check.")
    return parser.parse_known_args()


def make_normalization_mapping(kind: str) -> dict[FeatureType, NormalizationMode]:
    state_action_mode = NormalizationMode.QUANTILES if kind == "quantiles" else NormalizationMode.MEAN_STD
    return {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: state_action_mode,
        FeatureType.ACTION: state_action_mode,
    }


def build_train_config(
    args: argparse.Namespace,
    input_features: dict[str, PolicyFeature],
    output_features: dict[str, PolicyFeature],
    rename_map: dict[str, str],
) -> tuple[TrainPipelineConfig, Path]:
    push_to_hub = args.push_to_hub or args.policy_repo_id is not None
    if push_to_hub and not args.policy_repo_id:
        raise SystemExit("--push-to-hub requires --policy-repo-id.")

    job_name = args.job_name or f"pi05_{sanitize_repo_id(args.dataset_repo_id)}"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / "outputs" / "train" / job_name
    config_path = Path(args.config_output) if args.config_output else DEFAULT_CONFIG_DIR / f"{job_name}.json"

    policy_kwargs: dict[str, object] = {
        "input_features": input_features,
        "output_features": output_features,
        "device": args.policy_device,
        "dtype": args.policy_dtype,
        "normalization_mapping": make_normalization_mapping(args.normalization),
        "compile_model": args.compile_model,
        "gradient_checkpointing": args.gradient_checkpointing,
        "freeze_vision_encoder": args.freeze_vision_encoder,
        "train_expert_only": args.train_expert_only,
        "push_to_hub": push_to_hub,
        "repo_id": args.policy_repo_id,
    }
    if args.policy_path and args.policy_path.lower() != "none":
        policy_kwargs["pretrained_path"] = Path(args.policy_path)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            revision=args.revision,
        ),
        policy=PI05Config(**policy_kwargs),
        output_dir=output_dir,
        job_name=job_name,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        wandb=WandBConfig(
            enable=args.wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
        ),
        rename_map=rename_map,
    )
    return cfg, config_path


def main() -> int:
    args, extra_args = parse_args()

    should_check_hub = not args.skip_hf_check and not args.dataset_root
    if should_check_hub:
        preflight_dataset_access(args.dataset_repo_id, args.revision)
    elif args.dataset_root:
        print(f"Skipping Hub access check because a local dataset root was provided: {args.dataset_root}")
    else:
        print("Skipping Hugging Face dataset-access check.")

    dataset_meta = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.revision,
    )

    keep_cameras = parse_keep_cameras(args.keep_cameras)
    probe_sample = None
    if args.task_id_key != "none" and (
        args.task_id_key == "auto"
        or any(key.strip() not in dataset_meta.features for key in args.task_id_key.split(",") if key.strip())
    ):
        probe_sample = load_probe_sample(
            args.dataset_repo_id,
            root=args.dataset_root,
            revision=args.revision,
        )

    task_id_keys = resolve_task_id_keys(
        dataset_meta.features,
        args.task_id_key,
        sample_keys=set(probe_sample) if probe_sample is not None else None,
    )
    (
        input_features,
        output_features,
        rename_map,
        available_cameras,
        action_source,
        resolved_task_id_keys,
        tcp_offset_input_keys,
    ) = build_policy_features(
        dataset_meta.features,
        keep_cameras=keep_cameras,
        task_id_keys=task_id_keys,
        sample=probe_sample,
        require_task_id=False,
    )
    cfg, config_path = build_train_config(args, input_features, output_features, rename_map)
    write_train_config(cfg, config_path)

    selected_cameras = sorted(key for key, feature in input_features.items() if feature.type is FeatureType.VISUAL)
    print(f"Generated train config: {config_path}")
    print(f"Dataset repo: {args.dataset_repo_id}")
    print(f"Dataset revision: {args.revision}")
    print(f"Policy path: {args.policy_path}")
    print(f"Action source: {action_source}")
    print(f"Observation tcp_offset keys: {tcp_offset_input_keys}")
    print(f"Available cameras: {available_cameras}")
    print(f"Selected cameras: {selected_cameras}")
    print("Instruction source: dataset task text from meta/tasks.parquet")
    print(f"Optional task-id state keys: {resolved_task_id_keys}")
    if resolved_task_id_keys:
        print("Observation state: [unnormalized task_id keys ; normalized tcp_offset keys]")
    else:
        print("Observation state: [normalized tcp_offset keys]")

    if args.check_only:
        return 0

    command = build_train_command(config_path, extra_args)
    print("Generated training command:")
    print(shlex.join(command))

    if args.dry_run:
        return 0

    result = subprocess.run(command, cwd=ROOT_DIR, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
