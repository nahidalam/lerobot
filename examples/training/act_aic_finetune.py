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

"""Launch ACT training for `slobot/aic` without a prepared intermediate dataset.

This helper inspects the dataset metadata, generates an explicit `lerobot-train`
config, and then launches `lerobot-train` with that config.

It is opinionated in two ways:
1. It keeps only the requested camera keys in the ACT policy inputs so unused
   cameras do not get moved onto the GPU.
2. If a `task_id`-like observation exists, it is remapped to
   `observation.environment_state` and passed through with identity
   normalization.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from lerobot.configs import DatasetConfig, FeatureType, NormalizationMode, PolicyFeature, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

DEFAULT_DATASET_REPO_ID = "slobot/aic"
DEFAULT_KEEP_CAMERAS = "all"
DEFAULT_TASK_ID_KEY = "auto"
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT_DIR / "outputs" / "generated_configs"
AIC_ACTION_OFFSET_KEY = "action.tcp_offset"
AIC_ACTION_POSITION_KEY = "action.tcp.position"
AIC_ACTION_ORIENTATION_KEY = "action.tcp.orientation"
AIC_ACTION_OFFSET_COMPONENT_KEYS = [
    "action.tcp_offset.linear.x",
    "action.tcp_offset.linear.y",
    "action.tcp_offset.linear.z",
    "action.tcp_offset.angular.x",
    "action.tcp_offset.angular.y",
    "action.tcp_offset.angular.z",
]
AIC_OBS_TCP_OFFSET_KEY = "observation.tcp_offset"
AIC_OBS_TCP_OFFSET_COMPONENT_KEYS = [
    "observation.tcp_offset.linear.x",
    "observation.tcp_offset.linear.y",
    "observation.tcp_offset.linear.z",
    "observation.tcp_offset.angular.x",
    "observation.tcp_offset.angular.y",
    "observation.tcp_offset.angular.z",
]


def load_probe_sample(
    repo_id: str,
    root: str | None,
    revision: str | None,
) -> dict[str, Any] | None:
    try:
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            revision=revision,
            episodes=[0],
        )
    except Exception:
        return None

    try:
        return dataset[0]
    except Exception:
        return None


def infer_feature_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        parsed_shape = tuple(int(dim) for dim in shape)
        return parsed_shape or (1,)

    try:
        length = len(value)
    except TypeError:
        return (1,)
    return (int(length),)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dataset-aware ACT training config for slobot/aic and "
            "run lerobot-train with it."
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
        default=None,
        help="Optional dataset revision to pin when loading from the Hub.",
    )
    parser.add_argument(
        "--keep-cameras",
        default=DEFAULT_KEEP_CAMERAS,
        help=(
            "Comma-separated camera feature keys to expose to ACT. Use 'all' to keep "
            "every camera present in the dataset metadata."
        ),
    )
    parser.add_argument(
        "--task-id-key",
        default=DEFAULT_TASK_ID_KEY,
        help=(
            "Observation key to route into observation.environment_state without "
            "normalization. Use 'auto' to detect a task_id-like key or 'none' to disable."
        ),
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Training job name. Defaults to act_<dataset_repo_id>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to outputs/train/<job_name> inside this repo.",
    )
    parser.add_argument(
        "--config-output",
        default=None,
        help=(
            "Optional path for the generated lerobot-train JSON config. Defaults to "
            "outputs/generated_configs/<job_name>.json."
        ),
    )
    parser.add_argument(
        "--policy-path",
        default=None,
        help=(
            "Optional ACT checkpoint or Hub repo to fine-tune from. If omitted, "
            "training starts from a fresh ACT policy."
        ),
    )
    parser.add_argument(
        "--policy-device",
        default="cuda",
        help="Torch device to pass to ACT and lerobot-train.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--log-freq", type=int, default=100, help="Logging frequency in steps.")
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="Checkpoint save frequency in steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Training seed passed through to lerobot-train.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the resulting policy to the Hugging Face Hub when training finishes.",
    )
    parser.add_argument(
        "--policy-repo-id",
        default=None,
        help="Destination repo ID for pushing the trained policy.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default="lerobot-act-aic",
        help="Weights & Biases project name to use when --wandb is set.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional Weights & Biases entity.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        help="Optional Weights & Biases mode, for example offline.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dataset access and print the generated training summary, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated lerobot-train command without running it.",
    )
    parser.add_argument(
        "--skip-hf-check",
        action="store_true",
        help="Skip the preflight Hugging Face dataset-access check.",
    )
    return parser.parse_known_args()


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace("-", "_")


def preflight_dataset_access(repo_id: str, revision: str | None) -> None:
    try:
        info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    except RepositoryNotFoundError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code in {401, 403}:
            raise SystemExit(
                f"Dataset access check failed for '{repo_id}'. It looks private or gated from "
                "this machine. Authenticate with `huggingface-cli login` or export `HF_TOKEN`, "
                "then retry."
            ) from exc
        raise SystemExit(
            f"Dataset '{repo_id}' was not found on the Hugging Face Hub. Double-check the repo ID."
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f"Unable to validate dataset '{repo_id}': {exc}") from exc

    print(f"Dataset access OK: {info.id}")


def parse_keep_cameras(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    keep_cameras = {camera_key.strip() for camera_key in value.split(",") if camera_key.strip()}
    if not keep_cameras:
        raise SystemExit("--keep-cameras must contain at least one camera key or be set to 'all'.")
    return keep_cameras


def resolve_task_id_keys(
    dataset_features: dict[str, dict],
    requested_key: str,
    sample_keys: set[str] | None = None,
) -> list[str]:
    if requested_key.lower() == "none":
        return []

    if requested_key != "auto":
        requested_keys = [key.strip() for key in requested_key.split(",") if key.strip()]
        missing = [
            key
            for key in requested_keys
            if key not in dataset_features and (sample_keys is None or key not in sample_keys)
        ]
        if missing:
            raise SystemExit(
                f"Requested task-id keys were not found in the dataset metadata or sample keys: {missing}"
            )
        return requested_keys

    prefixed_keys = sorted(
        {
            key
            for key in dataset_features
            if key.startswith("observation.task_id.")
        }
        | {
            key
            for key in (sample_keys or set())
            if key.startswith("observation.task_id.")
        }
    )
    if prefixed_keys:
        return prefixed_keys

    fallback_keys = [
        key
        for key in ("observation.task_id", "task_id")
        if key in dataset_features or (sample_keys is not None and key in sample_keys)
    ]
    return fallback_keys


def resolve_component_keys(
    policy_features: dict[str, PolicyFeature],
    preferred_keys: list[str],
    aggregate_key: str | None = None,
) -> tuple[list[str], str]:
    if all(key in policy_features for key in preferred_keys):
        return preferred_keys, preferred_keys[0].rsplit(".", 2)[0]

    if aggregate_key is not None and aggregate_key in policy_features:
        return [aggregate_key], aggregate_key

    return [], ""


def build_policy_features(
    dataset_features: dict[str, dict],
    keep_cameras: set[str] | None,
    task_id_keys: list[str],
    sample: dict[str, Any] | None = None,
) -> tuple[
    dict[str, PolicyFeature],
    dict[str, PolicyFeature],
    dict[str, str],
    list[str],
    str,
    list[str],
    list[str],
]:
    policy_features = dataset_to_policy_features(dataset_features)
    action_keys, action_source = resolve_component_keys(
        policy_features,
        preferred_keys=AIC_ACTION_OFFSET_COMPONENT_KEYS,
        aggregate_key=AIC_ACTION_OFFSET_KEY,
    )
    if action_keys:
        action_dim = sum(policy_features[key].shape[0] for key in action_keys)
        policy_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
        action_source_label = f"{action_source} -> {ACTION}"
    elif ACTION not in policy_features:
        if (
            AIC_ACTION_POSITION_KEY in policy_features
            and AIC_ACTION_ORIENTATION_KEY in policy_features
        ):
            total_action_dim = (
                policy_features[AIC_ACTION_POSITION_KEY].shape[0]
                + policy_features[AIC_ACTION_ORIENTATION_KEY].shape[0]
            )
            policy_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(total_action_dim,),
            )
            action_source_label = f"{AIC_ACTION_POSITION_KEY} + {AIC_ACTION_ORIENTATION_KEY} -> {ACTION}"
        else:
            raise SystemExit(
                f"The dataset does not expose a top-level '{ACTION}' feature, and the expected raw AIC "
                f"action fields ({AIC_ACTION_OFFSET_KEY}), ({', '.join(AIC_ACTION_OFFSET_COMPONENT_KEYS)}) or ({AIC_ACTION_POSITION_KEY}, "
                f"{AIC_ACTION_ORIENTATION_KEY}) were not found."
            )
    else:
        action_source_label = ACTION

    tcp_offset_input_keys, tcp_offset_source = resolve_component_keys(
        policy_features,
        preferred_keys=AIC_OBS_TCP_OFFSET_COMPONENT_KEYS,
        aggregate_key=AIC_OBS_TCP_OFFSET_KEY,
    )
    if not tcp_offset_input_keys and all(key in policy_features for key in AIC_ACTION_OFFSET_COMPONENT_KEYS):
        tcp_offset_input_keys = AIC_ACTION_OFFSET_COMPONENT_KEYS
        tcp_offset_source = "action.tcp_offset.*"
    elif not tcp_offset_input_keys and AIC_ACTION_OFFSET_KEY in policy_features:
        tcp_offset_input_keys = [AIC_ACTION_OFFSET_KEY]
        tcp_offset_source = AIC_ACTION_OFFSET_KEY

    if not tcp_offset_input_keys:
        raise SystemExit(
            "The dataset does not expose tcp_offset features that can be used to build observation.state. "
            f"Expected one of: {AIC_OBS_TCP_OFFSET_KEY}, {AIC_ACTION_OFFSET_KEY}, "
            f"{AIC_OBS_TCP_OFFSET_COMPONENT_KEYS}, {AIC_ACTION_OFFSET_COMPONENT_KEYS}."
        )

    tcp_offset_dim = sum(policy_features[key].shape[0] for key in tcp_offset_input_keys)

    if not task_id_keys:
        raise SystemExit(
            "A task_id-like key is required to build observation.state as [task_id ; tcp_offset]. "
            "Please pass --task-id-key explicitly if auto-detection failed."
        )

    task_id_features: dict[str, PolicyFeature] = {}
    for key in task_id_keys:
        if key in policy_features:
            task_id_features[key] = PolicyFeature(type=FeatureType.STATE, shape=policy_features[key].shape)
        else:
            if sample is None or key not in sample:
                raise SystemExit(f"Resolved task-id key '{key}' is not part of the dataset features or sample.")
            task_id_features[key] = PolicyFeature(type=FeatureType.STATE, shape=infer_feature_shape(sample[key]))

    available_cameras = sorted(
        key for key, feature in policy_features.items() if feature.type is FeatureType.VISUAL
    )
    if keep_cameras is not None:
        unknown_cameras = sorted(camera_key for camera_key in keep_cameras if camera_key not in available_cameras)
        if unknown_cameras:
            raise SystemExit(
                f"Requested camera keys are missing from the dataset metadata: {unknown_cameras}. "
                f"Available cameras: {available_cameras}"
            )

    output_features = {ACTION: policy_features[ACTION]}
    input_features: dict[str, PolicyFeature] = {
        key: feature
        for key, feature in policy_features.items()
        if feature.type is FeatureType.VISUAL and (keep_cameras is None or key in keep_cameras)
    }

    for key in task_id_keys:
        input_features[key] = task_id_features[key]
    for key in tcp_offset_input_keys:
        input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=policy_features[key].shape)

    input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(sum(feature.shape[0] for feature in task_id_features.values()) + tcp_offset_dim,),
    )

    rename_map: dict[str, str] = {}
    return (
        input_features,
        output_features,
        rename_map,
        available_cameras,
        action_source_label,
        task_id_keys,
        tcp_offset_input_keys,
    )


def write_train_config(config: TrainPipelineConfig, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f, draccus.config_type("json"):
        draccus.dump(config, f, indent=4)


def build_train_config(
    args: argparse.Namespace,
    input_features: dict[str, PolicyFeature],
    output_features: dict[str, PolicyFeature],
    rename_map: dict[str, str],
) -> tuple[TrainPipelineConfig, Path]:
    push_to_hub = args.push_to_hub or args.policy_repo_id is not None
    if push_to_hub and not args.policy_repo_id:
        raise SystemExit("--push-to-hub requires --policy-repo-id.")

    job_name = args.job_name or f"act_{sanitize_repo_id(args.dataset_repo_id)}"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / "outputs" / "train" / job_name
    config_path = Path(args.config_output) if args.config_output else DEFAULT_CONFIG_DIR / f"{job_name}.json"

    normalization_mapping = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    policy_kwargs: dict[str, object] = {
        "input_features": input_features,
        "output_features": output_features,
        "device": args.policy_device,
        "normalization_mapping": normalization_mapping,
        "push_to_hub": push_to_hub,
        "repo_id": args.policy_repo_id,
    }
    if args.policy_path:
        policy_kwargs["pretrained_path"] = Path(args.policy_path)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            revision=args.revision,
        ),
        policy=ACTConfig(**policy_kwargs),
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


def build_train_command(config_path: Path, extra_args: list[str]) -> list[str]:
    command = [
        "uv",
        "run",
        "lerobot-train",
        f"--config_path={config_path}",
    ]
    command.extend(extra_args)
    return command


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
    )
    cfg, config_path = build_train_config(args, input_features, output_features, rename_map)
    write_train_config(cfg, config_path)

    selected_cameras = sorted(key for key, feature in input_features.items() if feature.type is FeatureType.VISUAL)
    print(f"Generated train config: {config_path}")
    print(f"Dataset repo: {args.dataset_repo_id}")
    print(f"Action source: {action_source}")
    print(f"Observation tcp_offset keys: {tcp_offset_input_keys}")
    print(f"Available cameras: {available_cameras}")
    print(f"Selected cameras: {selected_cameras}")
    print(f"Task-id source keys: {resolved_task_id_keys}")
    print("Observation state: [unnormalized task_id keys ; normalized tcp_offset keys]")

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
