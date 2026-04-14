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

"""Launch ACT training or fine-tuning for a Hugging Face dataset.

This wrapper defaults to the `slobot/aic` dataset and forwards extra arguments to
`lerobot-train`, so it stays flexible even when you need to tweak policy or
dataset-specific options later on.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

DEFAULT_DATASET_REPO_ID = "slobot/aic"
DEFAULT_PREPARED_DATASET_REPO_ID = "slobot/aic_act"
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PREPARED_DATASET_ROOT = ROOT_DIR / "outputs" / "datasets" / "act_ready_slobot_aic"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Launch ACT training for a Hugging Face dataset. By default this targets "
            "`slobot/aic`, and any unknown flags are forwarded directly to "
            "`lerobot-train`."
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
        help="Optional local source dataset root to use instead of downloading from the Hub.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision to pin when loading from the Hub.",
    )
    parser.add_argument(
        "--prepared-dataset-repo-id",
        default=DEFAULT_PREPARED_DATASET_REPO_ID,
        help="Repo ID recorded in the locally prepared ACT-ready dataset.",
    )
    parser.add_argument(
        "--prepared-dataset-root",
        default=str(DEFAULT_PREPARED_DATASET_ROOT),
        help="Local output directory for the prepared ACT-ready dataset.",
    )
    parser.add_argument(
        "--skip-prepare-dataset",
        action="store_true",
        help="Skip the AIC-to-ACT dataset preparation step and train directly on --dataset-root.",
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
        help="Torch device to pass to lerobot-train (for example: cuda, cuda:0, cpu, mps).",
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
        help="Only validate that the dataset looks reachable, then exit.",
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


def bool_flag(value: bool) -> str:
    return "true" if value else "false"


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


def build_train_command(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    dataset_repo_id = args.prepared_dataset_repo_id if not args.skip_prepare_dataset else args.dataset_repo_id
    dataset_root = args.prepared_dataset_root if not args.skip_prepare_dataset else args.dataset_root

    job_name = args.job_name or f"act_{sanitize_repo_id(args.dataset_repo_id)}"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / "outputs" / "train" / job_name

    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device={args.policy_device}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        "--eval_freq=0",
        f"--seed={args.seed}",
        f"--policy.push_to_hub={bool_flag(args.push_to_hub)}",
        f"--wandb.enable={bool_flag(args.wandb)}",
    ]

    if dataset_root:
        command.append(f"--dataset.root={dataset_root}")
    if args.revision:
        command.append(f"--dataset.revision={args.revision}")

    if args.policy_path:
        command.append(f"--policy.path={args.policy_path}")
    else:
        command.append("--policy.type=act")

    if args.push_to_hub:
        if not args.policy_repo_id:
            raise SystemExit("--push-to-hub requires --policy-repo-id.")
        command.append(f"--policy.repo_id={args.policy_repo_id}")

    if args.wandb:
        command.append(f"--wandb.project={args.wandb_project}")
        if args.wandb_entity:
            command.append(f"--wandb.entity={args.wandb_entity}")
        if args.wandb_mode:
            command.append(f"--wandb.mode={args.wandb_mode}")

    command.extend(extra_args)
    return command


def build_prepare_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT_DIR / "examples" / "training" / "prepare_aic_act_dataset.py"),
        f"--source-repo-id={args.dataset_repo_id}",
        f"--prepared-repo-id={args.prepared_dataset_repo_id}",
        f"--output-dir={args.prepared_dataset_root}",
    ]
    if args.dataset_root:
        command.append(f"--source-root={args.dataset_root}")
    if args.revision:
        command.append(f"--source-revision={args.revision}")
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

    if args.check_only:
        return 0

    if not args.skip_prepare_dataset:
        prepare_command = build_prepare_command(args)
        print("Preparing ACT-ready dataset:")
        print(shlex.join(prepare_command))
        if not args.dry_run:
            prepare_result = subprocess.run(prepare_command, cwd=ROOT_DIR, check=False)
            if prepare_result.returncode != 0:
                return prepare_result.returncode

    command = build_train_command(args, extra_args)
    print("Generated training command:")
    print(shlex.join(command))

    if args.dry_run:
        return 0

    result = subprocess.run(command, cwd=ROOT_DIR, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
