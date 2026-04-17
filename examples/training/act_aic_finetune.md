# ACT training on `slobot/aic`

The current path on this branch trains ACT directly from `slobot/aic` with plain
`lerobot-train`. We no longer need to build a prepared intermediate dataset first
as long as the dataset already exposes a top-level `action` feature.

The launcher generates a concrete `lerobot-train` config before running, so the
training setup is explicit and reproducible.

## 1. Install training dependencies

```bash
uv sync --locked --extra training
```

## 2. Authenticate to Hugging Face

If the dataset is private or gated from the current machine:

```bash
huggingface-cli login
```

or:

```bash
export HF_TOKEN=hf_your_token
```

## 3. Direct ACT training

The simplest path is now:

```bash
uv run python examples/training/act_aic_finetune.py --wandb
```

What this helper does before it launches `lerobot-train`:

- reads the dataset metadata from `slobot/aic`
- prefers `action.tcp_offset` as the ACT action target, with a legacy fallback for older split TCP action keys
- keeps only the requested camera keys in ACT inputs
- routes `task_id` into `observation.environment_state` when that key is present
- disables normalization for that env-state input
- writes a generated JSON train config, then runs `lerobot-train --config_path=...`

By default it keeps only:

```bash
observation.images.center_camera
```

## 4. Docs-style shell wrapper

If you prefer a shell entrypoint:

```bash
examples/training/train_act_aic.sh
```

That script is now just a thin wrapper around
`examples/training/act_aic_finetune.py`.

## 5. Push the trained policy to the Hub

```bash
POLICY_REPO_ID=<your-hf-username>/act-aic \
examples/training/train_act_aic.sh
```

## 6. Fine-tune from an existing ACT checkpoint

```bash
POLICY_PATH=/path/to/act_checkpoint \
examples/training/train_act_aic.sh
```

`POLICY_PATH` can be a local checkpoint directory or a Hugging Face model repo.

## 7. Useful overrides

Override the kept cameras:

```bash
ACT_KEEP_CAMERAS=all \
examples/training/train_act_aic.sh
```

or:

```bash
ACT_KEEP_CAMERAS=observation.images.center_camera,observation.images.left_camera \
examples/training/train_act_aic.sh
```

Override the task-id observation key explicitly:

```bash
TASK_ID_KEY=observation.task_id \
examples/training/train_act_aic.sh
```

Disable task-id passthrough entirely:

```bash
TASK_ID_KEY=none \
examples/training/train_act_aic.sh
```

Forward extra `lerobot-train` flags:

```bash
examples/training/train_act_aic.sh \
  --batch_size=16 \
  --steps=50000 \
  --policy.use_amp=true
```

Dry-run the generated command:

```bash
DRY_RUN=true examples/training/train_act_aic.sh
```

## 8. Legacy fallback

If you ever need to work with an older `slobot/aic` snapshot that still stores the
target action in separate `action.tcp.position` and `action.tcp.orientation`
features instead of `action.tcp_offset`, the old dataset-prep helper is still available:

```bash
uv run python examples/training/prepare_aic_act_dataset.py
```

That helper creates a local ACT-ready dataset copy and is no longer the default
path for the current dataset layout.
