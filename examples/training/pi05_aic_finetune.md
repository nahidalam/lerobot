# Pi0.5 Finetuning On `slobot/aic`

This branch adds a Pi0.5 launcher for finetuning `lerobot/pi05_base` on the AIC dataset without creating an intermediate dataset.

The generated training config uses the current `slobot/aic` layout on `main`:

- `action` is synthesized from `action.tcp_offset.linear.*` and `action.tcp_offset.angular.*`.
- the Pi0.5 language instruction comes from the dataset's standard `task` text in `meta/tasks.parquet`.
- `observation.state` is built in the Pi0.5 preprocessor from `observation.tcp_offset.*` by default.
- `observation.tcp_offset.*` is normalized before concatenation.
- all three camera feeds are kept by default.

## Setup

Install Pi0.5 dependencies:

```bash
uv sync --locked --extra training --extra pi
```

Log in to Hugging Face and Weights & Biases before launching training:

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
uv run wandb login --relogin "$WANDB_API_KEY"
```

## Validate Without Training

```bash
CHECK_ONLY=true examples/training/train_pi05_aic.sh
```

The validation should print:

- `Dataset revision: main`
- all three selected cameras
- `Instruction source: dataset task text from meta/tasks.parquet`
- `Optional task-id state keys: []`
- `Observation tcp_offset keys: ['observation.tcp_offset...']`
- `Observation state: [normalized tcp_offset keys]`

## Train

```bash
examples/training/train_pi05_aic.sh
```

The wrapper expands to a `lerobot-train` run with these defaults:

- dataset: `slobot/aic`
- dataset revision: `main`
- policy type: `pi05`
- pretrained checkpoint: `lerobot/pi05_base`
- dtype: `bfloat16`
- gradient checkpointing: enabled
- `torch.compile`: enabled
- batch size: `4`
- steps: `3000`
- W&B project: `lerobot-pi05-aic`

Override defaults with environment variables:

```bash
BATCH_SIZE=2 STEPS=6000 COMPILE_MODEL=false TRAIN_EXPERT_ONLY=true examples/training/train_pi05_aic.sh
```

If you explicitly want to append the numeric one-hot task id to the state in addition to the language
instruction, set `TASK_ID_KEY=auto`:

```bash
TASK_ID_KEY=auto examples/training/train_pi05_aic.sh
```

Use quantile normalization only if the dataset stats include quantiles:

```bash
NORMALIZATION=quantiles examples/training/train_pi05_aic.sh
```

Push the resulting policy to the Hub after training:

```bash
POLICY_REPO_ID=slobot/aic-pi05-nahid examples/training/train_pi05_aic.sh
```

## Notes

The upstream Pi0.5 docs recommend `policy.pretrained_path=lerobot/pi05_base`, `policy.dtype=bfloat16`, gradient checkpointing, and optionally `policy.train_expert_only=true` when memory is tight. They also note that if a dataset does not include quantile stats, Pi0.5 can be trained with `MEAN_STD` normalization for state and action. This launcher uses `MEAN_STD` by default for compatibility with AIC snapshots.
