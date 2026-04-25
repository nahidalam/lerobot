# SmolVLA Finetuning On `slobot/aic`

This branch adds a SmolVLA launcher for finetuning `lerobot/smolvla_base` on the AIC dataset without creating an intermediate dataset.

The generated training config uses the current `slobot/aic` layout on `main`:

- `action` is synthesized from `action.tcp_offset.linear.*` and `action.tcp_offset.angular.*`.
- the SmolVLA language instruction comes from the dataset's standard `task` text in `meta/tasks.parquet`.
- `observation.state` is built in the SmolVLA preprocessor from `observation.tcp_offset.*` by default.
- `observation.tcp_offset.*` is normalized before concatenation.
- all three camera feeds are kept by default.

## What SmolVLA Expects

The official SmolVLA docs say the model conditions on:

- multiple camera views,
- the robot's current sensorimotor state,
- a natural-language instruction.

For AIC, this launcher maps those inputs as:

- cameras: all three dataset image streams by default,
- state: normalized `observation.tcp_offset.*`,
- language: the dataset `task` text.

If you explicitly want to append a numeric task id to the state as well, set `TASK_ID_KEY=auto`.

## Setup

Install SmolVLA dependencies:

```bash
uv sync --locked --extra smolvla
```

Log in to Hugging Face and Weights & Biases before launching training:

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
uv run wandb login --relogin "$WANDB_API_KEY"
```

The wrapper defaults W&B logging to the `slobot` workspace for the competition.

## Validate Without Training

```bash
CHECK_ONLY=true examples/training/train_smolvla_aic.sh
```

The validation should print:

- `Dataset revision: main`
- all three selected cameras
- `Instruction source: dataset task text from meta/tasks.parquet`
- `Optional task-id state keys: []`
- `Observation tcp_offset keys: ['observation.tcp_offset...']`
- `Observation state: [normalized tcp_offset keys]`
- `W&B entity/project: slobot/lerobot-smolvla-aic`

## Train

```bash
examples/training/train_smolvla_aic.sh
```

The wrapper expands to a `lerobot-train` run with these defaults:

- dataset: `slobot/aic`
- dataset revision: `main`
- policy type: `smolvla`
- pretrained checkpoint: `lerobot/smolvla_base`
- batch size: `8`
- steps: `20000`
- frozen vision encoder: enabled
- train expert only: enabled
- train state projection: enabled
- W&B entity/project: `slobot/lerobot-smolvla-aic`

Override defaults with environment variables:

```bash
BATCH_SIZE=4 STEPS=10000 FREEZE_VISION_ENCODER=false TRAIN_EXPERT_ONLY=false examples/training/train_smolvla_aic.sh
```

If you explicitly want to append the numeric one-hot task id to the state in addition to the language
instruction, set `TASK_ID_KEY=auto`:

```bash
TASK_ID_KEY=auto examples/training/train_smolvla_aic.sh
```

Use quantile normalization only if the dataset stats include quantiles:

```bash
NORMALIZATION=quantiles examples/training/train_smolvla_aic.sh
```

Push the resulting policy to the Hub after training:

```bash
POLICY_REPO_ID=slobot/aic-smolvla-nahid examples/training/train_smolvla_aic.sh
```

## Notes

The official SmolVLA docs say fine-tuning `smolvla_base` for `20k` steps takes roughly `~4 hours` on a single A100 GPU, and they recommend starting with a smaller batch size if needed and scaling up as memory allows.
