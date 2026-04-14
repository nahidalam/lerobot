# ACT training on `slobot/aic`

The main path on this branch is plain `lerobot-train`, matching the ACT docs. The shell wrapper just fills in the dataset name and keeps the command easy to rerun on EC2.

## 1. Install training dependencies

```bash
uv sync --locked --extra training
```

## 2. Authenticate to Hugging Face

This dataset looks private or gated from the current machine, so you will likely need one of:

```bash
huggingface-cli login
```

or:

```bash
export HF_TOKEN=hf_your_token
```

## 3. Run the docs-style ACT training command

This is the direct command shape from the LeRobot ACT docs, adapted to `slobot/aic`:

```bash
uv run lerobot-train \
  --dataset.repo_id=slobot/aic \
  --policy.type=act \
  --output_dir=outputs/train/act_slobot_aic \
  --job_name=act_slobot_aic \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=<your-hf-username>/act-aic
```

## 4. Use the included shell script

The wrapper below runs `lerobot-train` directly:

```bash
examples/training/train_act_aic.sh
```

If you want to push the trained policy to the Hub:

```bash
POLICY_REPO_ID=<your-hf-username>/act-aic \
examples/training/train_act_aic.sh
```

## 5. Fine-tune from an existing ACT checkpoint

If by "finetune" you want to continue from an existing ACT model, point the script at it:

```bash
POLICY_PATH=/path/to/act_checkpoint \
examples/training/train_act_aic.sh
```

`POLICY_PATH` can be a local checkpoint directory or a Hugging Face model repo.

## 6. Useful overrides

The shell script forwards extra arguments to `lerobot-train`, so you can still tune it inline:

```bash
POLICY_REPO_ID=<your-hf-username>/act-aic \
examples/training/train_act_aic.sh \
  --batch_size=16 \
  --steps=50000 \
  --policy.use_amp=true
```

If your EC2 instance already has a local dataset copy:

```bash
DATASET_ROOT=/path/to/local/aic_dataset \
examples/training/train_act_aic.sh
```
