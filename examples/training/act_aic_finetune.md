# ACT training on `slobot/aic`

The training path on this branch still uses plain `lerobot-train`, but `slobot/aic` needs one prep step first: the raw dataset stores the target action in `action.tcp.position` and `action.tcp.orientation`, while ACT expects a single `action` vector. The included helper creates a local ACT-ready copy and then launches `lerobot-train` on that prepared dataset.

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

## 3. Prepare an ACT-ready local dataset copy

```bash
uv run python examples/training/prepare_aic_act_dataset.py
```

By default this creates a local dataset at:

```bash
outputs/datasets/act_ready_slobot_aic
```

## 4. Run the docs-style ACT training command

This is the direct command shape from the LeRobot ACT docs, adapted to the prepared local dataset:

```bash
uv run lerobot-train \
  --dataset.repo_id=slobot/aic_act \
  --dataset.root=outputs/datasets/act_ready_slobot_aic \
  --policy.type=act \
  --output_dir=outputs/train/act_slobot_aic \
  --job_name=act_slobot_aic \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=<your-hf-username>/act-aic
```

## 5. Use the included shell script

The wrapper below prepares the ACT-ready dataset if needed, then runs `lerobot-train` directly:

```bash
examples/training/train_act_aic.sh
```

If you want to push the trained policy to the Hub:

```bash
POLICY_REPO_ID=<your-hf-username>/act-aic \
examples/training/train_act_aic.sh
```

## 6. Fine-tune from an existing ACT checkpoint

If by "finetune" you want to continue from an existing ACT model, point the script at it:

```bash
POLICY_PATH=/path/to/act_checkpoint \
examples/training/train_act_aic.sh
```

`POLICY_PATH` can be a local checkpoint directory or a Hugging Face model repo.

## 7. Useful overrides

The shell script forwards extra arguments to `lerobot-train`, so you can still tune it inline:

```bash
POLICY_REPO_ID=<your-hf-username>/act-aic \
examples/training/train_act_aic.sh \
  --batch_size=16 \
  --steps=50000 \
  --policy.use_amp=true
```

If your EC2 instance already has an ACT-ready local dataset copy and you want to skip rebuilding it:

```bash
PREPARE_ACT_DATASET=false \
ACT_DATASET_ROOT=/path/to/local/act_ready_aic_dataset \
ACT_DATASET_REPO_ID=slobot/aic_act \
examples/training/train_act_aic.sh
```
