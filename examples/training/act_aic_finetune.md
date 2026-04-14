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

By default the prepared dataset keeps only `observation.images.center_camera` so ACT fits on a single GPU more comfortably.

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

If you want to keep a different camera set when preparing the dataset:

```bash
ACT_KEEP_CAMERAS=observation.images.center_camera,observation.images.left_camera \
examples/training/train_act_aic.sh
```

## 8. EC2 runbook used for the actual launch

These are the concrete steps used on the EC2 instance to get the current training run working.

### 8.1 Pull the feature branch

```bash
cd /home/ubuntu/lerobot
git fetch origin
git switch codex/act-aic-finetune
git pull --ff-only
```

### 8.2 Make sure `uv` is on `PATH`

```bash
export PATH=$HOME/.local/bin:$PATH
```

### 8.3 Install training dependencies

```bash
uv sync --locked --extra training
```

### 8.4 Load Hugging Face credentials from `.env`

The working setup used a local `.env` file at `/home/ubuntu/lerobot/.env` containing at least:

```bash
HF_TOKEN=hf_...
```

Load it into the shell before dataset access or training:

```bash
set -a
source /home/ubuntu/lerobot/.env
set +a
```

### 8.5 Verify private dataset access

```bash
uv run python - <<'PY'
import os
from huggingface_hub import HfApi
token = os.environ["HF_TOKEN"]
print(HfApi().dataset_info("slobot/aic", token=token).id)
PY
```

### 8.6 Prepare the ACT-ready local dataset

This step is required because raw `slobot/aic` stores the target action across
`action.tcp.position` and `action.tcp.orientation`, while ACT expects a single `action`.

The working low-memory setup also keeps only the center camera.

```bash
uv run python examples/training/prepare_aic_act_dataset.py \
  --force \
  --output-dir=outputs/datasets/act_ready_slobot_aic \
  --prepared-repo-id=slobot/aic_act
```

Prepared dataset path:

```bash
/home/ubuntu/lerobot/outputs/datasets/act_ready_slobot_aic
```

### 8.7 Launch training

The launch used:
- prepared local dataset
- center camera only
- W&B enabled
- a unique timestamped output directory

```bash
RUN_TS=$(date +%Y%m%d_%H%M%S)
JOB_NAME="act_slobot_aic_${RUN_TS}"
OUTPUT_DIR="outputs/train/${JOB_NAME}"
WANDB_KEY=$(python3 - <<'PY'
import netrc
print(netrc.netrc().authenticators('api.wandb.ai')[2])
PY
)

nohup env \
  WANDB_API_KEY="$WANDB_KEY" \
  PREPARE_ACT_DATASET=false \
  JOB_NAME="$JOB_NAME" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  ACT_DATASET_ROOT="outputs/datasets/act_ready_slobot_aic" \
  ACT_DATASET_REPO_ID="slobot/aic_act" \
  bash -lc '
    cd /home/ubuntu/lerobot
    export PATH=$HOME/.local/bin:$PATH
    set -a
    source /home/ubuntu/lerobot/.env
    set +a
    /home/ubuntu/lerobot/examples/training/train_act_aic.sh \
      --wandb.project=lerobot-act-aic \
      --batch_size=8 \
      --steps=100000
  ' > "/home/ubuntu/lerobot/logs/${JOB_NAME}.log" 2>&1 &
```

### 8.8 Monitor training

```bash
tail -f /home/ubuntu/lerobot/logs/<job_name>.log
```

```bash
nvidia-smi
```

Checkpoints and outputs land under:

```bash
/home/ubuntu/lerobot/outputs/train/<job_name>
```

### 8.9 Important note about memory

Using all three `1024x1152` cameras caused CUDA OOM on the L40S during the first backward pass.
The working fix was to prepare the ACT-ready dataset with only:

```bash
observation.images.center_camera
```

If you want to try multiple cameras later, start by reducing `--batch_size` and expect much higher memory use.
