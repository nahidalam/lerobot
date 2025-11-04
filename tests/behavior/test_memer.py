import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import logging
import torch
import numpy as np
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.memer.configuration_pi05 import PI05MemerConfig
from lerobot.policies.memer.modeling_pi05 import PI05MemerPolicy
from lerobot.policies.memer.processor_pi05 import make_pi05_memer_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_batch(batch: dict):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            details = f"{key}: {tuple(value.shape)} | {value.dtype}"
            if value.is_cuda:
                details += " | cuda"
            if key.endswith("attention_mask"):
                details += f" | sum={value.sum().item()}"
            if "images" in key:
                details += f" | min={value.min().item():.3f} max={value.max().item():.3f}"
            print(details)
        elif isinstance(value, (list, tuple)):
            print(f"{key}: list(len={len(value)})")
        else: 
            print(f"{key}: {value}")

def simulate_environment_inference(
    dataset: LeRobotDataset,
    policy: PI05MemerPolicy,
    preprocessor,
    general_task: str,
    num_steps: int = 100,
    start_episode: int = 0
):
    policy.reset()
    preprocessor.reset()
    
    # Get episode boundaries from metadata
    start_idx = dataset.meta.episodes[start_episode]["dataset_from_index"]
    end_idx = dataset.meta.episodes[start_episode]["dataset_to_index"]
    available_steps = end_idx - start_idx
    num_steps = min(num_steps, available_steps)
    
    logger.info(f"Starting simulation: {num_steps} steps from episode {start_episode}")
    
    for step in range(num_steps):
        frame_idx = start_idx + step
        # Get frame from dataset
        frame = dataset[frame_idx]
        
        # Prepare observation in the same way as the real inference pipeline.
        observation_np = {}
        for key, value in frame.items():
            if not key.startswith("observation."):
                continue

            if "_is_pad" in key:
                continue

            if not isinstance(value, torch.Tensor):
                observation_np[key] = np.asarray(value)
                continue

            tensor = value.detach().cpu()

            if key.startswith("observation.images"):
                arr = tensor.numpy()
                if arr.ndim == 4:
                    arr = arr.squeeze(0)
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))
                # Convert to uint8 in [0, 255] as expected by prepare_observation_for_inference
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255)
                arr = arr.astype(np.uint8)
            else:
                arr = tensor.numpy().astype(np.float32)

            observation_np[key] = arr

        device = torch.device(policy.config.device)
        observation_for_policy = prepare_observation_for_inference(
            observation_np,
            device=device,
            task=general_task,
            robot_type=None,
        )

        # Apply preprocessor to tokenize/normalize inputs. This mirrors lerobot_record.
        processed_batch = preprocessor(observation_for_policy)
        
        print_batch(processed_batch)
        
        is_query_step = (step > 0) and (step % policy.config.high_level_query_interval == 0)
        if is_query_step:
            logger.info(f"\n--- Step {step}/{num_steps} (Frame {frame_idx}) ---")
            logger.info(f"🔍 HIGH-LEVEL QUERY STEP")
        
        with torch.no_grad():
            action = policy.select_action(processed_batch)
        
        if policy.high_level_policy is not None and is_query_step:
            logger.info(f"Memory:")
            logger.info(f"  Recent frames: {len(policy._recent_frames)}/{policy.config.recent_frames_window}")
            logger.info(f"  Keyframes: {len(policy._keyframes)}/{policy.config.max_keyframes}")
            if len(policy._keyframes) > 0:
                keyframe_indices = [kf[0] for kf in policy._keyframes]
                logger.info(f"  Keyframe indices: {keyframe_indices}")
            if policy._current_subtask is not None:
                logger.info(f"  Current subtask: '{policy._current_subtask}'")
            else:
                logger.info(f"  Current subtask: None (using general task)")
    
    if policy.high_level_policy is not None:
        logger.info("\n" + "="*80)
        logger.info("FINAL STATISTICS")
        logger.info("="*80)
        logger.info(f"Total high-level queries: {len(policy._candidate_frame_lists)}")
        logger.info(f"Final keyframes: {len(policy._keyframes)}")
        if len(policy._keyframes) > 0:
            keyframe_indices = [kf[0] for kf in policy._keyframes]
            logger.info(f"Keyframe indices: {keyframe_indices}")
        logger.info(f"Final subtask: '{policy._current_subtask}'")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test MEMER high-level policy inference by simulating environment interaction with dataset frames."
    )
    
    # Dataset and task configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="NONHUMAN-RESEARCH/SARM-DATASET-TEST",
        help="HuggingFace dataset repository ID (default: NONHUMAN-RESEARCH/SARM-DATASET-TEST)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Complete the task shown in the demonstration",
        help="General task description for the episode"
    )
    
    # Simulation parameters
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of steps to simulate (default: 50)"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Which episode to start from in the dataset (default: 0)"
    )
    
    # High-level policy settings
    parser.add_argument(
        "--high-level-policy",
        dest="use_high_level_policy",
        action="store_true",
        default=True,
        help="Enable high-level policy for MEMER (default: disabled)"
    )
    parser.add_argument(
        "--no-high-level-policy",
        dest="use_high_level_policy",
        action="store_false",
        help="Disable high-level policy (test without MEMER)"
    )
    parser.add_argument(
        "--query-interval",
        type=int,
        default=5,
        help="Query high-level policy every N steps (default: 5)"
    )
    parser.add_argument(
        "--recent-frames-window",
        type=int,
        default=10,
        help="Number of recent frames to keep (default: 10)"
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=8,
        help="Maximum number of keyframes to maintain (default: 8)"
    )
    parser.add_argument(
        "--keyframe-distance-threshold",
        type=int,
        default=10,
        help="Distance threshold for keyframe clustering (default: 10)"
    )
    
    # Camera settings for high-level policy
    parser.add_argument(
        "--camera-key",
        type=str,
        default="observation.images.camera_0",
        help="Camera key for high-level policy (default: observation.images.camera_0)"
    )
    
    return parser.parse_args()


def main():
    """Main test function."""
    
    # Parse command line arguments
    args = parse_args()
    
    # Configuration from arguments
    dataset_repo_id = args.dataset
    general_task = args.task
    num_simulation_steps = args.num_steps
    start_episode = args.episode
    
    # High-level policy settings
    use_high_level_policy = args.use_high_level_policy
    high_level_query_interval = args.query_interval
    
    logger.info(f"Dataset: {dataset_repo_id}")
    logger.info(f"Task: {general_task}")
    logger.info(f"Steps: {num_simulation_steps}")
    logger.info(f"Episode: {start_episode}")
    logger.info(f"High-level policy: {'Enabled' if use_high_level_policy else 'Disabled'}")
    if use_high_level_policy:
        logger.info(f"  Camera key: {args.camera_key}")
        logger.info(f"  Query interval: {high_level_query_interval}")
        logger.info(f"  Recent frames window: {args.recent_frames_window}")
        logger.info(f"  Max keyframes: {args.max_keyframes}")
        logger.info(f"  Distance threshold: {args.keyframe_distance_threshold}")
    logger.info("="*80 + "\n")
    
    dataset = LeRobotDataset(repo_id=dataset_repo_id)
    logger.info(f"Dataset loaded: {dataset.repo_id} ({dataset.num_episodes} episodes, {len(dataset)} frames)")
    
    # Extract features from dataset
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy_config = PI05MemerConfig(
        use_high_level_policy=use_high_level_policy,
        high_level_query_interval=high_level_query_interval,
        recent_frames_window=args.recent_frames_window,
        max_keyframes=args.max_keyframes,
        keyframe_distance_threshold=args.keyframe_distance_threshold,
        high_level_policy_name="Qwen/Qwen3-VL-8B-Instruct",
        camera_key_high_level_policy=args.camera_key,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="float32",
        chunk_size=50,
        n_action_steps=50,
        device=device,
        input_features=input_features,
        output_features=output_features,
    )
    
    policy = PI05MemerPolicy(config=policy_config)
    policy.eval()
    
    # Create preprocessor and postprocessor (needed to tokenize language input)
    preprocessor, postprocessor = make_pi05_memer_pre_post_processors(
        config=policy_config,
        dataset_stats=dataset.meta.stats,
    )
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING INFERENCE SIMULATION")
    logger.info("="*80 + "\n")
    
    simulate_environment_inference(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        general_task=general_task,
        num_steps=num_simulation_steps,
        start_episode=start_episode,
    )


if __name__ == "__main__":
    main()