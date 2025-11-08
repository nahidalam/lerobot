import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import logging
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.memer.configuration_pi05 import PI05MemerConfig
from lerobot.policies.memer.modeling_pi05 import PI05MemerPolicy
from lerobot.policies.memer.processor_pi05 import make_pi05_memer_pre_post_processors


# Configure logging with force=True to override any previous configurations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Python 3.8+ - forces reconfiguration
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure this specific logger is set to INFO

# Also set the root logger to INFO to ensure all messages are shown
logging.getLogger().setLevel(logging.INFO)


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


def save_main_camera_frame(observation: dict, output_dir: Path, step: int, camera_key: str, subtask: str = None):
    """
    Save only the main camera image from an observation dict with subtask overlay.
    
    Args:
        observation: Dict containing observation.images.* keys with tensors
        output_dir: Directory to save images
        step: Current step number
        camera_key: Key for the main camera (e.g., 'observation.images.cam_high')
        subtask: Current subtask to display on the image
    """
    if camera_key not in observation:
        logger.warning(f"Camera key {camera_key} not found in observation")
        return False
    
    # Get the camera tensor
    image_tensor = observation[camera_key].cpu().clamp(0, 1)
    
    # Convert tensor to PIL Image
    # Tensor is in format (C, H, W) with values in [0, 1]
    pil_image = TF.to_pil_image(image_tensor)
    
    # Add subtask text overlay if provided
    if subtask:
        # Create a drawing context
        draw = ImageDraw.Draw(pil_image)
        
        # Try to use a good font, fallback to default if not available
        try:
            # Try different common font paths
            font_size = max(20, int(pil_image.height * 0.04))  # Scale font with image size
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Prepare text
        text = f"Subtask: {subtask}"
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position at bottom center with padding
        padding = 10
        x = (pil_image.width - text_width) // 2
        y = pil_image.height - text_height - padding * 2
        
        # Draw black background rectangle for better visibility
        rect_coords = [
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        ]
        draw.rectangle(rect_coords, fill=(0, 0, 0, 180))
        
        # Draw white text on top
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    # Save image directly in output_dir with step number
    image_path = output_dir / f"frame_{step:04d}.png"
    pil_image.save(image_path)
    
    return True

def simulate_environment_inference(
    dataset: LeRobotDataset,
    policy: PI05MemerPolicy,
    preprocessor,
    general_task: str,
    num_steps: int = 100,
    start_episode: int = 0,
    output_dir: Path = None,
    camera_key: str = "observation.images.cam_high",
):
    policy.reset()
    preprocessor.reset()
    
    # Get episode boundaries from metadata
    start_idx = dataset.meta.episodes[start_episode]["dataset_from_index"]
    end_idx = dataset.meta.episodes[start_episode]["dataset_to_index"]
    available_steps = end_idx - start_idx
    num_steps = min(num_steps, available_steps)
    
    # Use the same sampling interval as recent_frames for saving
    query_interval = policy.config.high_level_query_interval
    recent_frame_interval = policy.config.recent_frames_sampling_interval
    
    logger.info(f"Starting simulation: {num_steps} steps from episode {start_episode}")
    if output_dir:
        logger.info(f"Saving frames to: {output_dir}")
        logger.info(f"  Camera: {camera_key}")
        logger.info(f"  Saving every {recent_frame_interval} steps (same as recent_frames sampling)")
    
    for step in range(num_steps):
        frame_idx = start_idx + step
        # Get frame from dataset
        frame = dataset[frame_idx]
        
        # Extract task from frame
        task_from_frame = frame.get("task", general_task)
        if isinstance(task_from_frame, torch.Tensor):
            # If task is somehow a tensor, try to decode it
            task_from_frame = task_from_frame.item() if task_from_frame.numel() == 1 else general_task
        elif not isinstance(task_from_frame, str):
            task_from_frame = str(task_from_frame)
        
        logger.info(f"\n--- Step {step}/{num_steps} (Frame {frame_idx}) ---")
        
        device = torch.device(policy.config.device)
        
        # Build observation dict directly from dataset frame
        # Dataset frames already have tensors in the correct format (C, H, W), [0, 1]
        # No need to convert to numpy and back!
        observation = {}
        for key in frame.keys():
            if key.startswith("observation."):
                # Move tensor to correct device
                observation[key] = frame[key].to(device)
        
        # Add task to the observation dict (preprocessor expects it here)
        observation["task"] = task_from_frame
        
        # Apply preprocessor to tokenize/normalize inputs
        # The preprocessor expects a dict with observations and task
        processed_batch = preprocessor(observation)
        
        # Check if this is a high-level query step
        is_query_step = (step > 0) and (step % query_interval == 0)
        is_recent_frame_step = (step % recent_frame_interval == 0)
        
        # Get current subtask (if high-level policy is enabled)
        current_subtask = None
        if policy.high_level_policy is not None:
            current_subtask = policy._current_subtask if hasattr(policy, '_current_subtask') and policy._current_subtask else task_from_frame
        else:
            current_subtask = task_from_frame
        
        # Save recent frames with subtask overlay (using same sampling as recent_frames)
        if output_dir and is_recent_frame_step:
            saved = save_main_camera_frame(observation, output_dir, step, camera_key, subtask=current_subtask)
            if saved:
                logger.info(f"📸 Saved frame {step} with subtask: '{current_subtask}'")
        
        if is_query_step:
            logger.info(f"🔍 HIGH-LEVEL QUERY STEP")
        
        # Run inference to get action
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
        logger.info("FINAL STATISTICS")
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
        default=1100,
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
        "--recent-frames-sampling-interval",
        type=int,
        default=25,
        help="How often to add frames to recent_frames, e.g., 5 = every 5th frame (default: 5)"
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
        default=5,
        help="Distance threshold for keyframe clustering (default: 10)"
    )
    
    # Camera settings for high-level policy
    parser.add_argument(
        "--camera-key",
        type=str,
        default="observation.images.cam_high",
        help="Camera key for high-level policy (default: observation.images.cam_high)"
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Vision-Language Model for high-level policy (default: Qwen/Qwen2.5-VL-7B-Instruct, also supports Qwen/Qwen3-VL-8B-Instruct)"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/frames",
        help="Directory to save frames (if not provided, frames won't be saved)"
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
        logger.info(f"  VLM model: {args.vlm_model}")
        logger.info(f"  Camera key: {args.camera_key}")
        logger.info(f"  Query interval: {high_level_query_interval}")
        logger.info(f"  Recent frames window: {args.recent_frames_window}")
        logger.info(f"  Recent frames sampling: every {args.recent_frames_sampling_interval} frames")
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
        recent_frames_sampling_interval=args.recent_frames_sampling_interval,
        max_keyframes=args.max_keyframes,
        keyframe_distance_threshold=args.keyframe_distance_threshold,
        high_level_policy_name=args.vlm_model,
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
    
    # Setup output directory if provided
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
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
        output_dir=output_dir,
        camera_key=args.camera_key,
    )


if __name__ == "__main__":
    main()