import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from PIL import Image
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.memer.configuration_pi05 import PI05MemerConfig
from lerobot.policies.memer.processor_pi05 import make_pi05_memer_pre_post_processors


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor image to PIL Image (same as in high_level_policy.py)."""
    # Handle different tensor formats
    if tensor.ndim == 3:
        # Check if channels first or channels last
        if tensor.shape[0] == 3:  # Channels first (C, H, W)
            tensor = tensor.permute(1, 2, 0)  # Convert to (H, W, C)
    
    # Convert to numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    img_np = tensor.numpy()
    
    # Normalize to [0, 255]
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype('uint8')
    elif img_np.min() < 0:  # [-1, 1] range
        img_np = ((img_np + 1) * 127.5).astype('uint8')
    else:
        img_np = img_np.astype('uint8')
    
    return Image.fromarray(img_np)


def main():
    # Load dataset
    dataset = LeRobotDataset(repo_id="NONHUMAN-RESEARCH/SARM-DATASET-TEST")
    print(f"Dataset loaded: {dataset.num_episodes} episodes, {len(dataset)} frames")
    
    # Get single frame
    frame = dataset[0]
    task = frame.get("task", "Complete the task shown in the demonstration")
    
    # Extract features
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # Create config (minimal, no model needed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PI05MemerConfig(
        use_high_level_policy=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="float32",
        chunk_size=50,
        n_action_steps=50,
        device=device,
        input_features=input_features,
        output_features=output_features,
    )
    
    # Create only the preprocessor
    preprocessor, _ = make_pi05_memer_pre_post_processors(
        config=config,
        dataset_stats=dataset.meta.stats,
    )
    
    # Build observation from frame
    observation = {}
    for key in frame.keys():
        if key.startswith("observation."):
            observation[key] = frame[key].to(device)
    observation["task"] = task if isinstance(task, str) else str(task)
    
    # Apply preprocessor
    processed_batch = preprocessor(observation)
    
    # Print batch info
    print("\n=== Processed Batch ===")
    for key, value in processed_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {tuple(value.shape)} | {value.dtype} | min={value.min().item():.3f} max={value.max().item():.3f}")
        else:
            print(f"{key}: {type(value)}")
    
    # Test tensor to PIL conversion for each camera
    camera_keys = [k for k in processed_batch.keys() if k.startswith("observation.images.")]
    
    print(f"\n=== Converting {len(camera_keys)} cameras to PIL ===")
    for cam_key in camera_keys:
        # Get tensor (remove batch dimension)
        img_tensor = processed_batch[cam_key][0]  # Shape: (C, H, W)
        print(f"\n{cam_key}:")
        print(f"  Tensor shape: {img_tensor.shape}")
        print(f"  Tensor range: [{img_tensor.min().item():.3f}, {img_tensor.max().item():.3f}]")
        
        # Convert to PIL
        pil_image = tensor_to_pil(img_tensor)
        print(f"  PIL size: {pil_image.size}")
        print(f"  PIL mode: {pil_image.mode}")
        
        # Save image
        cam_name = cam_key.split('.')[-1]
        output_path = f"test_{cam_name}.png"
        pil_image.save(output_path)
        print(f"  Saved to: {output_path}")
    
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()

