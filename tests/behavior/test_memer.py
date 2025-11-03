#!/usr/bin/env python3

"""
Unified test script for BiMan models with synthetic data.
This script can test any model by passing it as a parameter.
It creates fake batches to test the model forward pass and action sampling.
"""

import argparse
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from lerobot.utils.constants import OBS_STATE, ACTION, OBS_IMAGES
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.factory import get_policy_class, make_policy_config
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("awesome-logger")
logger.setLevel(logging.DEBUG)

# Model configurations mapping
MODEL_CONFIGS = {
    "pi0": {
        "state_dim": 14,
        "action_dim": 14,
        "img_height": 224,
        "img_width": 224,
        "cameras": ["top"],  # Single camera
        "n_obs_steps": 1,
        "chunk_size": 50,
    },
    "groot": {
        "state_dim": 14,
        "action_dim": 14,
        "img_height": 224,
        "img_width": 224,
        "cameras": ["top"],  # Single camera
        "n_obs_steps": 1,
        "chunk_size": 50,
    },
    "act": {
        "state_dim": 8,
        "action_dim": 8,
        "img_height": 480,
        "img_width": 640,
        "cameras": ["front", "wrist"],
        "n_obs_steps": 1,
        "chunk_size": 100,
    },
    "diffusion": {
        "state_dim": 8,
        "action_dim": 8,
        "img_height": 480,
        "img_width": 640,
        "cameras": ["front", "wrist"],
        "n_obs_steps": 2,
        "chunk_size": 16,
    },
}


def create_synthetic_batch(
    model_name: str,
    batch_size: int = 2,
    chunk_size: int = 50,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Create a synthetic batch that matches the expected model input format.
    
    Args:
        model_name: Name of the model to create batch for
        batch_size: Number of samples in the batch
        chunk_size: Number of action steps (sequence length)
        device: Device to create tensors on
    
    Returns:
        Dictionary containing synthetic batch data
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    img_height = config["img_height"]
    img_width = config["img_width"]
    cameras = config["cameras"]
    n_obs_steps = config["n_obs_steps"]
    
    batch = {}
    
    # Basic metadata
    batch["episode_index"] = torch.randint(0, 100, (batch_size,), device=device)
    batch["frame_index"] = torch.randint(0, 1000, (batch_size,), device=device)
    batch["timestamp"] = torch.rand(batch_size, device=device) * 10.0
    batch["task_index"] = torch.randint(0, 10, (batch_size,), device=device)
    
    # Language tasks - using sample robot manipulation tasks
    sample_tasks = [
        "pick up the red cube",
        "place the object in the box",
        "grasp the blue cylinder",
        "move the cup to the right",
        "stack the blocks"
    ]
    batch["task"] = [sample_tasks[i % len(sample_tasks)] for i in range(batch_size)]
    
    # Create observations based on n_obs_steps
    if n_obs_steps > 1:
        # Multiple observation steps: (batch_size, n_obs_steps, state_dim)
        batch[OBS_STATE] = torch.randn(batch_size, n_obs_steps, state_dim, device=device) * 0.1
        
        # Camera images: (batch_size, n_obs_steps, 3, height, width) 
        for camera in cameras:
            batch[f"{OBS_IMAGES}.{camera}"] = torch.rand(batch_size, n_obs_steps, 3, img_height, img_width, device=device)
    else:
        # Single observation step: (batch_size, state_dim)
        batch[OBS_STATE] = torch.randn(batch_size, state_dim, device=device) * 0.1
        
        # Camera images: (batch_size, 3, height, width)
        for camera in cameras:
            batch[f"{OBS_IMAGES}.{camera}"] = torch.rand(batch_size, 3, img_height, img_width, device=device)
    
    # Actions: (batch_size, chunk_size, action_dim)
    batch[ACTION] = torch.randn(batch_size, chunk_size, action_dim, device=device) * 0.1
    
    # Optional padding masks for actions (useful for variable length sequences)
    # True means the token is valid, False means it's padding
    batch["action_is_pad"] = torch.zeros(batch_size, chunk_size, dtype=torch.bool, device=device)
    # Let's add some padding to the last few steps for realism
    for i in range(batch_size):
        # Randomly make last 5-10 steps padding
        num_padding = torch.randint(5, 11, (1,)).item()
        batch["action_is_pad"][i, -num_padding:] = True
    
    return batch


def create_dataset_stats(model_name: str, device: str = "cpu") -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create fake dataset statistics for normalization.
    
    Args:
        model_name: Name of the model
        device: Device to create tensors on
    
    Returns:
        Dictionary containing mean, std, min, max statistics
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    cameras = config["cameras"]
    
    stats = {
        OBS_STATE: {
            "mean": torch.zeros(state_dim, device=device),
            "std": torch.ones(state_dim, device=device),
            "min": torch.ones(state_dim, device=device) * -1.0,
            "max": torch.ones(state_dim, device=device) * 1.0
        },
        ACTION: {
            "mean": torch.zeros(action_dim, device=device), 
            "std": torch.ones(action_dim, device=device),
            "min": torch.ones(action_dim, device=device) * -1.0,
            "max": torch.ones(action_dim, device=device) * 1.0
        }
    }
    
    # Add image statistics for each camera
    for camera in cameras:
        # For images, we use per-channel statistics (3 channels for RGB)
        stats[f"{OBS_IMAGES}.{camera}"] = {
            "mean": torch.zeros(3, device=device),
            "std": torch.ones(3, device=device),
            "min": torch.ones(3, device=device) * -1.0,
            "max": torch.ones(3, device=device) * 1.0
        }
    
    return stats


def setup_config_features(config, model_name: str):
    """Set up input and output features for the configuration."""
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[model_name]
    state_dim = model_config["state_dim"]
    action_dim = model_config["action_dim"]
    img_height = model_config["img_height"]
    img_width = model_config["img_width"]
    cameras = model_config["cameras"]
    
    # Set up input features
    input_features = {
        OBS_STATE: PolicyFeature(
            type=FeatureType.STATE,
            shape=(state_dim,)
        )
    }
    
    # Add camera features
    for camera in cameras:
        input_features[f"{OBS_IMAGES}.{camera}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, img_height, img_width)
        )
    
    config.input_features = input_features
    
    # Set up output features
    config.output_features = {
        ACTION: PolicyFeature(
            type=FeatureType.ACTION,
            shape=(action_dim,)
        )
    }
    
    # Adjust config dimensions
    config.max_state_dim = max(getattr(config, 'max_state_dim', 0), state_dim)
    config.max_action_dim = max(getattr(config, 'max_action_dim', 0), action_dim)


def test_model_forward(model_name: str, batch_size: int = 2, device: str = None):
    """Test the model forward pass with synthetic data."""
    
    logger.info(f"🤖 Testing {model_name} model...")
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create configuration
    config = make_policy_config(model_name)
    
    # Set logging if available
    if hasattr(config, 'use_logger'):
        config.use_logger = True
    
    # Get model configuration
    model_config = MODEL_CONFIGS[model_name]
    
    # Set up features properly
    setup_config_features(config, model_name)
    
    logger.info(f"📝 Configuration:")
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - Chunk size: {getattr(config, 'chunk_size', 'N/A')}")
    logger.info(f"  - Max state dim: {config.max_state_dim}")
    logger.info(f"  - Max action dim: {config.max_action_dim}")
    logger.info(f"  - Image resize: {getattr(config, 'resize_imgs_with_padding', 'N/A')}")
    logger.info(f"  - Input features: {list(config.input_features.keys())}")
    logger.info(f"  - Output features: {list(config.output_features.keys())}")
    if hasattr(config, 'image_features'):
        logger.info(f"  - Image features: {list(config.image_features.keys())}")
    
    # Create synthetic dataset stats
    dataset_stats = create_dataset_stats(model_name, device)
    logger.info(f"📊 Created synthetic dataset stats")
    
    # Create model
    logger.info(f"🏗️  Creating {model_name} model...")
    try:
        policy_cls = get_policy_class(model_name)
        # New API: only pass config, not dataset_stats
        policy = policy_cls(config=config)
        policy = policy.to(device)
        policy.eval()  # Set to evaluation mode
        logger.info(f"✅ Model created successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info(f"📈 Model parameters:")
        logger.info(f"  - Total: {total_params:,}")
        logger.info(f"  - Trainable: {trainable_params:,}")
        
    except Exception as e:
        logger.info(f"❌ Error creating model: {e}")
        raise
    
    # Create synthetic batch
    logger.info(f"\n🎲 Creating synthetic batch (batch_size={batch_size})...")
    
    try:
        # Get chunk_size from config or model config
        chunk_size = model_config.get("chunk_size", getattr(config, 'chunk_size', 50))
            
        batch = create_synthetic_batch(
            model_name=model_name,
            batch_size=batch_size,
            chunk_size=chunk_size,
            device=device
        )
        
        logger.info(f"📦 Batch created with keys: {list(batch.keys())}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  - {key}: {value.shape} {value.dtype}")
            else:
                logger.info(f"  - {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
                
    except Exception as e:
        logger.info(f"❌ Error creating batch: {e}")
        raise
    
    # Test forward pass (training mode)
    logger.info(f"\n🔄 Testing training forward pass...")
    try:
        policy.train()
        
        with torch.no_grad():  # We don't need gradients for testing
            loss, loss_dict = policy.forward(batch)
            
        logger.info(f"✅ Training forward pass successful!")
        logger.info(f"📉 Loss: {loss:.6f}")
        if loss_dict is not None:
            logger.info(f"📊 Loss dict keys: {list(loss_dict.keys())}")
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  - {key}: {value.shape if hasattr(value, 'shape') else value}")
                else:
                    logger.info(f"  - {key}: {value}")
        else:
            logger.info(f"📊 Loss dict: None (model returned no additional loss information)")
                
    except Exception as e:
        logger.info(f"❌ Error in training forward pass: {e}")
        raise
    
    # Test inference (action selection)
    logger.info(f"\n🎯 Testing inference (action selection)...")
    try:
        policy.eval()
        policy.reset()  # Reset internal queues
        
        # Get n_obs_steps from model config
        n_obs_steps = model_config.get("n_obs_steps", 1)
        cameras = model_config["cameras"]
        
        if n_obs_steps > 1:
            # For models with multiple observation steps, simulate queue-based inference
            # by sending observations one by one
            for obs_step in range(n_obs_steps):
                # Create single timestep observation (no temporal dimension)
                step_batch = {}
                step_batch[OBS_STATE] = batch[OBS_STATE][:1, obs_step]  # (1, state_dim)
                for camera in cameras:
                    step_batch[f"{OBS_IMAGES}.{camera}"] = batch[f"{OBS_IMAGES}.{camera}"][:1, obs_step]  # (1, 3, H, W)
                step_batch["task"] = [batch["task"][0]]
                
                with torch.no_grad():
                    action = policy.select_action(step_batch)
        else:
            # For single observation models, use simpler approach
            inference_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and key not in [ACTION, "action_is_pad"]:
                    inference_batch[key] = value[:1]  # First sample only
                elif key == "task":
                    inference_batch[key] = [value[0]]  # First task only
            
            with torch.no_grad():
                action = policy.select_action(inference_batch)
            
        logger.info(f"✅ Inference successful!")
        logger.info(f"🎯 Generated action shape: {action.shape}")
        logger.info(f"🎯 Action values (first 5): {action.flatten()[:5]}")
        
    except Exception as e:
        logger.info(f"❌ Error in inference: {e}")
        raise
    
    logger.info(f"\n🎉 All tests passed! {model_name} model is working correctly.")
    return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified test script for BiMan models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models: {', '.join(MODEL_CONFIGS.keys())}

Examples:
  python test_model.py --model pi0
  python test_model.py --model smolvla --batch-size 4
  python test_model.py --model zero_ki_realtime --device cuda
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=False,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to test"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=2,
        help="Batch size for testing (default: 2)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: auto-detect)"
    )
    
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="List all available models and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model, config in MODEL_CONFIGS.items():
            cameras_str = ", ".join(config["cameras"])
            print(f"  - {model}: {config['state_dim']}D state, {config['action_dim']}D action, "
                  f"{config['img_height']}x{config['img_width']} images, cameras: {cameras_str}, "
                  f"n_obs_steps: {config['n_obs_steps']}, chunk_size: {config['chunk_size']}")
        return
    
    if not args.model:
        parser.error("--model is required unless using --list-models")
    
    logger.info("🚀 Starting unified model test...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device or 'auto-detect'}")
    
    try:
        test_model_forward(
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device
        )
        logger.info(f"\n🎊 Test completed successfully!")
        
    except Exception as e:
        logger.info(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()