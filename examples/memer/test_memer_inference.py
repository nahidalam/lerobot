#!/usr/bin/env python3

"""
Example script demonstrating MEMER (Memory-based Robot) inference.

This script shows how to:
1. Enable the high-level policy for hierarchical task decomposition
2. Use visual memory with keyframes
3. Let the VLM generate subtasks automatically

MEMER decomposes the policy as:
π(A_t | o_{0:t}) = π_l(A_t | I_t, q_t, l'_t) * π_h(l'_t, J_t | I_{t-N+1:t}, K_t)

Where:
- π_l: Low-level policy (PI0.5) that executes actions
- π_h: High-level policy (VLM) that generates subtasks
- l'_t: Subtask (e.g., "look in left bin")
- J_t: Candidate frames selected by high-level policy
- K_t: Keyframes built from clustering candidate frames
"""

import torch
import logging
from lerobot.policies.memer.configuration_pi05 import PI05MemerConfig
from lerobot.policies.memer.modeling_pi05 import PI05MemerPolicy
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.utils.constants import OBS_STATE, ACTION, OBS_IMAGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_config(use_memer: bool = False):
    """Create a simple configuration for testing."""
    config = PI05MemerConfig()
    
    # Basic settings
    config.max_state_dim = 14
    config.max_action_dim = 14
    config.chunk_size = 50
    config.n_action_steps = 50
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # MEMER-specific settings
    config.use_high_level_policy = use_memer
    if use_memer:
        config.high_level_policy_name = "Qwen/Qwen2-VL-7B-Instruct"  # Can be changed
        config.recent_frames_window = 10  # Keep last 10 frames
        config.max_keyframes = 8  # Up to 8 keyframes
        config.keyframe_distance_threshold = 10  # Clustering threshold
        config.high_level_query_interval = 5  # Query VLM every 5 steps
    
    # Define input features
    config.input_features = {
        OBS_STATE: PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,)
        ),
        f"{OBS_IMAGES}.top": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224)
        )
    }
    
    # Define output features
    config.output_features = {
        ACTION: PolicyFeature(
            type=FeatureType.ACTION,
            shape=(14,)
        )
    }
    
    return config


def create_synthetic_observation(device: str = "cpu"):
    """Create a synthetic observation for testing."""
    batch = {
        OBS_STATE: torch.randn(1, 14, device=device) * 0.1,
        f"{OBS_IMAGES}.top": torch.rand(1, 3, 224, 224, device=device),
        "task": ["search for ketchup in the bins"],  # High-level task
    }
    return batch


def run_memer_inference_example():
    """Run a complete example of MEMER inference."""
    
    logger.info("=" * 80)
    logger.info("MEMER Inference Example")
    logger.info("=" * 80)
    
    # Create configuration with MEMER enabled
    config = create_simple_config(use_memer=True)
    logger.info(f"\n📋 Configuration:")
    logger.info(f"  - Use MEMER: {config.use_high_level_policy}")
    logger.info(f"  - High-level policy: {config.high_level_policy_name}")
    logger.info(f"  - Recent frames window: {config.recent_frames_window}")
    logger.info(f"  - Max keyframes: {config.max_keyframes}")
    logger.info(f"  - Query interval: {config.high_level_query_interval} steps")
    
    # Create policy
    logger.info(f"\n🏗️  Creating PI05MemerPolicy with MEMER...")
    policy = PI05MemerPolicy(config=config)
    policy = policy.to(config.device)
    policy.eval()
    
    logger.info(f"✅ Policy created successfully!")
    logger.info(f"   - High-level policy enabled: {policy.high_level_policy is not None}")
    
    # Reset policy (initializes memory structures)
    policy.reset()
    logger.info(f"\n🔄 Policy reset complete")
    
    # Simulate multiple steps
    num_steps = 15
    logger.info(f"\n🚀 Running {num_steps} inference steps...")
    logger.info("=" * 80)
    
    for step in range(num_steps):
        logger.info(f"\nStep {step + 1}/{num_steps}")
        logger.info("-" * 40)
        
        # Create observation
        batch = create_synthetic_observation(device=config.device)
        
        # Select action (this will update MEMER memory internally)
        with torch.no_grad():
            action = policy.select_action(batch)
        
        logger.info(f"  ✓ Action selected: shape={action.shape}")
        logger.info(f"  ✓ Action values (first 5): {action.flatten()[:5].cpu().numpy()}")
        
        # Show memory state if MEMER is active
        if policy.high_level_policy is not None:
            logger.info(f"  📊 Memory state:")
            logger.info(f"     - Recent frames: {len(policy._recent_frames)}/{config.recent_frames_window}")
            logger.info(f"     - Keyframes: {len(policy._keyframes)}/{config.max_keyframes}")
            if policy._current_subtask:
                logger.info(f"     - Current subtask: '{policy._current_subtask}'")
            else:
                logger.info(f"     - Current subtask: (not set yet)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ MEMER inference example completed successfully!")
    logger.info("=" * 80)


def compare_with_and_without_memer():
    """Compare inference with and without MEMER."""
    
    logger.info("\n\n" + "=" * 80)
    logger.info("Comparison: PI0.5 vs PI0.5 + MEMER")
    logger.info("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test without MEMER
    logger.info("\n1️⃣  Testing PI0.5 (baseline)...")
    config_baseline = create_simple_config(use_memer=False)
    policy_baseline = PI05MemerPolicy(config=config_baseline)
    policy_baseline = policy_baseline.to(device)
    policy_baseline.eval()
    policy_baseline.reset()
    
    batch = create_synthetic_observation(device=device)
    with torch.no_grad():
        action_baseline = policy_baseline.select_action(batch)
    
    logger.info(f"   ✓ Baseline action: {action_baseline.shape}")
    
    # Test with MEMER
    logger.info("\n2️⃣  Testing PI0.5 + MEMER...")
    config_memer = create_simple_config(use_memer=True)
    policy_memer = PI05MemerPolicy(config=config_memer)
    policy_memer = policy_memer.to(device)
    policy_memer.eval()
    policy_memer.reset()
    
    with torch.no_grad():
        action_memer = policy_memer.select_action(batch)
    
    logger.info(f"   ✓ MEMER action: {action_memer.shape}")
    
    logger.info("\n📊 Results:")
    logger.info(f"   - Both policies produce actions of shape: {action_baseline.shape}")
    logger.info(f"   - MEMER adds hierarchical task decomposition")
    logger.info(f"   - MEMER maintains visual memory across timesteps")
    logger.info(f"   - High-level policy queries VLM every {config_memer.high_level_query_interval} steps")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    # Note: This example requires the Qwen2-VL model to be available
    # If the model is not found, the high-level policy will fall back to a dummy implementation
    
    logger.info("""
╔══════════════════════════════════════════════════════════════════════════╗
║                     MEMER Inference Example                              ║
║                                                                          ║
║  MEMER: Memory-based Embodied Robot                                     ║
║  Hierarchical policy with visual memory and VLM-based task decomposition║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run main example
        run_memer_inference_example()
        
        # Run comparison
        compare_with_and_without_memer()
        
    except Exception as e:
        logger.error(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    logger.info("\n✨ All examples completed successfully!")

