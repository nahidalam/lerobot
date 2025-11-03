#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from lerobot.policies.memer.configuration_pi05 import PI05MemerConfig


class HighLevelPolicy:
    """
    High-level policy for MEMER that generates subtasks and selects candidate keyframes.
    
    The high-level policy π_h(l'_t, J_t | R_t, K_t, l_t) takes:
    - R_t: Recent frames (last N frames)
    - K_t: Current keyframes (up to 8)
    - l_t: High-level task description
    
    And outputs:
    - l'_t: Subtask description
    - J_t: Candidate frames (subset of recent frames)
    """

    def __init__(self, config: PI05MemerConfig):
        """
        Initialize the high-level policy with Qwen3-VL model.
        
        Args:
            config: Configuration object containing model settings
        """
        self.config = config
        self.device = config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu"
        
        logging.info(f"Initializing HighLevelPolicy with model: {config.high_level_policy_name}")
        
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            
            # Load the VLM model for high-level planning
            # Using torch.bfloat16 for better performance (recommended by Qwen3-VL)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.high_level_policy_name,
                dtype="auto",
                device_map="auto"
            )
            
            # Optional: Enable flash_attention_2 for better acceleration and memory saving
            # Especially useful in multi-image and video scenarios
            # self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            #     config.high_level_policy_name,
            #     dtype=torch.bfloat16,
            #     attn_implementation="flash_attention_2",
            #     device_map="auto",
            # )
            
            self.processor = AutoProcessor.from_pretrained(config.high_level_policy_name)
            
            logging.info("HighLevelPolicy model loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load Qwen3-VL model: {e}. Using dummy implementation.")
            self.model = None
            self.processor = None

    def generate_subtasks_and_key_frames(
        self,
        general_task: str,
        recent_frames: List[torch.Tensor],
        keyframes: List[torch.Tensor],
        frame_indices: List[int],
    ) -> Tuple[str, List[int]]:
        """
        Generate subtask and candidate frame indices for the next step.
        
        Args:
            general_task: High-level task description (e.g., "search for ketchup")
            recent_frames: List of recent frame tensors (N frames)
            keyframes: List of keyframe tensors (up to 8 frames)
            frame_indices: Indices corresponding to recent_frames
            
        Returns:
            subtask: Generated subtask description (e.g., "look in left bin")
            candidate_indices: List of frame indices selected as candidates from recent_frames
        """
        if self.model is None or self.processor is None:
            # Dummy implementation for when model is not available
            logging.warning("Using dummy high-level policy (model not loaded)")
            subtask = general_task  # Just return the general task
            # Select middle frames as candidates
            num_candidates = min(3, len(recent_frames))
            step = max(1, len(recent_frames) // num_candidates)
            candidate_indices = frame_indices[::step][:num_candidates]
            return subtask, candidate_indices
        
        # Prepare the prompt for the VLM
        prompt = self._create_prompt(general_task, len(recent_frames), len(keyframes))
        
        # Prepare images for the model
        images = []
        
        # Add keyframes first
        for kf in keyframes:
            images.append(self._tensor_to_pil(kf))
        
        # Add recent frames
        for rf in recent_frames:
            images.append(self._tensor_to_pil(rf))
        
        # Create messages for the model
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Generate response
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
            
            # Parse the output to extract subtask and candidate frame indices
            subtask, candidate_indices = self._parse_model_output(
                output_text, frame_indices, len(keyframes)
            )
            
            return subtask, candidate_indices
            
        except Exception as e:
            logging.warning(f"Error generating subtask with VLM: {e}. Using fallback.")
            # Fallback: return general task and select evenly spaced frames
            subtask = general_task
            num_candidates = min(3, len(recent_frames))
            step = max(1, len(recent_frames) // num_candidates)
            candidate_indices = frame_indices[::step][:num_candidates]
            return subtask, candidate_indices

    def _create_prompt(self, general_task: str, num_recent: int, num_keyframes: int) -> str:
        """
        Create the prompt for the VLM to generate subtask and select candidate frames.
        
        Args:
            general_task: The high-level task
            num_recent: Number of recent frames provided
            num_keyframes: Number of keyframes provided
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a robot vision assistant helping to break down tasks into subtasks and identify important frames.

High-level task: {general_task}

I'm providing you with:
- {num_keyframes} keyframes from previous steps (shown first)
- {num_recent} recent frames from the current observation window (shown after keyframes)

Please analyze these frames and provide:
1. The next subtask that should be executed to progress towards the goal
2. Which of the recent frames (numbered 1 to {num_recent}) are most relevant for completing this subtask

Format your response as a JSON object with the following structure:
{{"current_subtask": "your subtask description", "keyframe_positions": [list of frame numbers]}}

Example response:
{{"current_subtask": "look in the left bin", "keyframe_positions": [1, 3, 5]}}

IMPORTANT: Respond ONLY with the JSON object, no additional text.
"""
        return prompt

    def _parse_model_output(
        self, output: str, frame_indices: List[int], num_keyframes: int
    ) -> Tuple[str, List[int]]:
        """
        Parse the VLM output to extract subtask and candidate frame indices.
        
        The output is expected to be in JSON format:
        {"current_subtask": "description", "keyframe_positions": [1, 3, 5]}
        
        Args:
            output: Raw text output from the VLM (JSON string)
            frame_indices: Available frame indices for recent frames
            num_keyframes: Number of keyframes (to offset the numbering)
            
        Returns:
            subtask: Extracted subtask description
            candidate_indices: List of selected frame indices
        """
        subtask = ""
        candidate_positions = []
        
        try:
            # Try to parse as JSON
            # First, clean the output - remove any markdown code blocks if present
            cleaned_output = output.strip()
            if cleaned_output.startswith("```"):
                # Remove markdown code blocks
                lines = cleaned_output.split("\n")
                # Find the actual JSON content
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or not line.strip().startswith("```"):
                        json_lines.append(line)
                cleaned_output = "\n".join(json_lines).strip()
            
            # Try to find JSON object in the output
            # Look for the first { and last }
            start_idx = cleaned_output.find("{")
            end_idx = cleaned_output.rfind("}")
            
            if start_idx != -1 and end_idx != -1:
                json_str = cleaned_output[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                
                # Extract subtask
                subtask = parsed.get("current_subtask", "")
                
                # Extract keyframe positions
                candidate_positions = parsed.get("keyframe_positions", [])
                
                # Ensure positions are integers
                candidate_positions = [int(pos) for pos in candidate_positions]
                
                logging.info(f"Successfully parsed JSON: subtask='{subtask}', positions={candidate_positions}")
            else:
                logging.warning("Could not find JSON object in output")
                
        except json.JSONDecodeError as e:
            logging.warning(f"Could not parse JSON output: {e}")
            logging.debug(f"Output was: {output}")
        except Exception as e:
            logging.warning(f"Error parsing model output: {e}")
            logging.debug(f"Output was: {output}")
        
        # Convert 1-indexed positions to actual frame indices
        candidate_indices = []
        for pos in candidate_positions:
            # Positions are 1-indexed and refer to recent frames only
            if 1 <= pos <= len(frame_indices):
                candidate_indices.append(frame_indices[pos - 1])
        
        # If no valid candidates found, select default frames
        if not candidate_indices:
            logging.info("No valid candidates found, using default selection")
            num_candidates = min(3, len(frame_indices))
            step = max(1, len(frame_indices) // num_candidates)
            candidate_indices = frame_indices[::step][:num_candidates]
        
        # If no subtask found, use a generic one
        if not subtask:
            subtask = "continue task"
        
        return subtask, candidate_indices

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor image to PIL Image for the VLM.
        
        Args:
            tensor: Image tensor of shape (C, H, W) or (H, W, C), normalized to [0, 1] or [-1, 1]
            
        Returns:
            PIL Image
        """
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
            img_np = (img_np * 255).astype(np.uint8)
        elif img_np.min() < 0:  # [-1, 1] range
            img_np = ((img_np + 1) * 127.5).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        return Image.fromarray(img_np)


def build_visual_memory(
    candidate_frame_lists: List[List[int]],
    distance_threshold: int = 10
) -> List[int]:
    """
    Build visual memory by clustering candidate frame indices and selecting representative keyframes.
    
    This implements the BUILDVISUALMEMORY algorithm from MEMER paper:
    1. Collect all candidate frame indices from J'_{0:t} = (J_0, J_1, ..., J_t)
    2. Sort them to form G_{0:t}
    3. Create clusters where consecutive frames are separated by at most d
    4. Select the median index from each cluster as a keyframe
    
    Args:
        candidate_frame_lists: List of lists, where each inner list contains frame indices
                               selected as candidates at different timesteps
        distance_threshold: Maximum distance d between consecutive frames in a cluster
        
    Returns:
        List of keyframe indices (sorted)
    """
    # Step 1: Collect and sort all candidate indices
    all_indices = []
    for candidate_list in candidate_frame_lists:
        all_indices.extend(candidate_list)
    
    if not all_indices:
        return []
    
    # Sort the indices
    sorted_indices = sorted(all_indices)
    
    # Step 2: Build clusters using single-linkage with distance threshold
    clusters = []
    current_cluster = [sorted_indices[0]]
    
    for i in range(1, len(sorted_indices)):
        # Check distance between consecutive sorted indices
        if sorted_indices[i] - sorted_indices[i - 1] <= distance_threshold:
            # Add to current cluster
            current_cluster.append(sorted_indices[i])
        else:
            # Save current cluster and start a new one
            clusters.append(current_cluster)
            current_cluster = [sorted_indices[i]]
    
    # Don't forget the last cluster
    clusters.append(current_cluster)
    
    # Step 3: Select median index from each cluster
    keyframe_indices = []
    for cluster in clusters:
        # Get median index
        median_idx = cluster[len(cluster) // 2]
        keyframe_indices.append(median_idx)
    
    return sorted(keyframe_indices)