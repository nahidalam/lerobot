#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    CastObservationKeysProcessorStep,
    ConcatObservationKeysProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    SelectObservationKeysProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_smolvla import SmolVLAConfig


def _resolve_aic_state_sources(config: SmolVLAConfig) -> tuple[list[str], list[str]]:
    input_features = config.input_features or {}
    if OBS_STATE not in input_features:
        return [], []

    task_id_source_keys = sorted(
        key
        for key in input_features
        if key.startswith("observation.task_id.") or key in {"observation.task_id", "task_id"}
    )
    tcp_offset_source_keys = sorted(
        key
        for key in input_features
        if key.startswith("observation.tcp_offset.")
        or key.startswith("action.tcp_offset.")
        or key in {"observation.tcp_offset", "action.tcp_offset"}
    )
    return task_id_source_keys, tcp_offset_source_keys


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    task_id_source_keys, tcp_offset_source_keys = _resolve_aic_state_sources(config)
    aic_state_source_keys = [*task_id_source_keys, *tcp_offset_source_keys]
    normalize_observation_keys = None
    concat_step = None

    if aic_state_source_keys:
        normalize_observation_keys = set(config.image_features) | set(tcp_offset_source_keys)
        concat_step = ConcatObservationKeysProcessorStep(
            source_keys=aic_state_source_keys,
            output_key=OBS_STATE,
            drop_source_keys=True,
            source_feature_shapes={
                key: tuple(config.input_features[key].shape)
                for key in aic_state_source_keys
                if key in config.input_features
            },
        )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        SelectObservationKeysProcessorStep(keep_keys=sorted(config.input_features or {})),
        CastObservationKeysProcessorStep(cast_keys=aic_state_source_keys),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            normalize_observation_keys=normalize_observation_keys,
        ),
    ]
    if concat_step is not None:
        input_steps.append(concat_step)
    input_steps.extend(
        [
            NewLineTaskProcessorStep(),
            TokenizerProcessorStep(
                tokenizer_name=config.vlm_model_name,
                padding=config.pad_language_to,
                padding_side="right",
                max_length=config.tokenizer_max_length,
            ),
            DeviceProcessorStep(device=config.device),
        ]
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
