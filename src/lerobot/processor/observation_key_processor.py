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

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="select_observation_keys_processor")
class SelectObservationKeysProcessorStep(ObservationProcessorStep):
    """Keep only the configured observation keys."""

    keep_keys: list[str] = field(default_factory=list)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.keep_keys:
            return observation
        return {key: observation[key] for key in self.keep_keys if key in observation}

    def get_config(self) -> dict[str, Any]:
        return {"keep_keys": self.keep_keys}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = features.copy()
        new_features[PipelineFeatureType.OBSERVATION] = {
            key: feature
            for key, feature in features[PipelineFeatureType.OBSERVATION].items()
            if key in self.keep_keys
        }
        return new_features


@dataclass
@ProcessorStepRegistry.register(name="cast_observation_keys_processor")
class CastObservationKeysProcessorStep(ObservationProcessorStep):
    """Cast selected observation keys to a floating-point dtype."""

    cast_keys: list[str] = field(default_factory=list)
    dtype: str = "float32"

    DTYPE_MAPPING = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }

    def __post_init__(self) -> None:
        if self.dtype not in self.DTYPE_MAPPING:
            raise ValueError(f"Invalid dtype '{self.dtype}'. Available options: {list(self.DTYPE_MAPPING)}")
        self._target_dtype = self.DTYPE_MAPPING[self.dtype]

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.cast_keys:
            return observation

        new_observation = dict(observation)
        for key in self.cast_keys:
            if key in new_observation:
                new_observation[key] = torch.as_tensor(new_observation[key], dtype=self._target_dtype)
        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {"cast_keys": self.cast_keys, "dtype": self.dtype}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
