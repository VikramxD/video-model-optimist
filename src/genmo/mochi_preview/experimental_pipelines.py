import json
import os
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, cast
import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from safetensors.torch import load_file
from torch import nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block

import genmo.mochi_preview.dit.joint_model.context_parallel as cp
import genmo.mochi_preview.vae.cp_conv as cp_conv
from genmo.lib.progress import get_new_progress_bar, progress_bar
from genmo.lib.utils import Timer
from genmo.mochi_preview.vae.models import (
    Decoder,
    decode_latents,
    decode_latents_tiled_full,
    decode_latents_tiled_spatial,
)
from genmo.mochi_preview.vae.vae_stats import dit_latents_to_vae_latents
import bitsandbytes as bnb

def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


T5_MODEL = "google/t5-v1_1-xxl"
MAX_T5_TOKEN_LENGTH = 256


def setup_fsdp_sync(model, device_id, *, param_dtype, auto_wrap_policy) -> FSDP:
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=device_id,
        sync_module_states=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


class ModelFactory(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def get_model(self, *, local_rank: int, device_id: Union[int, Literal["cpu"]], world_size: int) -> Any:
        if device_id == "cpu":
            assert world_size == 1, "CPU offload only supports single-GPU inference"







class T5ModelFactory(ModelFactory):
    def __init__(self, quantize: bool = True):
        super().__init__()
        self.quantize = quantize

    def get_model(self, *, local_rank, device_id, world_size):
        super().get_model(local_rank=local_rank, device_id=device_id, world_size=world_size)
        
        # Load the T5 model with or without quantization
        if self.quantize:
            model = T5EncoderModel.from_pretrained(
                T5_MODEL,
                load_in_8bit=True,    # Enables 8-bit quantization with BitsAndBytes
                device_map="auto" if world_size > 1 else {local_rank: device_id},
            )
        else:
            model = T5EncoderModel.from_pretrained(T5_MODEL)

        # Set up for FSDP if needed
        if world_size > 1 and not self.quantize:
            model = setup_fsdp_sync(
                model,
                device_id=device_id,
                param_dtype=torch.float32,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={
                        T5Block,
                    },
                ),
            )
        elif isinstance(device_id, int) and not self.quantize:
            model = model.to(torch.device(f"cuda:{device_id}"))  # type: ignore

        return model.eval()
