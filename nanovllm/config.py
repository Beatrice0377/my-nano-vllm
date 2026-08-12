import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


def validate_supported_config(hf_config) -> None:
    """Fail fast on Qwen3 configuration features this engine does not support.

    The custom model implementation (nanovllm/models/qwen3.py) is validated on
    Qwen3-0.6B. Rejecting known-unsupported features here prevents silently
    running with incorrect attention semantics; this is a guard, not a
    compatibility framework.
    """
    supported_model_types = ("qwen3",)
    if hf_config.model_type not in supported_model_types:
        raise ValueError(
            f"my-nano-vLLM currently validates the custom model path only for "
            f"Qwen3 configs (got model_type={hf_config.model_type!r}). "
            f"Validated configuration: Qwen3-0.6B."
        )
    if getattr(hf_config, "attention_bias", False):
        raise ValueError(
            f"Qwen3 attention_bias=True is not supported by the custom model "
            f"implementation (validated configuration: Qwen3-0.6B, "
            f"attention_bias=False)."
        )
    if getattr(hf_config, "sliding_window", None) is not None:
        raise ValueError(
            f"Qwen3 sliding-window attention is not supported by the custom "
            f"model implementation (validated configuration: Qwen3-0.6B, "
            f"sliding_window=None)."
        )
    if getattr(hf_config, "rope_scaling", None) is not None:
        raise ValueError(
            f"Qwen3 RoPE scaling is not supported by the custom model "
            f"implementation (validated configuration: Qwen3-0.6B, "
            f"rope_scaling=None)."
        )


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        validate_supported_config(self.hf_config)
        # Compatibility shim: transformers removed the deprecated
        # PretrainedConfig.dtype attribute (upstream nano-vLLM relies on it).
        self.hf_config.dtype = self.hf_config.torch_dtype or torch.bfloat16
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
