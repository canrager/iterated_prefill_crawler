import os

# Set environment variables to disable TorchDynamo before importing torch
# This needs to be done before any torch imports for gemma3 models
# Currently, this is being done in the run script
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# os.environ["TORCHDYNAMO_DISABLE"] = "1"

from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from src.directory_config import INPUT_DIR, resolve_cache_dir
from vllm import LLM


def load_model_and_tokenizer(
    model_name: Optional[str],
    cache_dir: str,
    device: str,
    quantization_bits: int = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: Optional[int] = None,
):
    """
    Load a local vLLM model and tokenizer. Returns (None, None) if model_name is None.

    Args:
        model_name: HuggingFace model name or path, or None to skip loading
        cache_dir: Directory to cache the downloaded model
        device: Device hint (unused for vLLM, kept for compatibility)
        quantization_bits: Quantization bits (4, 8, or None)
        vllm_tensor_parallel_size: Number of GPUs for vLLM tensor parallelism
        vllm_gpu_memory_utilization: GPU memory fraction for vLLM
        vllm_max_model_len: Max sequence length for vLLM (None = use model default)
    """
    if model_name is None:
        return None, None

    return load_vllm_model(
        model_name=model_name,
        cache_dir=cache_dir,
        tensor_parallel_size=vllm_tensor_parallel_size,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        dtype="bfloat16",
        quantization_bits=quantization_bits,
    )


def load_from_path(path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=path,
    )
    model.eval()
    # Disable compilation for Gemma3 models due to TorchDynamo compatibility issues
    if "gemma-3" not in path:
        model = torch.compile(model)
    return model, tokenizer


def load_vllm_model(
    model_name: str,
    cache_dir: str = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    dtype: str = "bfloat16",
    quantization_bits: Optional[int] = None,
):
    """
    Load a model using vLLM for efficient inference.

    Args:
        model_name: HuggingFace model name or path
        cache_dir: Directory to cache the downloaded model
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        max_model_len: Maximum sequence length (None = use model default)
        dtype: Data type for model weights (default: "bfloat16")
        quantization_bits: Quantization bits (4, 8, or None). When 4 or 8, uses bitsandbytes quantization.

    Returns:
        LLM: vLLM model instance
        AutoTokenizer: HuggingFace tokenizer
    """
    # Load tokenizer separately (vLLM also loads it internally but we need it for preprocessing)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Determine quantization parameter for vLLM
    # vLLM supports bitsandbytes quantization via quantization="bitsandbytes"
    # Note: vLLM defaults to 4-bit quantization when using bitsandbytes
    quantization = None
    if quantization_bits in [4, 8]:
        quantization = "bitsandbytes"
        print(
            f"Using bitsandbytes quantization (vLLM defaults to 4-bit) for {quantization_bits}-bit request"
        )
    else:
        print("Using no quantization with vLLM")

    # Initialize vLLM model
    llm_kwargs = {
        "model": model_name,
        "download_dir": cache_dir,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "trust_remote_code": True,  # Some models need this
    }

    # Add quantization parameter if specified
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)

    return llm, tokenizer


