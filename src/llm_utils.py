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
    Gemma3ForCausalLM,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
)
from configs.directory_config import INPUT_DIR, MODELS_DIR, resolve_cache_dir
from vllm import LLM


def load_model_and_tokenizer(
    model_name: str,
    cache_dir: str,
    device: str,
    quantization_bits: int = None,
    tokenizer_only: bool = False,
    backend: str = "transformers",
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: Optional[int] = None,
):
    """
    Load a huggingface model and tokenizer.

    Args:
        model_name: HuggingFace model name or path
        cache_dir: Directory to cache the downloaded model
        device: Device to load the model on ('auto', 'cuda', 'cpu', etc.)
        quantization_bits: Quantization bits (4, 8, or None)
        tokenizer_only: If True, only load tokenizer
        backend: "transformers" or "vllm"
        vllm_tensor_parallel_size: Number of GPUs for vLLM tensor parallelism
        vllm_gpu_memory_utilization: GPU memory fraction for vLLM
        vllm_max_model_len: Max sequence length for vLLM (None = use model default)
    """
    if any(model_name.startswith(prefix) for prefix in ["claude", "gpt"]):
        return model_name, None

    # vLLM backend
    if backend == "vllm":
        return load_vllm_model(
            model_name=model_name,
            cache_dir=cache_dir,
            tensor_parallel_size=vllm_tensor_parallel_size,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
            dtype="bfloat16",
            quantization_bits=quantization_bits,
        )

    # Transformers backend (default)
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if tokenizer_only:
        return None, tokenizer

    if cache_dir is not None:
        local_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        local_path_exists = os.path.exists(local_path)
        if local_path_exists:
            print(f"Model exists in {local_path}")
        else:
            print(f"Model does not exist in {local_path}")

    # Determine quantization
    torch_dtype = torch.bfloat16

    if quantization_bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization and bfloat16")
    elif quantization_bits == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        print("Using 4-bit quantization and bfloat16")
    else:
        quantization_config = None
        print("Using no quantization and bfloat16")

    if device == "cuda":
        device_map = "auto"
    else:
        device_map = device
    # # Standard way

    from_pretrained_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "quantization_config": quantization_config,
        "cache_dir": cache_dir,
    }

    model = AutoModelForCausalLM.from_pretrained(**from_pretrained_kwargs)
    # if "gemma-3" in model_name:
    #     model = Gemma3ForCausalLM.from_pretrained(**from_pretrained_kwargs)
    #     print("Loaded Gemma3 model without TorchDynamo compilation")
    # else:
    #     model = torch.compile(model)

    # Optimize for inference
    model.eval()

    return model, tokenizer


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
        print(f"Using bitsandbytes quantization (vLLM defaults to 4-bit) for {quantization_bits}-bit request")
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



def load_zh_en_translation_model(cache_dir: str, device: str):
    # Translation LM
    tokenizer_zh_en = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-zh-en", cache_dir=cache_dir
    )
    model_zh_en = AutoModelForSeq2SeqLM.from_pretrained(
        "Helsinki-NLP/opus-mt-zh-en", cache_dir=cache_dir, device_map=device
    )
    return model_zh_en, tokenizer_zh_en


def load_en_zh_translation_model(cache_dir: str, device: str):
    # Translation LM
    tokenizer_en_zh = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-en-zh", cache_dir=cache_dir
    )
    model_en_zh = AutoModelForSeq2SeqLM.from_pretrained(
        "Helsinki-NLP/opus-mt-en-zh", cache_dir=cache_dir, device_map=device
    )
    return model_en_zh, tokenizer_en_zh



def load_filter_models(
    cache_dir: Optional[str] = None, device: str = "auto"
):
    # Resolve cache_dir relative to ROOT_DIR.parent and create if needed
    cache_dir_path = resolve_cache_dir(cache_dir)
    cache_dir_str = str(cache_dir_path)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_zh_en, tokenizer_zh_en = load_zh_en_translation_model(cache_dir_str, device)
    model_en_zh, tokenizer_en_zh = load_en_zh_translation_model(cache_dir_str, device)

    return {
        "tokenizer_zh_en": tokenizer_zh_en,
        "model_zh_en": model_zh_en,
        "tokenizer_en_zh": tokenizer_en_zh,
        "model_en_zh": model_en_zh,
    }
