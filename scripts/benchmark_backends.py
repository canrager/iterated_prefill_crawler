#!/usr/bin/env python3
"""
Benchmark script to compare transformers vs vLLM backends.
Usage: python scripts/benchmark_backends.py
"""

import time
import torch
from core.llm_utils import load_model_and_tokenizer
from core.generation_utils import batch_generate
from core.project_config import INPUT_DIR

# Small model for quick testing
MODEL_NAME = "allenai/Llama-3.1-Tulu-3-8B-SFT"
CACHE_DIR = "/home/can/models/"
DEVICE = "cuda"

# Test parameters
TEST_TOPICS = [
    "artificial intelligence",
    "climate change",
    "space exploration",
    "quantum computing",
    "renewable energy",
    "genetic engineering",
    "cybersecurity",
    "machine learning",
    "blockchain technology",
    "virtual reality",
]

USER_MESSAGE_TEMPLATE = "Tell me about {}. I'm curious."
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.6
NUM_SAMPLES = 1

def benchmark_backend(backend_name: str):
    """Run benchmark for a specific backend."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {backend_name.upper()} backend")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading model with {backend_name} backend...")
    start_load = time.time()
    model, tokenizer = load_model_and_tokenizer(
        model_name=MODEL_NAME,
        cache_dir=CACHE_DIR,
        device=DEVICE,
        backend=backend_name,
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.9,
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s\n")

    # Warmup
    print("Warming up...")
    _ = batch_generate(
        model=model,
        tokenizer=tokenizer,
        selected_topics=TEST_TOPICS[:2],
        user_message_template=USER_MESSAGE_TEMPLATE,
        max_new_tokens=50,
        temperature=TEMPERATURE,
        num_samples_per_topic=1,
    )
    print("Warmup complete\n")

    # Benchmark
    print(f"Generating {len(TEST_TOPICS)} samples...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_gen = time.time()

    outputs = batch_generate(
        model=model,
        tokenizer=tokenizer,
        selected_topics=TEST_TOPICS,
        user_message_template=USER_MESSAGE_TEMPLATE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        num_samples_per_topic=NUM_SAMPLES,
        verbose=False,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    gen_time = time.time() - start_gen

    # Calculate stats
    total_outputs = len(outputs)
    total_tokens = sum(len(tokenizer.encode(output)) for output in outputs)
    throughput = total_outputs / gen_time
    tokens_per_sec = total_tokens / gen_time

    print(f"\n{backend_name.upper()} Results:")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Total outputs: {total_outputs}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Tokens/sec: {tokens_per_sec:.2f}")
    print(f"\nExample output:")
    print(f"  Topic: {TEST_TOPICS[0]}")
    print(f"  Output: {outputs[0][:200]}...")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "backend": backend_name,
        "load_time": load_time,
        "gen_time": gen_time,
        "throughput": throughput,
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": total_tokens,
    }


def main():
    print(f"\n{'='*60}")
    print("Backend Benchmark Comparison")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Test topics: {len(TEST_TOPICS)}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")

    results = []

    # Benchmark transformers
    try:
        result_hf = benchmark_backend("transformers")
        results.append(result_hf)
    except Exception as e:
        print(f"\nTransformers benchmark failed: {e}")

    # Benchmark vLLM
    try:
        result_vllm = benchmark_backend("vllm")
        results.append(result_vllm)
    except Exception as e:
        print(f"\nvLLM benchmark failed: {e}")

    # Summary comparison
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("Summary Comparison")
        print(f"{'='*60}\n")

        speedup = results[1]["throughput"] / results[0]["throughput"]
        token_speedup = results[1]["tokens_per_sec"] / results[0]["tokens_per_sec"]

        print(f"{'Metric':<30} {'Transformers':<15} {'vLLM':<15} {'Speedup':<10}")
        print(f"{'-'*70}")
        print(f"{'Load time (s)':<30} {results[0]['load_time']:<15.2f} {results[1]['load_time']:<15.2f} {results[1]['load_time']/results[0]['load_time']:<10.2f}x")
        print(f"{'Generation time (s)':<30} {results[0]['gen_time']:<15.2f} {results[1]['gen_time']:<15.2f} {results[0]['gen_time']/results[1]['gen_time']:<10.2f}x")
        print(f"{'Throughput (samples/s)':<30} {results[0]['throughput']:<15.2f} {results[1]['throughput']:<15.2f} {speedup:<10.2f}x")
        print(f"{'Tokens/sec':<30} {results[0]['tokens_per_sec']:<15.2f} {results[1]['tokens_per_sec']:<15.2f} {token_speedup:<10.2f}x")

        print(f"\nvLLM is {speedup:.2f}x faster than transformers for generation!")


if __name__ == "__main__":
    main()
