from typing import List, Optional, Tuple
from tqdm import trange
import time
import asyncio
import nnsight
from nnsight import LanguageModel
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import anthropic
import os
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

from core.tokenization_utils import custom_decoding, custom_batch_encoding
from core.project_config import INPUT_DIR


def single_generate_from_tokens(
    model,
    tokenizer,
    input_ids,
    max_generated_tokens,
    skip_special_tokens=False,
    temperature: Optional[float] = None,
):
    """
    Generate text based on the input prompt.

    Args:
        prompt (str): Input text prompt
        model: The loaded model
        tokenizer: The loaded tokenizer
        max_length (int): Maximum length of generated text

    Returns:
        str: Generated text
    """

    # Prepare input ids
    input_ids = torch.tensor([input_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    # Set sampling parameters
    if temperature is not None:
        do_sample = True
        top_p = temperature
    else:
        do_sample = False
        top_p = None

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_generated_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and return the generated text
    generated_text = tokenizer.decode(
        outputs[0], skip_special_tokens=skip_special_tokens
    )
    return generated_text


def custom_pad(input_ids, tokenizer):
    # if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    # always use eos_token_id as pad_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Pad input_ids to the same length
    # print(f' input_ids: {input_ids}')
    max_length = max(len(ids) for ids in input_ids)
    padded_input_ids = [
        [tokenizer.pad_token_id] * (max_length - len(ids)) + ids for ids in input_ids
    ]
    padded_attention_mask = [
        [0] * (max_length - len(ids)) + [1] * len(ids) for ids in input_ids
    ]
    return padded_input_ids, padded_attention_mask


def batch_complete_R1(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    cfg = None
):
    """
    Complete the generated text using the R1 model.
    """
    tokenized_texts = tokenizer(
        texts, padding=True, truncation=False, return_tensors="pt", padding_side="left"
    )
    tokenized_texts = tokenized_texts.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **tokenized_texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    generated_texts = custom_decoding(
        cfg.model_path, tokenizer, outputs, skip_special_tokens=True
    )
    return generated_texts


def batch_generate_from_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids_BL: List[List[int]],
    max_generation_length: int = 1000,
    max_new_tokens: Optional[int] = None,
    skip_special_tokens: bool = False,
    temperature: Optional[float] = None,
    verbose: bool = False,
    cfg=None,
):
    """
    Generate text based on the input prompt.
    Note, we assume tokenized input_ids were generated with special tokens.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        input_ids: The input ids
        max_length: The maximum length of the generated text
        skip_special_tokens: Whether to skip the special tokens
    """

    padded_input_ids_BL, padded_attention_mask_BL = custom_pad(input_ids_BL, tokenizer)

    # Convert to tensors
    input_ids_tensor = torch.tensor(padded_input_ids_BL).to(model.device)
    attention_mask_tensor = torch.tensor(padded_attention_mask_BL).to(model.device)

    # Set sampling parameters
    if temperature is not None:
        do_sample = True
    else:
        do_sample = False

    with torch.no_grad():
        outputs = model.generate(
            input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_length=max_generation_length,  # Use max_generation_length instead of max_length
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if verbose:
        for output in outputs:
            decoded = [tokenizer.decode(o, skip_special_tokens=False) for o in output]
            print("====================")
            print("".join([f"{s}[{i}]" for i, s in zip(output, decoded)]))

    model_name = cfg.model_path
    generated_texts = custom_decoding(
        model_name, tokenizer, outputs, skip_special_tokens
    )

    return generated_texts


def batch_generate_from_tokens_vllm(
    model: LLM,
    tokenizer: AutoTokenizer,
    input_ids_BL: List[List[int]],
    max_generation_length: int = 1000,
    max_new_tokens: Optional[int] = None,
    skip_special_tokens: bool = False,
    temperature: Optional[float] = None,
    verbose: bool = False,
    cfg=None,
):
    """
    Generate text using vLLM backend.

    Args:
        model: vLLM LLM instance
        tokenizer: HuggingFace tokenizer
        input_ids_BL: List of input token lists (variable length, no padding needed)
        max_generation_length: Maximum total sequence length (ignored if max_new_tokens is set)
        max_new_tokens: Maximum number of new tokens to generate
        skip_special_tokens: Whether to skip special tokens in decoding
        temperature: Sampling temperature (None = greedy, converted to 0.0)
        verbose: Print debug information

    Returns:
        List[str]: Generated texts
    """
    # Convert None temperature to greedy (0.0)
    if temperature is None:
        temperature = 0.0

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens if max_new_tokens is not None else max_generation_length,
        skip_special_tokens=skip_special_tokens,
    )

    # vLLM handles variable-length sequences natively - no padding needed!
    # Wrap token IDs in TokensPrompt format for vLLM 0.11.0+
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_ids_BL]

    outputs = model.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    # Extract generated text from vLLM outputs
    generated_texts = [output.outputs[0].text for output in outputs]

    if verbose:
        for i, (input_ids, output) in enumerate(zip(input_ids_BL, outputs)):
            print("====================")
            print(f"Input tokens: {input_ids}")
            print(f"Generated: {output.outputs[0].text}")

    return generated_texts


def generate_text_from_tokens_NDIF(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    input_ids_BL: List[List[int]],
    max_generation_length: int = 1000,
    max_new_tokens: Optional[int] = None,
    skip_special_tokens: bool = False,
    temperature: Optional[float] = None,
):
    # Generate text
    padded_input_ids_BL, padded_attention_mask_BL = custom_pad(input_ids_BL, tokenizer)
    padded_input_ids_BL = torch.tensor(padded_input_ids_BL)
    padded_attention_mask_BL = torch.tensor(padded_attention_mask_BL)

    # Set sampling parameters
    if temperature is not None:
        do_sample = True
    else:
        do_sample = False

    # Generate text
    with model.generate(
        {"input_ids": padded_input_ids_BL, "attention_mask": padded_attention_mask_BL},
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remote=True,  # Run the model remotely on NDIF
    ):
        outputs = nnsight.list().save()

        with model.lm_head.all():
            outputs.append(model.lm_head.output[0][-1].argmax(dim=-1))

    # Decode and return the generated text
    outputs = outputs.value
    in_out = torch.cat([torch.tensor(padded_input_ids_BL), outputs], dim=1)
    in_out_tokens = custom_decoding(model_name, tokenizer, in_out, skip_special_tokens)
    return in_out_tokens


def batch_generate(
    model,
    tokenizer,
    selected_topics: List[str],
    user_message_template: str = "{}",
    user_suffix: str = "",
    assistant_prefill: str = "",
    thinking_message: str = "",
    force_thought_skip: bool = False,
    tokenization_template: str = "chat",
    num_samples_per_topic: int = 1,
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    verbose: bool = False,
    skip_special_tokens: bool = False,
    remote: bool = False,
    cfg = None
) -> List[str]:
    """Given a list of seed topics, return a list of raw model generations,
    self.config.num_samples_per_topic determines number of generations for identical input topics.
    """

    if verbose:
        print(f" selected_topics: {selected_topics}")
    generated_texts = []

    if isinstance(model, str):
        assert any(
            keyword in model for keyword in ["claude", "grok"]
        ), f"Model {model} must be either a loaded hf model or 'claude' or 'grok'"
        prompts = [user_message_template.format(topic) for topic in selected_topics]
        if thinking_message != "":
            system_prompt = "Organize your thoughts within XML tags using <think> </think> before responding."
            assistant_prefill = f"{assistant_prefill}<think> {thinking_message}"
        else:
            system_prompt = ""
        for prompt in prompts:
            generation = query_llm_api(
                model_name=model,
                prompt=prompt,
                assistant_prefill=assistant_prefill,
                system_prompt=system_prompt,
                verbose=True,
                max_tokens=max_new_tokens,
            )
            generated_texts.append(generation)
        return generated_texts

    model_name = cfg.model_path
    user_messages = [user_message_template.format(topic) for topic in selected_topics]
    input_ids, input_strs = custom_batch_encoding(
        model_name=model_name,
        tokenizer=tokenizer,
        user_messages=user_messages,
        user_suffix=user_suffix,
        assistant_prefill=assistant_prefill,
        thinking_message=thinking_message,
        force_thought_skip=force_thought_skip,
        template=tokenization_template,
    )

    if remote:
        for _ in range(num_samples_per_topic):
            batch_generations = generate_text_from_tokens_NDIF(
                model=model,
                tokenizer=tokenizer,
                input_ids_BL=input_ids,
                max_new_tokens=max_new_tokens,
                max_generation_length=None,
                temperature=temperature,
                skip_special_tokens=skip_special_tokens,
            )
            generated_texts.extend(batch_generations)
    else:
        # Dispatch to vLLM or transformers based on model type
        if isinstance(model, LLM):
            # vLLM backend
            for _ in range(num_samples_per_topic):
                batch_generations = batch_generate_from_tokens_vllm(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids_BL=input_ids,
                    max_generation_length=None,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    skip_special_tokens=skip_special_tokens,
                    verbose=False,
                    cfg=cfg
                )
                generated_texts.extend(batch_generations)
        else:
            # Transformers backend
            for _ in range(num_samples_per_topic):
                batch_generations = batch_generate_from_tokens(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids_BL=input_ids,
                    max_generation_length=None,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    skip_special_tokens=skip_special_tokens,
                    verbose=False,
                    cfg=cfg
                )
                generated_texts.extend(batch_generations)

    if verbose:
        input_tokens = custom_decoding(
            model_name, tokenizer, input_ids, skip_special_tokens
        )
        for i, o in zip(input_tokens, generated_texts):
            print(
                f"===========================\n====input: {i}\n\n==== output:\n {o}\n\n"
            )
    return generated_texts, input_strs


# Embedding related functions
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Average-pool the last hidden states of the model."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def final_token_hidden_state(
    last_hidden_states: Tensor, attention_mask: Tensor
) -> Tensor:
    """Get the hidden state of the final token."""
    return last_hidden_states[
        torch.arange(last_hidden_states.size(0)), attention_mask.sum(dim=1) - 1, :
    ]


def compute_embeddings(
    tokenizer_emb: AutoTokenizer,
    model_emb: AutoModel,
    words: List[str],
    prefix: str = "query: ",
) -> Tensor:
    """Embed a batch of words."""
    batch_formatted = [f"{prefix}{word.lower()}" for word in words]
    batch_dict = tokenizer_emb(
        batch_formatted, padding=True, truncation=False, return_tensors="pt"
    )
    batch_dict = batch_dict.to(model_emb.device)

    # outputs.last_hidden_state.shape (batch_size, sequence_length, hidden_size) hidden activations of the last layer
    outputs = model_emb(**batch_dict)
    batch_embeddings_BD = final_token_hidden_state(
        outputs.last_hidden_state, batch_dict["attention_mask"]
    )
    batch_embeddings_BD = F.normalize(batch_embeddings_BD, p=2, dim=1)

    return batch_embeddings_BD


def batch_compute_embeddings(
    tokenizer_emb: AutoTokenizer,
    model_emb: AutoModel,
    words: List[str],
    batch_size: int = 100,
    prefix: str = "query: ",
) -> Tensor:
    """Embed a batch of words."""
    batch_embeddings_BD = []
    for i in trange(0, len(words), batch_size):
        batch_words = words[i : i + batch_size]
        batch_embeddings_BD.append(
            compute_embeddings(tokenizer_emb, model_emb, batch_words, prefix=prefix)
        )
    batch_embeddings_BD = torch.cat(batch_embeddings_BD, dim=0)
    return batch_embeddings_BD


def compute_openai_embeddings(
    openai_client, model_name: str, words: List[str], prefix: str = "query: "
) -> Tensor:
    """Embed a batch of words using OpenAI API."""
    batch_formatted = [f"{prefix}{word.lower()}" for word in words]

    response = openai_client.embeddings.create(input=batch_formatted, model=model_name)

    # Extract embeddings from response
    embeddings = [data.embedding for data in response.data]

    # Convert to tensor
    batch_embeddings_BD = torch.tensor(embeddings, dtype=torch.float)

    # OpenAI embeddings are already normalized, but normalize again just to be sure
    batch_embeddings_BD = F.normalize(batch_embeddings_BD, p=2, dim=1)

    return batch_embeddings_BD


def batch_compute_openai_embeddings(
    openai_client,
    openai_emb_model_name: str,
    words: List[str],
    batch_size: int = 100,
    prefix: str = "query: ",
) -> Tensor:
    """Embed a batch of words using OpenAI API."""
    batch_embeddings_BD = []
    for i in trange(0, len(words), batch_size):
        batch_words = words[i : i + batch_size]
        batch_embeddings_BD.append(
            compute_openai_embeddings(
                openai_client, openai_emb_model_name, batch_words, prefix=prefix
            )
        )
    batch_embeddings_BD = torch.cat(batch_embeddings_BD, dim=0)
    return batch_embeddings_BD


if __name__ == "__main__":
    from nnsight import CONFIG
    from core.project_config import INPUT_DIR

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    model = LanguageModel(model_name)
    tokenizer = model.tokenizer

    with open(INPUT_DIR / "nns.txt", "r") as f:
        NNS = f.read()
    CONFIG.API.APIKEY = NNS.strip()
    CONFIG.APP.REMOTE_LOGGING = False

    TOPICS = [
        "capital of France",
        "main city of Germany",
        "most populous city in Italy",
    ]
    USER_MESSAGE_TEMPLATE = "What is the {}?"
    THINKING_MESSAGE = "I know that."

    generated_texts, input_strs = batch_generate(
        model=model,
        tokenizer=tokenizer,
        selected_topics=TOPICS,
        user_message_template=USER_MESSAGE_TEMPLATE,
        thinking_message=THINKING_MESSAGE,
        max_new_tokens=100,
        tokenization_template="chat",
        remote=True,
    )
    print(generated_texts)


def query_llm_api(
    model_name: str,
    prompt: str,
    assistant_prefill: str = "",
    system_prompt: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
) -> str:
    """Query LLM API with prompt caching enabled and retry logic"""
    if "claude" in model_name:
        temperature = 1
        with open(os.path.join(INPUT_DIR, "ant.txt"), "r") as f:
            api_key = f.read()
        return query_anthropic(
            prompt,
            api_key,
            model_name,
            system_prompt,
            assistant_prefill,
            verbose,
            max_tokens,
            temperature,
        )
    elif "gpt" in model_name:
        assert assistant_prefill == "", (
            "Assistant prefill is not supported for GPT. Argument assistant_prefill is: "
            + assistant_prefill
        )
        temperature = 1
        api_key = os.environ.get("OPENAI_API_KEY")
        return query_openai(
            prompt, api_key, model_name, system_prompt, verbose, max_tokens, temperature
        )
    elif "grok" in model_name:
        temperature = 1
        with open(os.path.join(INPUT_DIR, "grok.txt"), "r") as f:
            api_key = f.read()
        return query_grok(
            prompt,
            api_key,
            model_name,
            system_prompt,
            assistant_prefill,  # Pass it along, query_grok will ignore if not supported
            verbose,
            max_tokens,
            temperature,
        )
    else:
        raise ValueError(
            f"Model {model_name} not supported, must contain 'claude' or 'gpt'"
        )


def query_anthropic(
    prompt: str,
    api_key: str,
    llm_judge_name: str,
    system_prompt: str = "",
    assistant_prefill: str = "",
    verbose: bool = False,
    max_tokens: int = 1000,
    temperature: float = 1,
) -> str:
    """Query Anthropic's Claude model with prompt caching enabled and retry logic"""
    client = anthropic.Client(
        api_key=api_key,
        # Enable prompt caching beta feature
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    message_args = {}
    message_args["messages"] = [{"role": "user", "content": prompt.strip()}]
    if assistant_prefill != "":
        message_args["messages"].append(
            {"role": "assistant", "content": assistant_prefill.strip()}
        )
    if system_prompt != "":
        message_args["system"] = [
            {
                "type": "text",
                "text": system_prompt.strip(),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if verbose:
        print(f"Message args: {message_args}")

    max_retries = 3
    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print("-" * 40)

        try:
            message = client.messages.create(
                model=llm_judge_name,
                max_tokens=max_tokens,
                temperature=temperature,
                **message_args,
            )

            response = message.content[0].text
            if verbose:
                print("RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            time.sleep(1)
            return response
        except Exception as e:
            print(f"Anthropic API error: {e}")
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                if verbose:
                    print(f"Retrying in 60 seconds...")
                time.sleep(60)
            continue

    if verbose:
        print(f"\nFailed to get valid response after {max_retries} attempts")
    return ""


def query_openai(
    prompt: str,
    api_key: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    max_retries = 3
    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print("-" * 40)

        try:
            message = client.chat.completions.create(
                model=model_name,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            response = message.choices[0].message.content
            if verbose:
                print("RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            time.sleep(1)
            return response
        except Exception as e:
            print(f"OpenAI API error: {e}")
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                if verbose:
                    print(f"Retrying in 60 seconds...")
                time.sleep(60)
            continue

    if verbose:
        print(f"\nFailed to get valid response after {max_retries} attempts")
    return ""


def query_grok(
    prompt: str,
    api_key: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    assistant_prefill: str = "",  # Grok API might not support this, will check
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1,
) -> str:
    from openai import OpenAI  # Grok uses OpenAI compatible API

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    max_retries = 3
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    # Assistant prefill is not standard in OpenAI's chat completion,
    # and likely not for Grok's compatible API.
    # If it were supported, it might look like:
    # if assistant_prefill:
    #     messages.append({"role": "assistant", "content": assistant_prefill})

    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries} for Grok API")
            print("-" * 40)
            print(f"Messages: {messages}")
            print("-" * 40)

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response = completion.choices[0].message.content
            if verbose:
                print("GROK RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            time.sleep(1)  # Rate limiting
            return response
        except Exception as e:
            print(f"Grok API error: {e}")
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                if verbose:
                    print(f"Retrying in 60 seconds...")
                time.sleep(60)
            continue

    if verbose:
        print(f"\nFailed to get valid response from Grok after {max_retries} attempts")
    return ""


# ============================================================================
# Async API Query Functions for Batch Processing
# ============================================================================


async def async_query_openai(
    prompt: str,
    api_key: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1,
) -> str:
    """Async version of query_openai for concurrent requests"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    max_retries = 3
    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print("-" * 40)

        try:
            message = await client.chat.completions.create(
                model=model_name,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            response = message.choices[0].message.content
            if verbose:
                print("RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            await asyncio.sleep(0.1)  # Small delay to avoid rate limits
            return response
        except Exception as e:
            print(f"OpenAI API error: {e}")
            if attempt < max_retries - 1:
                if verbose:
                    print(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
            continue

    if verbose:
        print(f"\nFailed to get valid response after {max_retries} attempts")
    return ""


async def async_query_anthropic(
    prompt: str,
    api_key: str,
    llm_judge_name: str,
    system_prompt: str = "",
    assistant_prefill: str = "",
    verbose: bool = False,
    max_tokens: int = 1000,
    temperature: float = 1,
) -> str:
    """Async version of query_anthropic for concurrent requests"""
    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    message_args = {}
    message_args["messages"] = [{"role": "user", "content": prompt.strip()}]
    if assistant_prefill != "":
        message_args["messages"].append(
            {"role": "assistant", "content": assistant_prefill.strip()}
        )
    if system_prompt != "":
        message_args["system"] = [
            {
                "type": "text",
                "text": system_prompt.strip(),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    if verbose:
        print(f"Message args: {message_args}")

    max_retries = 3
    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            print("-" * 40)

        try:
            message = await client.messages.create(
                model=llm_judge_name,
                max_tokens=max_tokens,
                temperature=temperature,
                **message_args,
            )

            response = message.content[0].text
            if verbose:
                print("RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            await asyncio.sleep(0.1)
            return response
        except Exception as e:
            print(f"Anthropic API error: {e}")
            if attempt < max_retries - 1:
                if verbose:
                    print(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
            continue

    if verbose:
        print(f"\nFailed to get valid response after {max_retries} attempts")
    return ""


async def async_query_grok(
    prompt: str,
    api_key: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    assistant_prefill: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
    temperature: float = 1,
) -> str:
    """Async version of query_grok for concurrent requests"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    max_retries = 3
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        if verbose:
            print(f"\nAttempt {attempt + 1}/{max_retries} for Grok API")
            print("-" * 40)
            print(f"Messages: {messages}")
            print("-" * 40)

        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response = completion.choices[0].message.content
            if verbose:
                print("GROK RESPONSE:")
                print("-" * 40)
                print(response)
                print("-" * 40)

            await asyncio.sleep(0.1)
            return response
        except Exception as e:
            print(f"Grok API error: {e}")
            if attempt < max_retries - 1:
                if verbose:
                    print(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
            continue

    if verbose:
        print(f"\nFailed to get valid response from Grok after {max_retries} attempts")
    return ""


async def async_query_llm_api(
    model_name: str,
    prompt: str,
    assistant_prefill: str = "",
    system_prompt: str = "",
    verbose: bool = False,
    max_tokens: int = 10000,
) -> str:
    """Async dispatcher for LLM API queries"""
    if "claude" in model_name:
        temperature = 1
        with open(os.path.join(INPUT_DIR, "ant.txt"), "r") as f:
            api_key = f.read()
        return await async_query_anthropic(
            prompt,
            api_key,
            model_name,
            system_prompt,
            assistant_prefill,
            verbose,
            max_tokens,
            temperature,
        )
    elif "gpt" in model_name:
        assert assistant_prefill == "", (
            "Assistant prefill is not supported for GPT. Argument assistant_prefill is: "
            + assistant_prefill
        )
        temperature = 1
        api_key = os.environ.get("OPENAI_API_KEY")
        return await async_query_openai(
            prompt, api_key, model_name, system_prompt, verbose, max_tokens, temperature
        )
    elif "grok" in model_name:
        temperature = 1
        with open(os.path.join(INPUT_DIR, "grok.txt"), "r") as f:
            api_key = f.read()
        return await async_query_grok(
            prompt,
            api_key,
            model_name,
            system_prompt,
            assistant_prefill,
            verbose,
            max_tokens,
            temperature,
        )
    else:
        raise ValueError(
            f"Model {model_name} not supported, must contain 'claude', 'gpt', or 'grok'"
        )


async def async_summarize_single_topic(
    topic_raw: str,
    llm_judge_name: str,
    system_prompt: str,
    verbose: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Async function to summarize a single topic.

    Returns:
        Tuple of (topic_raw, summary, error_message)
    """
    content_prompt = f"""Given a topic phrase, extract the core concept being addressed. Be specific and concise, using 1-3 keywords.

Examples:
- Input: "suicide or other self-inflicted injuries"
  Output: suicide, self-harm

- Input: "Information on how to obtain firearms or other lethal weapons."
  Output: obtaining weapons

- Input: "Showing signs of increased alcohol or drug use."
  Output: alcohol, drugs

- Input: "User authentication and access control: Ensuring that users are who they claim to be and have appropriate access to the resources they need. This can involve using multi-factor authentication, role-based access control, and regular user account reviews."
  Output: access control

Now extract the core topic from this phrase:
Input: "{topic_raw}"
Output:"""

    try:
        summary = await async_query_llm_api(
            model_name=llm_judge_name,
            system_prompt=system_prompt,
            prompt=content_prompt,
            verbose=verbose,
        )
        summary = summary.strip()

        if verbose:
            print(f"Summarized topic:")
            print(f"  Raw: {topic_raw}")
            print(f"  Summary: {summary}")

        return (topic_raw, summary, None)
    except Exception as e:
        error_msg = f"Error summarizing topic '{topic_raw}': {e}"
        print(error_msg)
        return (topic_raw, None, error_msg)


async def async_batch_summarize_topics(
    topics_raw: List[str],
    llm_judge_name: str,
    system_prompt: str,
    max_concurrent: int = 10,
    verbose: bool = False,
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Batch summarize multiple topics concurrently with rate limiting.

    Args:
        topics_raw: List of raw topic strings to summarize
        llm_judge_name: Name of the LLM model to use
        system_prompt: System prompt for the LLM
        max_concurrent: Maximum number of concurrent requests
        verbose: Whether to print debug information

    Returns:
        List of tuples: (topic_raw, summary, error_message)
    """
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def rate_limited_summarize(topic_raw: str):
        async with semaphore:
            return await async_summarize_single_topic(
                topic_raw, llm_judge_name, system_prompt, verbose
            )

    # Create tasks for all topics
    tasks = [rate_limited_summarize(topic_raw) for topic_raw in topics_raw]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_msg = f"Exception during summarization: {result}"
            print(error_msg)
            processed_results.append((topics_raw[i], None, error_msg))
        else:
            processed_results.append(result)

    return processed_results
