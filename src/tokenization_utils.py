import re
from typing import List, Optional, Tuple

from transformers import AutoTokenizer


def encode_for_generation(
    tokenizer: AutoTokenizer,
    messages: List[List[dict]],
) -> Tuple[List[List[int]], List[str]]:
    """
    Encode messages for generation using apply_chat_template as base,
    then appending thinking/prefill tokens on top.

    The chat template is responsible for opening any model-specific thinking
    block (e.g. <think> for R1 models). This function simply appends content
    after the template without re-emitting start tokens.

    Args:
        tokenizer: HuggingFace tokenizer
        messages: List of message lists in OpenAI chat format. Each inner list
                  is either [user_msg] or [user_msg, assistant_msg]. The
                  assistant content, if present, is appended as raw tokens
                  after the generation prompt (prefill).

    Returns:
        Tuple of (input_ids per message, decoded input strings)
    """
    all_input_ids = []
    for msg_list in messages:
        user_msg = msg_list[0]["content"]
        token_ids: List[int] = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            add_generation_prompt=True,
            tokenize=True,
        )

        prefill = msg_list[1]["content"] if len(msg_list) > 1 else None
        if prefill:
            token_ids = token_ids + tokenizer.encode(prefill, add_special_tokens=False)

        all_input_ids.append(token_ids)

    decoded_strs = [tokenizer.decode(ids) for ids in all_input_ids]
    return all_input_ids, decoded_strs


def get_thinking_skip_prefill(tokenizer: AutoTokenizer) -> Optional[str]:
    """Return the prefill string that makes a reasoning model skip its thinking phase.

    For local reasoning models the refusal-check generation is capped at a small
    number of tokens (e.g. 25).  Without intervention those tokens are consumed
    entirely by the ``<think>\\n…`` preamble, leaving nothing for the actual answer.

    This function returns the minimal assistant-turn prefill that closes the
    (possibly implicit) thinking block immediately so generation begins with
    substantive content.  Returns ``None`` for non-reasoning models so they
    receive no prefill at all.

    Detection uses two complementary methods, either of which is sufficient:

    **Method 1 — rendered generation prompt** (catches DeepSeek-R1 and similar):
    Render ``apply_chat_template(…, add_generation_prompt=True)`` and check
    whether the output ends with ``<think>`` (possibly followed by whitespace).
    DeepSeek-R1-Distill ends its generation prompt with
    ``<｜Assistant｜><think>\\n``, so the model's very first generated token is
    content *inside* the thinking block.  Injecting ``</think>\\n`` closes it.

    **Method 2 — template source scan** (catches Qwen3 and similar):
    Look for ``<think>`` appearing as an *emitted* Jinja string literal in the
    template source — i.e. inside a ``{{ … }}`` expression or after a ``+``
    concatenation operator.  This correctly rejects Phi-4-reasoning, whose
    template mentions ``<think>`` only as prose inside its hardcoded system-prompt
    string (not as a token the template outputs), and also rejects plain models
    (Llama-3, Mistral-Small, Tulu, …) whose templates contain no ``<think>`` at all.

    Model behaviour summary
    -----------------------
    * **DeepSeek-R1-Distill**: caught by Method 1 → ``"</think>\\n"``
    * **Qwen3**: caught by Method 2 → ``"</think>\\n"``
    * **Phi-4-reasoning**: rejected by both methods → ``None``
    * **Llama-3, Mistral-Small, Tulu**: rejected by both methods → ``None``
    """
    # Method 1: does the rendered generation prompt end with <think> (possibly + whitespace)?
    # This handles models that auto-inject the opening tag (DeepSeek-R1, etc.)
    try:
        rendered: str = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        if rendered.rstrip(" \n").endswith("<think>"):
            return "</think>\n"
    except Exception:
        pass

    # Method 2: does the template source emit <think> as a Jinja output token?
    # Match patterns like:  + '<think>   or  {{ '<think>   or  + "\n<think>
    # This handles models where the model naturally opens the block but the
    # template is aware of it (Qwen3 history rendering, etc.)
    template: str = (
        getattr(tokenizer, "chat_template", None)
        or getattr(tokenizer, "default_chat_template", None)
        or ""
    )
    if isinstance(template, list):
        # Some tokenizers expose a list of named templates; use the default one.
        for entry in template:
            if isinstance(entry, dict) and entry.get("name") == "default":
                template = entry.get("template", "")
                break
        else:
            template = template[0].get("template", "") if template else ""
    if re.search(r"""(?:{{|\+)\s*['"][^'"]*<think>""", str(template)):
        return "</think>\n"

    return None


if __name__ == "__main__":
    import os

    # Run from project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/config.yaml")
    model_name = cfg.model.local_model  # meta-llama/Llama-3.1-8B-Instruct

    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="models")

    user_msg = "Hello, what topics do you avoid?"

    print("\n=== Case 1: Plain user message ===")
    ids, strs = encode_for_generation(
        tokenizer, [[{"role": "user", "content": user_msg}]]
    )
    print(strs[0])

    print("\n=== Case 2: With thinking_message ===")
    ids, strs = encode_for_generation(
        tokenizer,
        [
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": "I know that."},
            ]
        ],
    )
    print(strs[0])

    print("\n=== Case 3: get_thinking_skip_prefill ===")
    skip_prefill = get_thinking_skip_prefill(tokenizer)
    print(f"thinking_skip_prefill={skip_prefill!r}")
