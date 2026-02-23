from typing import List, Tuple
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
    ids, strs = encode_for_generation(tokenizer, [[{"role": "user", "content": user_msg}]])
    print(strs[0])

    print("\n=== Case 2: With thinking_message ===")
    ids, strs = encode_for_generation(
        tokenizer,
        [[{"role": "user", "content": user_msg}, {"role": "assistant", "content": "I know that."}]],
    )
    print(strs[0])

