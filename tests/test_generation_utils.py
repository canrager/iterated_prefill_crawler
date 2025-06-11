import pytest
from core.generation_utils import batch_generate
from core.llm_utils import load_model_and_tokenizer
from nnsight import LanguageModel

TOPICS = ["capital of France", "main city of Germany", "most populous city in Italy"]
USER_MESSAGE_TEMPLATE = "What is the {}?"
THINKING_MESSAGE = "I know that."

DEVICE = "cuda:0"
CACHE_DIR = "/share/u/models/"


@pytest.fixture
def local_model_and_tokenizer():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model, tokenizer = load_model_and_tokenizer(
        model_name, device=DEVICE, cache_dir=CACHE_DIR
    )
    return model, tokenizer, model_name


@pytest.fixture
def remote_model_and_tokenizer():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model = LanguageModel(model_name)
    tokenizer = model.tokenizer
    return model, tokenizer, model_name


def test_batch_generate_capitals(local_model_and_tokenizer):
    model, tokenizer, model_name = local_model_and_tokenizer

    generated_texts = batch_generate(
        model=model,
        tokenizer=tokenizer,
        selected_topics=TOPICS,
        user_message_template=USER_MESSAGE_TEMPLATE,
        thinking_message=THINKING_MESSAGE,
        max_new_tokens=100,
        tokenization_template="chat",
        remote=False,
    )

    assert len(generated_texts) == len(TOPICS)
    for text in generated_texts:
        assert isinstance(text, str)
        assert len(text) > 0


def test_batch_generate_capitals_remote(remote_model_and_tokenizer):
    model, tokenizer, model_name = remote_model_and_tokenizer

    generated_texts = batch_generate(
        model=model,
        tokenizer=tokenizer,
        selected_topics=TOPICS,
        user_message_template=USER_MESSAGE_TEMPLATE,
        thinking_message=THINKING_MESSAGE,
        max_new_tokens=100,
        tokenization_template="chat",
        remote=True,
    )

    assert len(generated_texts) == len(TOPICS)
    for text in generated_texts:
        assert isinstance(text, str)
        assert len(text) > 0
