import uuid

import pytest

from llama_benchy.corpus import TokenizedCorpus
from llama_benchy.prompts import PromptGenerator


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token) for token in token_ids)


@pytest.fixture
def generator():
    corpus = TokenizedCorpus.__new__(TokenizedCorpus)
    corpus.tokenizer = _CharTokenizer()
    corpus.tokens = [ord(char) for char in "ABCDEF"]
    return PromptGenerator(corpus)


@pytest.mark.parametrize("context_tokens", [0, 2, 5])
@pytest.mark.parametrize("no_cache", [False, True])
def test_generate_accepts_exact_fit_corpus(
    generator, monkeypatch, context_tokens, no_cache
):
    nonce = uuid.UUID(int=0)
    monkeypatch.setattr("llama_benchy.prompts.uuid.uuid4", lambda: nonce)
    suffix = f" {nonce}" if no_cache else ""
    prompt_tokens = 6 - context_tokens + len(suffix)

    context, prompt = generator.generate(prompt_tokens, context_tokens, no_cache)

    assert context == "ABCDEF"[:context_tokens]
    assert prompt == "ABCDEF"[context_tokens:] + suffix


@pytest.mark.parametrize("context_tokens", [0, 1])
def test_generate_can_select_final_window(generator, monkeypatch, context_tokens):
    monkeypatch.setattr(
        "llama_benchy.prompts.np.random.randint", lambda low, high: high - 1
    )

    context, prompt = generator.generate(3 - context_tokens, context_tokens)

    assert context == "DEF"[:context_tokens]
    assert prompt == "DEF"[context_tokens:]
