import json
import sys
import types

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from llama_benchy.corpus import LightweightTokenizer, TokenizedCorpus


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeRawTokenizer:
    def __init__(self):
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, text, add_special_tokens=False):
        self.encode_calls.append((text, add_special_tokens))
        return _FakeEncoding([1, 2, 3])

    def decode(self, token_ids, skip_special_tokens=True):
        self.decode_calls.append((token_ids, skip_special_tokens))
        return "decoded"


def _install_tokenizers(monkeypatch, tokenizer_cls):
    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        types.SimpleNamespace(Tokenizer=tokenizer_cls),
    )


def _install_transformers(monkeypatch, auto_tokenizer_cls):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=auto_tokenizer_cls),
    )


def test_lightweight_tokenizer_matches_transformers_encode_decode_shape(monkeypatch):
    raw = _FakeRawTokenizer()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(name):
            assert name == "repo/model"
            return raw

    _install_tokenizers(monkeypatch, FakeTokenizer)

    tokenizer = LightweightTokenizer.from_pretrained("repo/model")

    assert tokenizer.encode("hello", add_special_tokens=False) == [1, 2, 3]
    assert tokenizer.decode([1, 2, 3]) == "decoded"
    assert raw.encode_calls == [("hello", False)]
    assert raw.decode_calls == [([1, 2, 3], False)]


def test_deepseek_v4_uses_local_tokenizer_json_before_transformers(tmp_path, monkeypatch):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "max_position_embeddings": 4096,
                "model_type": "deepseek_v4",
                "rope_scaling": {
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "factor": 2,
                    "original_max_position_embeddings": 2048,
                    "type": "yarn",
                },
                "rope_theta": 10000,
            }
        ),
        encoding="utf-8",
    )

    corpus = TokenizedCorpus.__new__(TokenizedCorpus)
    monkeypatch.setattr(
        corpus,
        "_get_transformers_tokenizer",
        lambda name: pytest.fail("DeepSeek tokenizer.json must load without AutoTokenizer"),
    )

    loaded = corpus._get_tokenizer(str(tmp_path))

    assert isinstance(loaded, LightweightTokenizer)
    assert loaded.encode("hello", add_special_tokens=False) == [1]


def test_get_tokenizer_uses_lightweight_backend_without_transformers(monkeypatch):
    class FakeTokenizer:
        @staticmethod
        def from_pretrained(name):
            return _FakeRawTokenizer()

    class FakeAutoTokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return object()

    _install_tokenizers(monkeypatch, FakeTokenizer)
    _install_transformers(monkeypatch, FakeAutoTokenizer)

    corpus = TokenizedCorpus.__new__(TokenizedCorpus)
    tokenizer = corpus._get_tokenizer("repo/model")

    assert isinstance(tokenizer, LightweightTokenizer)
    assert FakeAutoTokenizer.calls == []


def test_get_tokenizer_lazily_uses_transformers_when_lightweight_fails(monkeypatch):
    class FakeTokenizer:
        @staticmethod
        def from_pretrained(name):
            raise RuntimeError("missing tokenizer.json")

    class FakeAutoTokenizer:
        calls = []
        tokenizer = object()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            return cls.tokenizer

    _install_tokenizers(monkeypatch, FakeTokenizer)
    _install_transformers(monkeypatch, FakeAutoTokenizer)

    corpus = TokenizedCorpus.__new__(TokenizedCorpus)
    tokenizer = corpus._get_tokenizer("repo/model")

    assert tokenizer is FakeAutoTokenizer.tokenizer
    assert FakeAutoTokenizer.calls == [
        (("repo/model",), {"use_fast": True, "trust_remote_code": False})
    ]


def test_get_tokenizer_fails_instead_of_substituting_gpt2(monkeypatch):
    class FakeTokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, name):
            cls.calls.append(name)
            raise RuntimeError("missing tokenizer.json")

    class FakeAutoTokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))
            raise MemoryError("too large")

    _install_tokenizers(monkeypatch, FakeTokenizer)
    _install_transformers(monkeypatch, FakeAutoTokenizer)

    corpus = TokenizedCorpus.__new__(TokenizedCorpus)
    with pytest.raises(
        RuntimeError,
        match="Unable to load tokenizer 'repo/model' from its tokenizer assets",
    ) as error:
        corpus._get_tokenizer("repo/model")

    assert isinstance(error.value.__cause__, MemoryError)
    assert FakeTokenizer.calls == ["repo/model"]
    assert FakeAutoTokenizer.calls == [
        (("repo/model",), {"use_fast": True, "trust_remote_code": False})
    ]
