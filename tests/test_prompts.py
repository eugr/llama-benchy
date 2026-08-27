import json

import pytest

from llama_benchy.prompts import (
    HuggingFaceConversationPromptGenerator,
)


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token) for token in token_ids)


class _Corpus:
    def __init__(self):
        self.tokenizer = _CharTokenizer()

    def get_tokenizer(self):
        return self.tokenizer


def test_conversation_prompt_drops_completed_assistant_response():
    assert HuggingFaceConversationPromptGenerator._drop_trailing_assistant_messages([
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": "done"},
    ]) == [{"role": "user", "content": "request"}]


def test_huggingface_open_swe_normalization_preserves_tool_calls():
    tool_calls = [{
        "id": "call-1",
        "type": "function",
        "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"},
    }]
    messages = HuggingFaceConversationPromptGenerator._normalize_row({
        "trajectory": [
            {"role": "system", "content": "bootstrap"},
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "inspect first",
                "tool_calls": tool_calls,
            },
            {"role": "tool", "content": "files"},
        ]
    }, 8)

    assert messages[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
    }
    assert messages[3] == {"role": "tool", "content": "files"}


def test_huggingface_open_swe_normalization_preserves_tools():
    tool = {
        "type": "function",
        "function": {"name": "bash", "parameters": {"type": "object"}},
    }
    assert HuggingFaceConversationPromptGenerator._normalize_tools(
        {"tools": [json.dumps(tool)]}, 8
    ) == [tool]


def test_huggingface_generator_resolves_instances_and_sends_full_rows(monkeypatch, tmp_path):
    class _Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    def fake_get(url, **kwargs):
        if "/api/datasets/" in url:
            return _Response({"sha": "revision-1"})
        instance = "task-1" if "task-1" in kwargs["params"]["where"] else "task-0"
        row_id = int(instance[-1])
        trajectory = [
            {"role": "system", "content": "boot"},
            {"role": "user", "content": f"request-{row_id}" + "x" * 1000},
            {"role": "assistant", "content": "completed"},
        ]
        return _Response({"rows": [
            {"row_idx": row_id + 10, "row": {
                "trajectory_id": "unresolved-longer",
                "resolved": 0,
                "trajectory": trajectory * 2,
            }},
            {"row_idx": row_id, "row": {
                "trajectory_id": f"trajectory-{row_id}",
                "resolved": 1,
                "trajectory": trajectory,
                "tools": [json.dumps({
                    "type": "function",
                    "function": {"name": f"tool-{row_id}"},
                })],
            }},
        ]})

    monkeypatch.setattr("llama_benchy.prompts.requests.get", fake_get)
    monkeypatch.setenv("HOME", str(tmp_path))
    generator = HuggingFaceConversationPromptGenerator(
        _Corpus(), "owner/dataset?instance_id=task-0&instance_id=task-1"
    )

    warmup = generator.generate_batch(1, 20, run_index=0)[0]
    measured_zero = generator.generate_batch(1, 20, run_index=0)[0]
    measured_one = generator.generate_batch(1, 20, run_index=1)[0]
    concurrent = generator.generate_batch(2, 20, run_index=0)

    assert warmup.messages == measured_zero.messages
    assert measured_zero.messages[-1]["content"].startswith("request-0")
    assert measured_one.messages[-1]["content"].startswith("request-1")
    assert measured_zero.tools[0]["function"]["name"] == "tool-0"
    assert measured_one.tools[0]["function"]["name"] == "tool-1"
    assert [sample.messages[-1]["content"][:9] for sample in concurrent] == [
        "request-0", "request-0"
    ]
    with pytest.raises(ValueError, match="dataset row index is out of range"):
        generator.generate_batch(1, 20, run_index=2)
    assert generator.row_ids == [0, 1]
    assert generator.trajectory_ids == ["trajectory-0", "trajectory-1"]
    assert generator.dataset_revision == "revision-1"


def test_huggingface_selector_accepts_repeated_instance_ids():
    assert HuggingFaceConversationPromptGenerator._parse_selector(
        "nvidia/Open-SWE-Traces?subset=openhands&split=qwen35_122b"
        "&instance_id=task-a&instance_id=task-b"
    ) == (
        "nvidia/Open-SWE-Traces",
        "openhands",
        "qwen35_122b",
        ["task-a", "task-b"],
    )


def test_huggingface_generator_cache_hit_is_offline_and_plain_named(monkeypatch, tmp_path):
    class _Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    def fake_get(url, **kwargs):
        if "/api/datasets/" in url:
            return _Response({"sha": "revision-1"})
        row = {
            "trajectory_id": "trajectory-0",
            "resolved": 1,
            "trajectory": [
                {"role": "system", "content": "boot"},
                {"role": "user", "content": "request-0"},
                {"role": "assistant", "content": "completed"},
            ],
            "tools": [],
        }
        return _Response({"rows": [{"row_idx": 0, "row": row}]})

    monkeypatch.setattr("llama_benchy.prompts.requests.get", fake_get)
    monkeypatch.setenv("HOME", str(tmp_path))
    selector = "owner/dataset?instance_id=task-0"
    generator = HuggingFaceConversationPromptGenerator(_Corpus(), selector)

    # Cache file is plain-named (not a content hash).
    cache_dir = tmp_path / ".cache" / "llama-benchy" / "datasets"
    assert [f.name for f in cache_dir.iterdir()] == [
        "owner__dataset__default__train__task-0.json"
    ]
    assert generator.dataset_revision == "revision-1"

    # A second load from the same cache must not touch the network at all,
    # even if the upstream dataset revision has changed or the API is down.
    def offline_get(url, **kwargs):
        raise AssertionError(f"cache hit must not call the network: {url}")

    monkeypatch.setattr("llama_benchy.prompts.requests.get", offline_get)
    warm = HuggingFaceConversationPromptGenerator(_Corpus(), selector)
    assert warm.dataset_revision == "revision-1"
    assert warm.row_ids == generator.row_ids
    assert warm.trajectory_ids == generator.trajectory_ids


def test_huggingface_generator_accepts_legacy_bare_list_cache(monkeypatch, tmp_path):
    # Legacy cache files (pre-plain-naming) are a bare list of selections
    # with no revision metadata; they must still load, offline.
    def offline_get(url, **kwargs):
        raise AssertionError(f"cache hit must not call the network: {url}")

    monkeypatch.setattr("llama_benchy.prompts.requests.get", offline_get)
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_dir = tmp_path / ".cache" / "llama-benchy" / "datasets"
    cache_dir.mkdir(parents=True)
    (cache_dir / "owner__dataset__default__train__task-0.json").write_text(json.dumps([
        {
            "row_id": 0,
            "instance_id": "task-0",
            "trajectory_id": "trajectory-0",
            "messages": [{"role": "user", "content": "request-0"}],
            "tools": [],
        }
    ]))
    generator = HuggingFaceConversationPromptGenerator(
        _Corpus(), "owner/dataset?instance_id=task-0"
    )
    assert generator.dataset_revision is None
    assert generator.row_ids == [0]
    assert generator.trajectory_ids == ["trajectory-0"]
