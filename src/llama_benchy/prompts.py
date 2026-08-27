import uuid
import copy
import json
import os
import numpy as np
import requests
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple, List, Optional, cast
from urllib.parse import parse_qs, urlsplit

from .corpus import TokenizedCorpus


@dataclass
class PromptSample:
    context_text: str
    prompt_text: str
    messages: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None

    def __iter__(self) -> Iterator[str]:
        # Preserve the historical (context, prompt) unpacking contract used by
        # the prefix-caching runner paths. Conversation mode rejects those
        # paths at CLI validation time.
        yield self.context_text
        yield self.prompt_text

class PromptGenerator:
    def __init__(self, corpus: TokenizedCorpus):
        self.corpus = corpus
        self.tokenizer = corpus.get_tokenizer()
        self.all_tokens = corpus.get_tokens()

    def generate(self, prompt_tokens: int, context_tokens: int = 0, no_cache: bool = False) -> Tuple[str, str]:
        """
        Generates a single (context, prompt) pair.
        """
        suffix = ""
        suffix_len = 0
        if no_cache:
            suffix = f" {uuid.uuid4()}"
            suffix_len = len(self.tokenizer.encode(suffix, add_special_tokens=False))
        
        # Adjust prompt tokens to fetch from text
        text_prompt_tokens = max(0, prompt_tokens - suffix_len)
        
        # Create a pool of tokens large enough
        total_needed = text_prompt_tokens + context_tokens
        
        # Create a local reference to tokens to potentially extend
        current_tokens = self.all_tokens
        
        if len(current_tokens) < total_needed:
            # Repeat tokens if not enough
            current_tokens = current_tokens * (total_needed // len(current_tokens) + 2)
        
        # Pick a random start position
        max_start = len(current_tokens) - total_needed
        start_idx = np.random.randint(0, max_start)
        
        selected_tokens = current_tokens[start_idx : start_idx + total_needed]
        
        context_text = self.tokenizer.decode(selected_tokens[:context_tokens]) if context_tokens > 0 else ""
        prompt_text = self.tokenizer.decode(selected_tokens[context_tokens:])
        
        if no_cache:
            prompt_text += suffix
            
        return context_text, prompt_text

    def generate_batch(self, batch_size: int, prompt_tokens: int, context_tokens: int = 0, no_cache: bool = False) -> List[Tuple[str, str]]:
        """
        Generates a batch of (context, prompt) pairs.
        """
        return [self.generate(prompt_tokens, context_tokens, no_cache) for _ in range(batch_size)]


class HuggingFaceConversationPromptGenerator:
    """Load deterministic agent trajectories through the HF dataset server."""

    def __init__(
        self,
        corpus: TokenizedCorpus,
        selector: str,
    ):
        self.corpus = corpus
        (
            self.dataset,
            self.dataset_config,
            self.dataset_split,
            instances,
        ) = self._parse_selector(selector)
        selections, self.dataset_revision = self._load_instances(instances)
        self.row_ids = [int(item["row_id"]) for item in selections]
        self.instance_ids = [str(item["instance_id"]) for item in selections]
        self.trajectory_ids = [str(item["trajectory_id"]) for item in selections]
        self.conversations = [
            self._drop_trailing_assistant_messages(item["messages"])
            for item in selections
        ]
        self.tools = [item["tools"] for item in selections]
        if any(not messages for messages in self.conversations):
            raise ValueError("a selected dataset row has no request-ending conversation")

    @staticmethod
    def _parse_selector(selector: str) -> Tuple[str, str, str, List[str]]:
        parsed = urlsplit(selector)
        dataset = parsed.path.strip("/")
        params = parse_qs(parsed.query, keep_blank_values=True)
        allowed = {"subset", "split", "instance_id"}
        unknown = set(params) - allowed
        if "/" not in dataset or unknown:
            detail = f": {sorted(unknown)[0]}" if unknown else ""
            raise ValueError(f"invalid dataset selector{detail}")
        subset = params.get("subset", ["default"])
        split = params.get("split", ["train"])
        instances = list(dict.fromkeys(params.get("instance_id", [])))
        if len(subset) != 1 or len(split) != 1 or not all(instances):
            raise ValueError("dataset selector has invalid parameters")
        if not instances:
            raise ValueError("dataset selector requires instance_id")
        return dataset, subset[0], split[0], instances

    @staticmethod
    def _drop_trailing_assistant_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages = list(messages)
        while messages and messages[-1].get("role") == "assistant":
            messages.pop()
        return messages

    @staticmethod
    def _content_as_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _dataset_revision(dataset: str) -> str:
        response = requests.get(
            f"https://huggingface.co/api/datasets/{dataset}", timeout=30
        )
        response.raise_for_status()
        revision = response.json().get("sha")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"Hugging Face did not report a revision for {dataset}")
        return revision

    def _cache_path(self) -> str:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "llama-benchy", "datasets"
        )
        os.makedirs(cache_dir, exist_ok=True)
        # Plain, human-readable name: dataset__config__split__instance[+instance].
        # Deliberately NOT keyed on the live dataset revision: a cached
        # trajectory stays reusable after the upstream dataset is updated
        # (the datasets-server API may no longer serve the old split), and a
        # cache hit needs no network access at all.
        name = "__".join([
            self.dataset.replace("/", "__"),
            self.dataset_config,
            self.dataset_split,
            "+".join(self.instance_ids),
        ])
        return os.path.join(cache_dir, f"{name}.json")

    def _load_instances(
        self, instances: List[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        self.instance_ids = instances
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Legacy cache files are a bare list of selections (no revision).
            if isinstance(cached, list):
                return cast(List[Dict[str, Any]], cached), None
            revision = cached.get("revision")
            return (
                cast(List[Dict[str, Any]], cached.get("selections", [])),
                revision if isinstance(revision, str) else None,
            )

        revision = self._dataset_revision(self.dataset)
        selections: List[Dict[str, Any]] = []
        for instance in instances:
            escaped = instance.replace("'", "''")
            params: Dict[str, str | int] = {
                "dataset": self.dataset,
                "config": self.dataset_config,
                "split": self.dataset_split,
                "where": f'"instance_id" = \'{escaped}\'',
                "length": 100,
            }
            response = requests.get(
                "https://datasets-server.huggingface.co/filter",
                params=params,
                timeout=180,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("error"):
                raise ValueError(
                    f"could not resolve dataset instance {instance}: {body['error']}"
                )
            candidates = body.get("rows", [])
            if not candidates:
                raise ValueError(
                    f"dataset instance not found in {self.dataset_config}/{self.dataset_split}: "
                    f"{instance}"
                )
            resolved = [
                candidate for candidate in candidates
                if candidate.get("row", {}).get("resolved") == 1
            ]
            pool = resolved or candidates
            selected = min(
                pool,
                key=lambda candidate: (
                    -len(candidate.get("row", {}).get("trajectory", [])),
                    str(candidate.get("row", {}).get("trajectory_id", "")),
                ),
            )
            row = selected.get("row", {})
            trajectory_id = row.get("trajectory_id")
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise ValueError(f"dataset instance {instance} has no trajectory_id")
            selections.append({
                "row_id": int(selected["row_idx"]),
                "instance_id": instance,
                "trajectory_id": trajectory_id,
                "messages": self._normalize_row(row, int(selected["row_idx"])),
                "tools": self._normalize_tools(row, int(selected["row_idx"])),
            })
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "adapter_version": 4,
                "dataset": self.dataset,
                "config": self.dataset_config,
                "split": self.dataset_split,
                "instances": list(instances),
                "revision": revision,
                "selections": selections,
            }, f, ensure_ascii=False)
        print(f"Cached {len(selections)} dataset trajectories: {cache_path}")
        return selections, revision

    @staticmethod
    def _normalize_row(row: Dict[str, Any], row_id: int) -> List[Dict[str, Any]]:
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"dataset row {row_id} has no trajectory array")
        messages: List[Dict[str, Any]] = []
        for entry in trajectory:
            if not isinstance(entry, dict):
                raise ValueError(f"dataset row {row_id} contains a non-object trajectory entry")
            role = entry.get("role")
            content = entry.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(f"dataset row {row_id} contains an invalid trajectory entry")
            message: Dict[str, Any] = {
                "role": role,
                "content": content,
            }
            if isinstance(entry.get("tool_calls"), list) and entry["tool_calls"]:
                message["tool_calls"] = entry["tool_calls"]
            messages.append(message)
        return messages

    @staticmethod
    def _normalize_tools(row: Dict[str, Any], row_id: int) -> List[Dict[str, Any]]:
        raw_tools = row.get("tools")
        if not isinstance(raw_tools, list):
            raise ValueError(f"dataset row {row_id} has no tools array")
        tools: List[Dict[str, Any]] = []
        for raw_tool in raw_tools:
            try:
                tool = json.loads(raw_tool) if isinstance(raw_tool, str) else raw_tool
            except json.JSONDecodeError as exc:
                raise ValueError(f"dataset row {row_id} contains an invalid tool") from exc
            if not isinstance(tool, dict):
                raise ValueError(f"dataset row {row_id} contains an invalid tool")
            tools.append(tool)
        return tools

    def generate_batch(
        self,
        batch_size: int,
        prompt_tokens: int,
        context_tokens: int = 0,
        no_cache: bool = False,
        run_index: int = 0,
    ) -> List[PromptSample]:
        if context_tokens:
            raise ValueError("dataset conversations do not support --depth; use --depth 0")
        if run_index < 0 or run_index >= len(self.conversations):
            raise ValueError("dataset row index is out of range")
        samples = []
        for _ in range(batch_size):
            messages = copy.deepcopy(self.conversations[run_index])
            if no_cache:
                nonce = f"\n<!-- llama-benchy nonce: {uuid.uuid4()} -->"
                messages[0]["content"] = (
                    self._content_as_text(messages[0].get("content")) + nonce
                )
            samples.append(PromptSample(
                context_text="",
                prompt_text="",
                messages=messages,
                tools=copy.deepcopy(self.tools[run_index]),
            ))
        return samples
