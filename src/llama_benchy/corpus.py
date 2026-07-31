import os
import hashlib
import requests
from typing import Any, Optional


class LightweightTokenizer:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name: str) -> "LightweightTokenizer":
        from tokenizers import Tokenizer

        if os.path.isfile(name):
            tokenizer = Tokenizer.from_file(name)
        else:
            tokenizer_json = os.path.join(name, "tokenizer.json")
            if os.path.isdir(name) and os.path.exists(tokenizer_json):
                tokenizer = Tokenizer.from_file(tokenizer_json)
            else:
                tokenizer = Tokenizer.from_pretrained(name)
        return cls(tokenizer)

    def encode(self, text: str, add_special_tokens: bool = False):
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

    def decode(self, token_ids):
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)


class TokenizedCorpus:
    def __init__(self, book_url: str, tokenizer_name: Optional[str], model_name: str):
        self.book_url = book_url
        self.tokenizer = self._get_tokenizer(model_name, tokenizer_name)
        self.tokens = self._load_data()

    def _get_transformers_tokenizer(self, name: str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            name,
            use_fast=True,
            trust_remote_code=False,
        )

    def _get_tokenizer(self, model_name: str, tokenizer_name: Optional[str] = None):
        name = tokenizer_name if tokenizer_name else model_name
        lightweight_error = None

        try:
            return LightweightTokenizer.from_pretrained(name)
        except Exception as e:
            lightweight_error = e

        try:
            return self._get_transformers_tokenizer(name)
        except Exception as auto_error:
            raise RuntimeError(
                f"Unable to load tokenizer '{name}' from its tokenizer assets "
                f"or Transformers configuration. Lightweight tokenizer error: "
                f"{lightweight_error}. AutoTokenizer error: {auto_error}."
            ) from auto_error

    def _load_data(self):
        try:
            # Create cache directory if it doesn't exist
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "llama-benchy")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Generate hash of the URL for the filename
            url_hash = hashlib.md5(self.book_url.encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{url_hash}.txt")
            
            if os.path.exists(cache_file):
                print(f"Loading text from cache: {cache_file}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                print(f"Downloading book from {self.book_url}...")
                response = requests.get(self.book_url)
                response.raise_for_status()
                text = response.text
                # Basic cleanup
                start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
                if start_idx != -1:
                    text = text[start_idx:]
                
                # Save to cache
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved text to cache: {cache_file}")
                
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            print(f"Error downloading or processing book: {e}")
            exit(1)

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokens(self):
        return self.tokens

    def __len__(self):
        return len(self.tokens)
