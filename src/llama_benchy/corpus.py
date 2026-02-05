import os
import hashlib
import requests
from transformers import AutoTokenizer

class TokenizedCorpus:
    def __init__(self, book_url: str, tokenizer_name: str, model_name: str):
        self.book_url = book_url
        self.tokenizer = self._get_tokenizer(model_name, tokenizer_name)
        self.tokens = self._load_data()

    def _get_tokenizer(self, model_name, tokenizer_name=None):
        try:
            name = tokenizer_name if tokenizer_name else model_name
            return AutoTokenizer.from_pretrained(name)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Falling back to 'gpt2' tokenizer as approximation.")
            return AutoTokenizer.from_pretrained("gpt2")

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
