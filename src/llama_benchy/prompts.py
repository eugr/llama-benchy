import uuid
import numpy as np
from typing import Tuple, List

from .corpus import TokenizedCorpus

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
