import asyncio
import time
import uuid
import json
import logging
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_server")

app = FastAPI()

# Configuration constants
PROMPT_SPEED_TPS = 1000.0
GEN_SPEED_TPS = 50.0

# Global Tokenizer Cache
TOKENIZERS = {}
DEFAULT_TOKENIZER_NAME = "gpt2"

def get_tokenizer(model_name: str):
    """
    Get cached tokenizer or load it.
    Fallback to gpt2 for 'test', 'mock-model', or if load fails.
    """
    # Normalize model name for lookup
    if model_name in ["test", "mock-model"]:
        model_name = DEFAULT_TOKENIZER_NAME
    
    if model_name in TOKENIZERS:
        return TOKENIZERS[model_name]
    
    try:
        logger.info(f"Loading tokenizer for: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for {model_name}: {e}. Falling back to {DEFAULT_TOKENIZER_NAME}")
        # Try loading default if distinct
        if model_name != DEFAULT_TOKENIZER_NAME:
            try:
                tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)
            except Exception:
                # If even default fails (network?), we might need a dummy fallback, 
                # but transformers usually caches gpt2. 
                # For now let's assume gpt2 works as per env check.
                raise e
        else:
            raise e

    TOKENIZERS[model_name] = tokenizer
    return tokenizer

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 10
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    # Loose fields
    cache_prompt: Optional[bool] = True
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1

def count_tokens(text: str, model_name: str) -> int:
    """Count tokens using the appropriate tokenizer."""
    if not text:
        return 0
    
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))

@app.get("/models")
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "mock-model",
                "object": "model",
                "created": 1677610602,
                "owned_by": "mock"
            }
        ]
    }

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    start_proc_time = time.perf_counter()

    # Analyze messages for token counting and prefix caching logic
    system_tokens = 0
    user_tokens = 0
    other_tokens = 0
    
    has_system = False
    has_user = False
    
    # Pre-warm/Get tokenizer for consistency
    # (Doing this one by one might be slightly inefficient but accurate)
    
    for msg in request.messages:
        t_count = count_tokens(msg.content, request.model)
        if msg.role == "system":
            system_tokens += t_count
            has_system = True
        elif msg.role == "user":
            user_tokens += t_count
            has_user = True
        else:
            other_tokens += t_count

    total_prompt_tokens = system_tokens + user_tokens + other_tokens
    
    # Emulate Prompt Processing Logic
    tokens_to_process = total_prompt_tokens
    
    if has_system:
        if user_tokens > 0:
            # Case: System + User (with content) -> System cached, User processed + lookup overhead
            tokens_to_process = user_tokens + other_tokens + 1
        else:
            # Case: System + User (empty) -> Context preload / Warmup -> Full System processing
            tokens_to_process = total_prompt_tokens
            
    if request.cache_prompt is False:
        tokens_to_process = total_prompt_tokens

    # Calculate processing delay
    prompt_delay = tokens_to_process / PROMPT_SPEED_TPS
    
    # Calculate generation delay parameters
    num_completion_tokens = request.max_tokens if request.max_tokens else 10
    token_interval = 1.0 / GEN_SPEED_TPS

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    # Log the decision for debugging
    logger.info(f"Model: {request.model}, Prompt Tokens: {total_prompt_tokens} (Sys: {system_tokens}, User: {user_tokens}), Delay tokens: {tokens_to_process}, Delay: {prompt_delay:.4f}s")
    
    # Simulate Prompt Processing Delay using drift correction
    # Account for time spent in token counting and logic
    elapsed_proc = time.perf_counter() - start_proc_time
    adjusted_prompt_delay = max(0.0, prompt_delay - elapsed_proc)
    
    if adjusted_prompt_delay > 0:
        await asyncio.sleep(adjusted_prompt_delay)
    
    if request.stream:
        async def event_generator():
            # Generate tokens
            stream_start_time = time.perf_counter()
            for i in range(num_completion_tokens):
                target_time = stream_start_time + ((i + 1) * token_interval)
                now = time.perf_counter()
                sleep_duration = target_time - now
                
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
                
                token_text = "mock "
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Final finish chunk
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
            # Usage chunk if requested
            if request.stream_options and request.stream_options.get("include_usage"):
                usage_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "usage": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": num_completion_tokens,
                        "total_tokens": total_prompt_tokens + num_completion_tokens
                    },
                    "choices": [] 
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    else:
        # Non-streaming
        await asyncio.sleep(num_completion_tokens * token_interval)
        
        response_text = "mock " * num_completion_tokens
        
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": num_completion_tokens,
                "total_tokens": total_prompt_tokens + num_completion_tokens
            }
        }

if __name__ == "__main__":
    # Pre-load gpt2 to avoid runtime delay on first request in tests
    try:
        print("Pre-loading tokenizer...")
        AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Warning: Could not pre-load tokenizer: {e}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
