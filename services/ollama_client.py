from typing import Any, Dict, Optional, AsyncGenerator, Generator
import json
import httpx

from config import OLLAMA_URL, REQUEST_TIMEOUT


class OllamaError(Exception):
    pass


async def generate(
    prompt: str,
    *,
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    stream: bool = False,
    extra_options: Optional[Dict[str, Any]] = None,
) -> str:
    """Call Ollama's /api/generate and return the 'response' text.

    Raises OllamaError on non-200 or malformed responses.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }

    # Ollama accepts additional options under 'options'
    options: Dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if num_predict is not None:
        options["num_predict"] = num_predict

    if extra_options:
        options.update(extra_options)

    if options:
        payload["options"] = options

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
    except httpx.RequestError as e:
        raise OllamaError(f"Request to Ollama failed: {e}") from e

    if resp.status_code != 200:
        raise OllamaError(f"Ollama error {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except ValueError as e:
        raise OllamaError(f"Invalid JSON from Ollama: {e}") from e

    response_text = data.get("response")
    if not isinstance(response_text, str):
        raise OllamaError("Missing 'response' in Ollama output")

    return response_text


async def stream_generate(
    prompt: str,
    *,
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """Async generator yielding text chunks from Ollama streaming JSONL."""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    options: Dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if num_predict is not None:
        options["num_predict"] = num_predict
    if extra_options:
        options.update(extra_options)
    if options:
        payload["options"] = options

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise OllamaError(f"Ollama error {resp.status_code}: {text.decode(errors='ignore')}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = data.get("response")
                    if isinstance(chunk, str) and chunk:
                        yield chunk
    except httpx.RequestError as e:
        raise OllamaError(f"Request to Ollama failed: {e}") from e


def stream_generate_sync(
    prompt: str,
    *,
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Generator[str, None, None]:
    """Synchronous generator for streaming, useful for Gradio sync UI."""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    options: Dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if num_predict is not None:
        options["num_predict"] = num_predict
    if extra_options:
        options.update(extra_options)
    if options:
        payload["options"] = options

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        with client.stream("POST", OLLAMA_URL, json=payload) as resp:
            if resp.status_code != 200:
                raise OllamaError(f"Ollama error {resp.status_code}: {resp.text}")
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("response")
                if isinstance(chunk, str) and chunk:
                    yield chunk
