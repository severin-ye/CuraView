"""Fixed OpenAI-compatible transport for GraphRAG 3.3."""
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime, timezone

from .io import write_json

import httpx
from graphrag_llm.completion import LLMCompletion
from graphrag_llm.embedding import LLMEmbedding
from graphrag_llm.tokenizer.tiktoken_tokenizer import TiktokenTokenizer
from graphrag_llm.types import LLMCompletionResponse, LLMEmbeddingResponse
from graphrag_llm.utils import structure_completion_response

PROVIDER = "curaview_fixed"
CHAT_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DIMENSIONS = 1024


class Transport:
    """Retain client-visible request and response in an exclusive local run."""

    def __init__(self, trace_dir, *, timeout=180):
        self.directory = Path(trace_dir)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.endpoint = os.environ.get("CURAVIEW_API_BASE", DEFAULT_ENDPOINT).rstrip("/")
        url = urlsplit(self.endpoint)
        if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("use an HTTP(S) API base URL without authentication or parameters")
        self.key = os.environ.get("GRAPHRAG_API_KEY")
        if not self.key:
            raise ValueError("GRAPHRAG_API_KEY is required")
        self.timeout = timeout
        self.calls = 0
        self.failed = False

    def request(self, resource, body):
        if self.failed:
            raise RuntimeError("a prior provider call failed")
        self.calls += 1
        prefix = self.directory / f"{self.calls:04d}"
        write_json(prefix.with_suffix(".request.json"), {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "endpoint": resource, "timeout_seconds": self.timeout, "body": body,
        })
        try:
            response = httpx.post(
                f"{self.endpoint}/{resource}", json=body, timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.key}"},
            )
            cleaned = response.text.replace(self.key, "[redacted]")
            write_json(prefix.with_suffix(".response.json"), {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status_code, "body": cleaned,
            })
            response.raise_for_status()
            result = json.loads(cleaned)
            return result
        except Exception as error:
            self.failed = True
            write_json(prefix.with_suffix(".error.json"), {"error_type": type(error).__name__})
            raise RuntimeError(f"provider {resource} call {self.calls} failed") from None


def register_transport(transport):
    from graphrag_llm.completion.completion_factory import register_completion
    from graphrag_llm.embedding.embedding_factory import register_embedding

    class Base:
        def __init__(self, *, model_config, metrics_store, cache=None, **kwargs):
            self.config = model_config
            self._metrics_store = metrics_store
            self._tokenizer = TiktokenTokenizer(encoding_name="cl100k_base")
            self.cache = cache

        @property
        def metrics_store(self):
            return self._metrics_store

        @property
        def tokenizer(self):
            return self._tokenizer

        async def cached(self, resource, body):
            if self.cache is None:
                return transport.request(resource, body)
            key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            result = await self.cache.get(key)
            if result is None:
                result = transport.request(resource, body)
                await self.cache.set(key, result)
            return result

    class Completion(Base, LLMCompletion):
        def body(self, kwargs):
            messages = kwargs["messages"]
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            messages = [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in messages]
            return chat_body(messages, max_tokens=self.config.call_args["max_tokens"],
                             structured=kwargs.get("response_format") is not None)

        def completion(self, **kwargs):
            response = transport.request("chat/completions", self.body(kwargs))
            return parse_completion(response, kwargs.get("response_format"))

        async def completion_async(self, **kwargs):
            response = await self.cached("chat/completions", self.body(kwargs))
            return parse_completion(response, kwargs.get("response_format"))

    class Embedding(Base, LLMEmbedding):
        def body(self, kwargs):
            texts = kwargs["input"]
            texts = [texts] if isinstance(texts, str) else texts
            if not 1 <= len(texts) <= 10:
                raise ValueError("native embedding batch must contain 1..10 texts")
            return {"model": EMBEDDING_MODEL, "input": texts,
                    "dimensions": DIMENSIONS, "encoding_format": "float"}

        def embedding(self, **kwargs):
            body = self.body(kwargs)
            return parse_embedding(transport.request("embeddings", body), len(body["input"]))

        async def embedding_async(self, **kwargs):
            body = self.body(kwargs)
            return parse_embedding(await self.cached("embeddings", body), len(body["input"]))

    register_completion(PROVIDER, Completion)
    register_embedding(PROVIDER, Embedding)


def chat_body(messages, *, max_tokens, structured=False):
    body = {"model": CHAT_MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False, "enable_thinking": False}
    if structured:
        body["response_format"] = {"type": "json_object"}
    return body


def parse_completion(raw, response_format=None):
    response = LLMCompletionResponse(**raw)
    if len(response.choices) != 1 or response.choices[0].finish_reason != "stop" or not response.content.strip():
        raise ValueError("empty or incomplete completion")
    if response_format is not None:
        response.formatted_response = structure_completion_response(response.content, response_format)
    return response


def parse_embedding(raw, batch_size):
    response = LLMEmbeddingResponse(**raw)
    indices = sorted(item.index for item in response.data)
    if indices != list(range(batch_size)):
        raise ValueError("embedding response indices mismatch")
    response.data.sort(key=lambda item: item.index)
    if any(len(item.embedding) != DIMENSIONS or any(not math.isfinite(v) for v in item.embedding)
           for item in response.data):
        raise ValueError("invalid embedding dimensions or values")
    return response
