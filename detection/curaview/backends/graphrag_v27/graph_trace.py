"""Scoped GraphRAG 2.7.0 provider/cache capture; never edits vendor files.

The supported method signatures are checked from installed-source evidence.
Model methods open the HTTP scope in their executing thread, including FNLLM's
sync-to-async worker threads. This is a dedicated-process instrumentation scope.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from functools import wraps
from importlib import import_module, metadata
import inspect
from pathlib import Path
import threading

from curaview._99_tracing.storage.workflow_trace import TraceCaptureError

SUPPORTED_VERSION = "2.7.0"
_METHODS = {
    "OpenAIChatFNLLM": ("achat", "achat_stream"),
    "AzureOpenAIChatFNLLM": ("achat", "achat_stream"),
    "OpenAIEmbeddingFNLLM": ("aembed", "aembed_batch"),
    "AzureOpenAIEmbeddingFNLLM": ("aembed", "aembed_batch"),
    "LitellmChatModel": ("achat", "achat_stream"),
    "LitellmEmbeddingModel": ("aembed", "aembed_batch"),
    "JsonPipelineCache": ("get", "set"),
    "InMemoryCache": ("get", "set"),
}
_LEGACY_CLASSES = frozenset(name for name in _METHODS if not name.startswith("Litellm"))
_lock = threading.RLock()
_owner = None
_depth = 0
_originals = []
_classes = None
_scope_parent = None


def evidence_value(value):
    """Explicit lossless JSON conversion. No repr/default=str fallback."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Enum):
        return evidence_value(value.value)
    if isinstance(value, datetime):
        return {"type": "datetime", "iso8601": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "iso8601": value.isoformat()}
    if isinstance(value, timedelta):
        return {"type": "timedelta", "days": value.days, "seconds": value.seconds, "microseconds": value.microseconds}
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if type(value) in (tuple, list):
        return [evidence_value(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TraceCaptureError("GraphRAG evidence requires string dictionary keys")
        return {key: evidence_value(item) for key, item in value.items()}
    if isinstance(value, type) and callable(getattr(value, "model_json_schema", None)):
        return {"type": "pydantic schema", "name": value.__name__, "schema": evidence_value(value.model_json_schema())}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return evidence_value(dump(mode="json"))
        except TraceCaptureError:
            raise
        except Exception as error:
            raise TraceCaptureError("GraphRAG model JSON serialization failed") from error
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: evidence_value(getattr(value, field.name)) for field in fields(value)}
    raise TraceCaptureError("unsupported GraphRAG evidence object; explicit conversion required")


safe_to_json = evidence_value


def _load_classes():
    try:
        version = metadata.version("graphrag")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("GraphRAG 2.7.0 distribution is required for scoped capture") from error
    if version != SUPPORTED_VERSION:
        raise RuntimeError(f"unsupported GraphRAG version {version}; expected {SUPPORTED_VERSION}")
    providers = import_module("graphrag.language_model.providers.fnllm.models")
    result = {name: getattr(providers, name) for name in _METHODS if name.endswith("FNLLM")}
    result["LitellmChatModel"] = getattr(import_module(
        "graphrag.language_model.providers.litellm.chat_model"), "LitellmChatModel")
    result["LitellmEmbeddingModel"] = getattr(import_module(
        "graphrag.language_model.providers.litellm.embedding_model"), "LitellmEmbeddingModel")
    result["JsonPipelineCache"] = getattr(import_module("graphrag.cache.json_pipeline_cache"), "JsonPipelineCache")
    result["InMemoryCache"] = getattr(import_module("graphrag.cache.memory_pipeline_cache"), "InMemoryCache")
    return result


def _namespace(instance):
    if hasattr(instance, "_name"):
        return {"name": evidence_value(instance._name), "basis": "InMemoryCache._name"}
    storage = getattr(instance, "_storage", None)
    result = {"storage_class": type(storage).__name__, "basis": "storage attributes"}
    for attribute in ("_root_dir", "_base_dir", "_prefix", "_path"):
        if hasattr(storage, attribute):
            result[attribute] = evidence_value(getattr(storage, attribute))
    if len(result) == 2:
        result["namespace_observation"] = "storage namespace not exposed by supported attributes"
    return result


def _request(instance, original, args, kwargs, method, cache=False):
    try:
        bound = inspect.signature(original).bind(instance, *args, **kwargs)
        arguments = {key: value for key, value in bound.arguments.items() if key != "self"}
        payload = {"method": method, "arguments": evidence_value(arguments),
                   "provided_args": evidence_value(args), "provided_kwargs": evidence_value(kwargs)}
        if cache:
            payload["namespace"] = _namespace(instance)
            key = arguments["key"]
            if callable(getattr(instance, "_create_cache_key", None)):
                payload["effective_key"] = evidence_value(instance._create_cache_key(key))
            else:
                payload["effective_key"] = evidence_value(key)
        else:
            payload["config"] = evidence_value(instance.config)
        return payload
    except TraceCaptureError:
        raise
    except Exception as error:
        raise TraceCaptureError("GraphRAG request capture failed before invocation") from error


def _async_wrapper(store, original, stage, method, cache=False, root_parent=None):
    @wraps(original)
    async def wrapped(instance, *args, **kwargs):
        try:
            request = _request(instance, original, args, kwargs, method, cache)
        except TraceCaptureError:
            store.invalidate()
            raise
        with store.call(stage, request,parent_call_id=store.active_call_id or root_parent) as call:
            if cache:
                result = await original(instance, *args, **kwargs)
                payload = {"value": evidence_value(result)}
                if method == "get":
                    payload["hit"] = result is not None
                    payload["hit_basis"] = "get returned non-None; stored None is indistinguishable"
                else:
                    payload["set_value"] = request["arguments"]["value"]
                    payload["debug_data"] = request["arguments"].get("debug_data")
                    payload["storage_semantics"] = ("JSON result plus debug_data; None set is a no-op"
                        if type(instance).__name__ == "JsonPipelineCache" else "in-memory value; debug_data not stored")
                    if type(instance).__name__ == "JsonPipelineCache":
                        value = request["arguments"]["value"]
                        payload["storage_contents"] = ({"result": value, **(payload["debug_data"] or {})}
                            if value is not None else None)
                        payload["set_was_noop"] = value is None
                    else:
                        payload["storage_contents"] = request["arguments"]["value"]
                call.finish(payload)
                return result
            from curaview._99_tracing.capture.http_trace import capture_http
            with capture_http(store):
                result = await original(instance, *args, **kwargs)
                call.finish({"response": evidence_value(result), "boundary": "GraphRAG provider return"})
                return result
    return wrapped


def _stream_wrapper(store, original, stage, method, root_parent=None):
    @wraps(original)
    async def wrapped(instance, *args, **kwargs):
        try:
            request = _request(instance, original, args, kwargs, method)
        except TraceCaptureError:
            store.invalidate()
            raise
        with store.call(stage, request,parent_call_id=store.active_call_id or root_parent) as call:
            from curaview._99_tracing.capture.http_trace import capture_http
            count = 0
            with capture_http(store):
                stream = original(instance, *args, **kwargs)
                try:
                    async for chunk in stream:
                        call.record("provider_stream_chunk", evidence_value(chunk))
                        count += 1
                        yield chunk
                except GeneratorExit:
                    call.record("provider_stream_interrupted", {"chunks_observed": count, "stream_complete": False})
                    raise
                finally:
                    await stream.aclose()
                call.finish({"chunks_observed": count, "stream_complete": True,
                    "boundary": "GraphRAG returned text chunks", "provider_response_metadata": "not exposed by achat_stream"})
    return wrapped


@contextmanager
def capture_graphrag(store, *, classes=None, version=None):
    """Temporarily wrap the eight supported provider/cache classes.

    ``classes`` is a full name->class mapping for offline fixtures; legacy six
    FNLLM/cache fixtures remain supported, but partial LiteLLM maps are rejected.
    Production
    resolves installed GraphRAG classes and validates its distribution version.
    Nested scopes may share one store/classes; overlapping stores are rejected.
    """
    global _owner, _depth, _originals, _classes, _scope_parent
    if version is not None and version != SUPPORTED_VERSION:
        raise RuntimeError(f"unsupported GraphRAG version {version}")
    resolved = classes if classes is not None else _load_classes()
    if set(resolved) not in (set(_METHODS), _LEGACY_CLASSES):
        raise RuntimeError("GraphRAG capture requires all eight supported classes or all six legacy FNLLM/cache classes")
    methods_to_wrap = {name: _METHODS[name] for name in resolved}
    for name, methods in methods_to_wrap.items():
        for method in methods:
            candidate = getattr(resolved[name], method, None)
            required = inspect.isasyncgenfunction if method == "achat_stream" else inspect.iscoroutinefunction
            if not required(candidate):
                raise RuntimeError(f"unsupported GraphRAG capture method {name}.{method}")
    with _lock:
        if _depth and (_owner is not store or _classes != resolved):
            raise RuntimeError("overlapping GraphRAG capture stores/classes are not supported")
        if _depth and store.active_call_id != _scope_parent:
            raise RuntimeError('different GraphRAG parent scopes must run sequentially or in separate processes')
        if not _depth:
            _owner, _classes, _originals = store, resolved, []
            root_parent=store.active_call_id
            _scope_parent=root_parent
            try:
                pending = [(name, resolved[name], method, getattr(resolved[name], method), method in resolved[name].__dict__)
                    for name, methods in methods_to_wrap.items() for method in methods]
                for name, cls, method, original, owned in pending:
                    # Track inheritance precisely so restoration does not add attributes.
                    _originals.append((cls, method, original, owned))
                    stage = f"graphrag.{name}.{method}"
                    wrapper = (_stream_wrapper(store, original, stage, method,root_parent) if method == "achat_stream"
                        else _async_wrapper(store, original, stage, method, cache=method in ("get", "set"),root_parent=root_parent))
                    setattr(cls, method, wrapper)
            except BaseException:
                for cls, method, original, owned in reversed(_originals):
                    if owned:
                        setattr(cls, method, original)
                    else:
                        delattr(cls, method)
                _owner, _classes, _originals = None, None, []
                _scope_parent=None
                raise
        _depth += 1
    try:
        yield store
    finally:
        with _lock:
            _depth -= 1
            if not _depth:
                for cls, method, original, owned in reversed(_originals):
                    if owned:
                        setattr(cls, method, original)
                    else:
                        delattr(cls, method)
                _owner, _classes, _originals = None, None, []
                _scope_parent=None
