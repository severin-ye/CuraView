"""New derived detection records, not a recovered historical implementation.

The six-argument caller interface is preserved. Records identify caller messages
and parsed output explicitly; they do not claim to be the SDK/server raw response.
Actual HTTP and local-token evidence is captured by the linked TraceStore scopes.
"""
from functools import wraps
from pathlib import Path
import os
import threading
import uuid
import json
import asyncio

from curaview.common.source_layout import project_root
_REPO_ROOT = project_root()
_STORE = None
_STORE_LOCK = threading.Lock()
_GRAPH_SCOPE_LOCK = threading.Lock()


def get_trace_store():
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            from curaview._99_tracing.storage.workflow_trace import TraceStore
            configured = os.environ.get("CURAVIEW_LEGACY_TRACE_ROOT")
            root = Path(configured) if configured else (
                _REPO_ROOT / "outputs" / "traces" / uuid.uuid4().hex
            )
            _STORE = TraceStore(root)
        return _STORE


def _parsed_output(result):
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if isinstance(result, dict):
        return result
    raise TypeError("detection_result must be a structured model or dictionary")


def save_detection_distillation(patient_id, sentence, system_prompt, user_prompt,
                                detection_result, graphrag_evidence):
    store = get_trace_store()
    with store.call("legacy.detection.derived", {
        "patient_id": patient_id, "sentence": sentence,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "graphrag_evidence": graphrag_evidence,
        "provenance": "caller_supplied_messages_and_parsed_output",
        "is_sdk_raw_response": False,
        "implementation": "new_20261006_not_historical_recovery",
    }) as call:
        call.record("parsed_response", _parsed_output(detection_result))
        call.finish({"capture_scope": "derived_record"})


def trace_detection_call(function):
    """Capture each attempt, including failed query/model/storage calls."""
    @wraps(function)
    def wrapped(sentence, patient_id, llm_structured):
        store = get_trace_store()
        from curaview._99_tracing.capture.http_trace import capture_http
        with store.call("legacy.detection.attempt", {
            "patient_id": patient_id, "sentence": sentence,
            "interface": "detect_single_sentence",
        }) as call, capture_http(store):
            result = function(sentence, patient_id, llm_structured)
            call.record("parsed_detection", _parsed_output(result["detection_result"]))
            call.finish({"distillation_status": result.get("distillation_status")})
            return result
    return wrapped


def trace_detection_method(function):
    """Same observable attempt boundary for Meditron's error-result interface."""
    @wraps(function)
    def wrapped(self, sentence, patient_id, sentence_index=None):
        store = get_trace_store()
        from curaview._99_tracing.capture.http_trace import capture_http
        with store.call("legacy.meditron.detection.attempt", {
            "patient_id": patient_id, "sentence": sentence, "sentence_index": sentence_index,
        }) as call, capture_http(store):
            result = function(self, sentence, patient_id, sentence_index)
            if result["status"] == "success":
                call.record("parsed_detection", _parsed_output(result["detection_result"]))
                call.finish({"status": "success"})
            else:
                call.fail(result.get("error", "Detection failed"))
            return result
    return wrapped


def _query_context_view(value):
    if hasattr(value, 'to_json') and hasattr(value, 'columns'):
        return {'representation': 'dataframe_split_json',
                'data': json.loads(value.to_json(orient='split', date_format='iso'))}
    if isinstance(value, dict):
        return {str(key): _query_context_view(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_query_context_view(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unsupported native query context type; cannot silently omit evidence')


def record_native_query_result(response, context_data):
    store = get_trace_store()
    store.record(store.active_call_id, 'native_query_result', {
        'response': response, 'context_before_legacy_selection': _query_context_view(context_data),
        'representation': 'native_return_serialized_view_not_sdk_raw',
    })


def trace_query_call(function):
    @wraps(function)
    def wrapped(sentence, patient_id='patient_2'):
        store = get_trace_store()
        from curaview._99_tracing.capture.http_trace import capture_http
        with store.call('legacy.graph.query', {'sentence': sentence, 'patient_id': patient_id}) as call, capture_http(store):
            result = function(sentence, patient_id)
            call.record('tool_result', result)
            call.finish({'status': 'success'})
            return result
    return wrapped


def trace_async_query_call(function):
    """Native provider/cache capture for both direct async and sync-tool calls.

    GraphRAG 2.7 capture patches are scoped process-wide; serialize their scopes
    while yielding the event loop. Cancellation before acquisition owns no lock.
    """
    @wraps(function)
    async def wrapped(sentence, patient_id, method='local'):
        store = get_trace_store()
        from curaview.backends.graphrag_v27.graph_trace import capture_graphrag
        from curaview._99_tracing.capture.http_trace import capture_http
        while not _GRAPH_SCOPE_LOCK.acquire(blocking=False):
            await asyncio.sleep(0.01)
        try:
            with store.call('legacy.graph.query.native', {
                'sentence': sentence, 'patient_id': patient_id, 'method': method,
            }) as call, capture_graphrag(store), capture_http(store):
                result = await function(sentence, patient_id, method)
                call.record('selected_tool_result', result)
                call.finish({'status': 'success'})
                return result
        finally:
            _GRAPH_SCOPE_LOCK.release()
    return wrapped
