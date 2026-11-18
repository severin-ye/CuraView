"""Keep failed tool execution separate from valid (including empty) evidence."""
import json


class GraphRAGQueryError(RuntimeError):
    """A legacy tool returned an execution error rather than query evidence."""


def require_query_evidence(raw_result):
    """Validate the legacy text interface without changing successful evidence.

The recovered legacy tool encodes exceptions as JSON with an `error` key.
An empty context in a successful envelope is not a query failure.
"""
    if not isinstance(raw_result, str):
        raise GraphRAGQueryError("GraphRAG query returned a non-text result")
    try:
        envelope = json.loads(raw_result)
    except (ValueError, TypeError):
        envelope = None
    if isinstance(envelope, dict) and (
        "error" in envelope or envelope.get("status") in {"error", "failed"}
    ):
        raise GraphRAGQueryError("GraphRAG query returned an error envelope")
    if raw_result.strip().startswith("查询失败"):
        raise GraphRAGQueryError("GraphRAG query returned a failure marker")
    return raw_result
