"""Exercise public detection and trace contracts with no external calls."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def run():
    with tempfile.TemporaryDirectory() as folder:
        with patch.dict(os.environ, {"CURAVIEW_LEGACY_TRACE_ROOT": folder}):
            from curaview._03_detection.inference import agent
            from curaview._03_detection.parsing.output_contract import parse_strict_detection
            from curaview._03_detection.sentences.shared_splitter import split_into_sentences
            from curaview._03_detection.validation.detection_errors import GraphRAGQueryError
            from curaview._03_detection.validation.schemas import HallucinationDetectionResult
            from curaview._03_detection.validation.validators import DetectionValidator

            import httpx
            from curaview._99_tracing.capture.http_trace import _metadata
            request = httpx.Request("GET", "https://example.invalid/test?sig=private&api-version=v1")
            assert _metadata(request, False)["query"] == {"sig": "[redacted]", "api-version": "v1"}

            sentences = split_into_sentences(
                "The synthetic marker changed from blue to green. The synthetic marker was recorded."
            )
            assert len(sentences) == 2
            evidence = json.dumps({"response": "The synthetic marker was blue.", "context": {}})
            model_reply = HallucinationDetectionResult(
                reasoning="The supplied evidence states blue.",
                hallucination_status=True,
                hallucination_type="invented_fact",
                evidence_grade="E4",
            )
            def invoke(messages):
                assert sentences[0] in messages[1]["content"]
                assert evidence in messages[1]["content"]
                return model_reply

            model = SimpleNamespace(invoke=invoke)
            with patch.object(agent, "query_graphrag_for_sentence",
                              SimpleNamespace(invoke=lambda _: evidence)):
                output = agent.detect_single_sentence(sentences[0], "patient_1", model)
            assert output["detection_result"] == model_reply
            assert output["distillation_status"] == "saved"
            assert any(Path(folder).rglob("*.sqlite*"))

            with patch.object(agent, "query_graphrag_for_sentence",
                              SimpleNamespace(invoke=lambda _: '{"error":"synthetic"}')):
                try:
                    agent.detect_single_sentence(sentences[0], "patient_1", model)
                except GraphRAGQueryError:
                    pass
                else:
                    raise AssertionError("Query failures must not produce medical labels")

            assert DetectionValidator.validate_consistency(model_reply.hallucination_status, model_reply.evidence_grade)[0]
            strict_reply = model_reply.model_dump(mode="json")
            strict_reply["is_hallucination"] = strict_reply.pop("hallucination_status")
            assert parse_strict_detection(json.dumps(strict_reply))["status"] == "valid"
            strict_reply["is_hallucination"] = False
            assert parse_strict_detection(json.dumps(strict_reply))["error"] == "label_grade_conflict"


if __name__ == "__main__":
    run()
    print("Detection smoke test passed.")
