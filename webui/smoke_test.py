"""Test the local routes and report status mapping with synthetic inputs."""
from concurrent.futures import Future
import json
from types import SimpleNamespace
import tempfile
from pathlib import Path
from unittest.mock import patch

from webui import app as server


def immediate(fn, *args):
    result = Future()
    try:
        result.set_result(fn(*args))
    except BaseException as error:
        result.set_exception(error)
        raise
    return result


def run():
    with tempfile.TemporaryDirectory() as folder, patch.object(server, "work_dir", Path(folder)):
        client = server.app.test_client()
        for page in ("/", "/detection", "/evaluation", "/graph", "/logs"):
            page_response = client.get(page)
            assert page_response.status_code == 200
            assert "frame-ancestors 'none'" in page_response.headers["Content-Security-Policy"]
        assert client.post("/detection/api/run", json={"patient_id": "../test", "text": "Synthetic"}).status_code == 400

        with patch.object(server.executor, "submit", immediate):
            for expected in ("completed", "partial", "failed"):
                def detect(text, patient_id, directory):
                    if expected == "failed":
                        raise RuntimeError("Synthetic test")
                    return {"processing_status": expected, "patient_id": patient_id, "records": []}
                with patch.object(server, "run_detection", detect):
                    response = client.post("/detection/api/run", json={
                        "patient_id": "patient_1", "text": "The synthetic marker is blue."
                    })
                    assert response.status_code == 202
                    task_id = response.json["task_id"]
                    assert client.get(f"/detection/api/status/{task_id}").json["status"] == expected
                    assert client.get(f"/detection/api/result/{task_id}").json["status"] == expected
                    assert client.get(f"/logs/api/tail?task_id={task_id}").json["lines"] == []

        pair = {"patient_id": "patient_1", "sentence_id": "synthetic-1"}
        response = client.post("/evaluation/api/compare", json={
            "manifest": [{**pair, "y_true": 1}], "predictions": [{**pair, "y_pred": 1}]
        })
        assert response.json["sentence_micro"]["f1"] == 1
        with patch.object(server.subprocess, "run", return_value=SimpleNamespace(
            stdout=json.dumps({"response": "Synthetic response", "context": {}})
        )):
            response = client.post("/graph/api/search", json={
                "query": "Synthetic query", "patient_id": "patient_1"
            })
            assert response.json["response"] == "Synthetic response"


if __name__ == "__main__":
    run()
    print("Local web routes smoke test passed.")
