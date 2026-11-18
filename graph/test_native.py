"""Provider-free network integration test using a loopback OpenAI-compatible fixture."""
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest


CORPUS = {"patient_id": "SYNTHETIC-P1", "documents": [
    {"id": "Note-A", "title": "Invented note A", "text": "The red token is assigned to the first drawer."},
    {"id": "Note-B", "title": "Invented note B", "text": "The blue token is assigned to the second drawer."},
]}


def complete(body):
    prompt = body["messages"][0]["content"]
    if len(body["messages"]) == 2:
        return json.dumps({"reasoning": "The record [S1] directly supports the statement.",
                           "is_hallucination": False, "hallucination_type": None, "evidence_grade": "E1"})
    if "response_format" in body:
        return json.dumps({"title": "Synthetic community", "summary": "The invented tokens and drawers.",
                           "findings": [{"summary": "Tokens", "explanation": "A fixture relation."}],
                           "rating": 1.0, "rating_explanation": "Synthetic test."})
    if "first drawer" in prompt or "second drawer" in prompt:
        color, drawer = ("RED", "FIRST") if "first drawer" in prompt else ("BLUE", "SECOND")
        return (f'("entity"<|>{color} TOKEN<|>TREATMENT<|>Synthetic {color} token)##'
                f'("entity"<|>{drawer} DRAWER<|>LOCATION<|>Synthetic {drawer} drawer)##'
                f'("relationship"<|>{color} TOKEN<|>{drawer} DRAWER<|>Assigned<|>10)'
                '<|COMPLETE|>')
    return "Synthetic description."


class FakeProvider(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            rows = []
            for i, value in enumerate(body["input"]):
                vector = [0.0] * body["dimensions"]
                vector[0 if "red" in value.lower() else 1] = 1.0
                vector[2 + hashlib.sha256(value.encode()).digest()[0]] = .2
                rows.append({"object": "embedding", "index": i, "embedding": vector})
            response = {"object": "list", "model": body["model"], "data": rows,
                        "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        else:
            response = {"id": "synthetic", "object": "chat.completion", "created": 1,
                        "model": body["model"], "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": complete(body)},
                             "finish_reason": "stop"}
                        ], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class NativeSmokeTest(unittest.TestCase):
    def test_graphs_and_paired_comparison(self):
        with TemporaryDirectory() as temp, ThreadingHTTPServer(("127.0.0.1", 0), FakeProvider) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                root = Path(temp)
                env = {**os.environ, "GRAPHRAG_API_KEY": "synthetic-local-key",
                       "CURAVIEW_API_BASE": f"http://127.0.0.1:{server.server_port}/v1",
                       "LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHONDONTWRITEBYTECODE": "1"}

                def run(*args, expected=0):
                    result = subprocess.run([sys.executable, "-m", *map(str, args)],
                                            env=env, capture_output=True, text=True,
                                            encoding="utf-8", errors="replace", timeout=180)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    return result

                corpus = root / "corpus.json"
                corpus.write_text(json.dumps(CORPUS), encoding="utf-8")
                for arm in ("generic", "domain"):
                    run("graph.native", "build", "--corpus", corpus, "--out", root / arm,
                        "--arm", arm)
                run("graph.native", "query", "--graph", root / "generic",
                    "--out", root / "query", "--text", "Where is the red token?")
                native = json.loads((root / "query" / "context.json").read_text(encoding="utf-8"))
                self.assertTrue(native["text_unit_ranking"])
                self.assertEqual(native["answer_generation"], "Not Run")
                cases = root / "cases.json"
                cases.write_text(json.dumps([{
                    "sample_id": "SYNTHETIC-Q1", "sentence": CORPUS["documents"][0]["text"],
                    "reference": {"source_grade": "E1", "source_quotes": [
                        {"evidence_id": "Note-A", "quote": "red token"}
                    ]},
                }]), encoding="utf-8")
                strategy = root / "strategy.txt"
                strategy.write_text("Use the provided source records.", encoding="utf-8")
                run("comparison.paired", "--generic", root / "generic", "--domain", root / "domain",
                    "--cases", cases, "--strategy", strategy, "--out", root / "paired")
                report = json.loads((root / "paired" / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["actual_calls"], 5)
                self.assertEqual(set(report["summary"]), {"flat_dense", "generic_graph", "domain_graph"})
                self.assertEqual([row["prediction"]["status"] for row in report["rows"]], ["valid"] * 3)
                self.assertTrue(all(row["reference_proxy"]["quoted_fact_coverage"] == 1 for row in report["rows"]))
                self.assertFalse(report["prompt_only_causal_attribution"])
                self.assertTrue(all("synthetic-local-key" not in p.read_text(encoding="utf-8")
                                    for p in root.rglob("*.json")))
                (root / "generic" / "prompts" / "generic.extract.txt").write_text("changed", encoding="utf-8")
                result = run("graph.native", "query", "--graph", root / "generic",
                             "--out", root / "invalid", "--text", "Where is the red token?", expected=1)
                self.assertIn("graph files differ from the build manifest", result.stderr)
                self.assertFalse((root / "invalid").exists())
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()

