"""Synthetic checks for patient preparation, atomic facts and native 2.7 inputs."""
import asyncio
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from .prepare_v27 import prepare
from .test_native import CORPUS
from .tools.atomic_facts import evaluate_atomic_facts
from .tools.prepare_ehr import csv_documents, jsonl_documents


class OfflineToolsTest(unittest.TestCase):
    def test_sources_keep_only_requested_patient_and_evidence(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            csv_file = root / "source.csv"
            csv_file.write_text('subject_id,text\n42,"First drawer\nRed token"\n17,Other patient\n',
                                encoding="utf-8", newline="")
            self.assertEqual([row["text"] for row in csv_documents([csv_file], "42")],
                             ["First drawer\nRed token"])
            csv_file.write_text('subject_id,text,label\n42,First drawer,E1\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                csv_documents([csv_file], "42")
            records = [
                {"universal": {"subject_id": 42}, "discharge": {"text": "Red token"},
                 "radiology": {"text": "First drawer"}, "discharge_target": {"text": "Secret output"}},
                {"universal": {"subject_id": 17}, "discharge": {"text": "Other patient"}},
            ]
            json_file = root / "source.jsonl"
            json_file.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
            source_text = [row["text"] for row in jsonl_documents(json_file, "42", "discharge-radiology")]
            self.assertEqual(source_text, ["Red token\n\n[Radiology Report]\nFirst drawer"])

    def test_atomic_facts_follow_reference_obligations(self):
        fact = {"patient": "SYNTHETIC-P1", "event": "Event-A", "subject": "red token",
                "predicate": "belongs in", "value": "first drawer", "unit": None,
                "polarity": "positive", "time": None,
                "source_anchor": {"document_id": "Note-A", "anchor": "red token"}}
        reference = {
            "complete": True, "patients": ["SYNTHETIC-P1"],
            "documents": [{"patient": "SYNTHETIC-P1", "document_id": "Note-A"}],
            "facts": [fact], "description_obligations": [fact], "community_obligations": [fact],
        }
        candidate = {"facts": [fact], "description_facts": [fact], "community_facts": [fact]}
        result = evaluate_atomic_facts(reference, candidate)
        self.assertTrue(result["eligible"])
        self.assertEqual(set(result["metrics"].values()), {1.0})
        missing = evaluate_atomic_facts(reference, {**candidate, "community_facts": []})
        self.assertEqual(missing["metrics"]["summary_retention"], 0.0)


class LegacyInputsTest(unittest.TestCase):
    def test_native_load_and_chunk(self):
        try:
            installed = version("graphrag")
        except PackageNotFoundError:
            installed = None
        if installed != "2.7.0":
            self.skipTest("needs GraphRAG 2.7.0")
        with patch.dict("os.environ", {"GRAPHRAG_API_KEY": "synthetic-offline-key",
                                       "LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            import pandas as pd
            from graphrag.api import build_index
            from graphrag.config.load_config import load_config
            with TemporaryDirectory() as temp:
                root = Path(temp)
                corpus = root / "corpus.json"
                corpus.write_text(json.dumps(CORPUS), encoding="utf-8")
                index = prepare(corpus, root / "project")
                config = load_config(index)
                config.workflows = ["load_input_documents", "create_base_text_units"]
                result = None
                try:
                    result = asyncio.run(build_index(config))
                finally:
                    logging.shutdown()
                self.assertTrue(all(not operation.errors for operation in result), result)
                self.assertEqual(len(result), 2)
                units = pd.read_parquet(index / "output" / "text_units.parquet")
                self.assertEqual(set(units["text"]), {row["text"] for row in CORPUS["documents"]})


if __name__ == "__main__":
    unittest.main()
