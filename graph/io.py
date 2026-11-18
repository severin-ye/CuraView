"""Common data and exclusive run artifacts."""
import json
from pathlib import Path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def load_corpus(path):
    corpus = json.loads(read_json(path))
    patient = corpus.get("patient_id")
    docs = corpus.get("documents")
    if not isinstance(patient, str) or not patient.strip() or not isinstance(docs, list) or not docs:
        raise ValueError("one nonempty patient_id and a nonempty documents list are required")
    for doc in docs:
        if (not isinstance(doc, dict) or doc.get("patient_id", patient) != patient
                or any(not isinstance(doc.get(key), str) or not doc[key].strip()
                       for key in ("id", "title", "text"))):
            raise ValueError("documents require nonempty id, title and text for the same patient")
    if len({doc["id"] for doc in docs}) != len(docs):
        raise ValueError("document IDs must be unique")
    return {"patient_id": patient, "documents": [
        {key: doc[key] for key in ("id", "title", "text")} for doc in docs
    ]}
