"""Prepare frozen CuraView rows for official external baseline runners.

The script performs no model inference.  It converts the existing CSV contract
to one-record-per-line text files used by QAGS and a label manifest shared by
all official baselines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "patient_id",
    "sentence_id",
    "sentence",
    "reference_text",
    "y_true",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def prepare_inputs(source_csv: Path, output_dir: Path) -> dict[str, Any]:
    source_csv = source_csv.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        rows = list(reader)

    seen: set[tuple[str, int]] = set()
    labels: list[dict[str, Any]] = []
    candidates: list[str] = []
    references: list[str] = []
    label_counts: Counter[str] = Counter()
    patients: set[str] = set()

    for index, row in enumerate(rows, start=2):
        patient_id = row["patient_id"].strip()
        try:
            sentence_id = int(row["sentence_id"])
            y_true = int(row["y_true"])
        except ValueError as exc:
            raise ValueError(f"invalid integer field at CSV row {index}") from exc
        if y_true not in (0, 1):
            raise ValueError(f"invalid y_true={y_true} at CSV row {index}")
        key = (patient_id, sentence_id)
        if key in seen:
            raise ValueError(f"duplicate row key: {patient_id}/{sentence_id}")
        seen.add(key)

        candidate = _one_line(row["sentence"])
        reference = _one_line(row["reference_text"])
        if not patient_id or not candidate or not reference:
            raise ValueError(f"blank required value at CSV row {index}")

        patients.add(patient_id)
        label_counts[str(y_true)] += 1
        candidates.append(candidate)
        references.append(reference)
        labels.append(
            {
                "stable_id": f"{patient_id}-s{sentence_id:04d}",
                "patient_id": patient_id,
                "sentence_id": sentence_id,
                "y_true": y_true,
            }
        )

    candidates_path = output_dir / "candidates.txt"
    references_path = output_dir / "references.txt"
    labels_path = output_dir / "labels.jsonl"
    candidates_path.write_text("".join(f"{value}\n" for value in candidates), encoding="utf-8")
    references_path.write_text("".join(f"{value}\n" for value in references), encoding="utf-8")
    labels_path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in labels),
        encoding="utf-8",
    )

    receipt = {
        "schema_version": 1,
        "source_csv": str(source_csv),
        "source_sha256": _sha256(source_csv),
        "row_count": len(rows),
        "patient_count": len(patients),
        "label_counts": dict(sorted(label_counts.items())),
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (candidates_path, references_path, labels_path)
        },
    }
    (output_dir / "input_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    receipt = prepare_inputs(args.source_csv, args.output_dir)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
