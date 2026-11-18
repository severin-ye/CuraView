#!/usr/bin/env python3
"""Compute deterministic metrics and patient-cluster bootstrap from supplied formal pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("rows", "records", "items", "审核项列表"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"No row list found in {path}")


def as_binary(value: Any) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "是", "hallucination", "positive"}:
        return 1
    if normalized in {"0", "false", "no", "否", "faithful", "negative"}:
        return 0
    raise ValueError(f"Unsupported binary label: {value!r}")


def join_pairs(manifest: list[dict[str, Any]], predictions: list[dict[str, Any]], patient_col: str, id_col: str, truth_col: str, pred_col: str) -> list[dict[str, Any]]:
    def keyed(rows: list[dict[str, Any]], required: tuple[str, ...]) -> dict[tuple[str, str], dict[str, Any]]:
        output: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"Missing columns {missing}")
            key = (str(row[patient_col]), str(row[id_col]))
            if key in output:
                raise ValueError(f"Duplicate key {key}")
            output[key] = row
        return output

    truths = keyed(manifest, (patient_col, id_col, truth_col))
    preds = keyed(predictions, (patient_col, id_col, pred_col))
    if set(truths) != set(preds):
        raise ValueError(f"Manifest/prediction keys differ: manifest_only={len(set(truths)-set(preds))}, prediction_only={len(set(preds)-set(truths))}")
    return [{"patient_id": key[0], "sentence_id": key[1], "y_true": as_binary(truths[key][truth_col]), "y_pred": as_binary(preds[key][pred_col])} for key in sorted(truths)]


METRIC_NAMES = ("precision", "recall", "f1", "accuracy")


def sentence_micro_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool sentences before scoring; undefined ratios follow the legacy evaluator's 0 rule."""
    counts = Counter((row["y_true"], row["y_pred"]) for row in rows)
    tn, fp, fn, tp = counts[(0, 0)], counts[(0, 1)], counts[(1, 0)], counts[(1, 1)]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"n_sentences": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "accuracy": (tp + tn) / len(rows) if rows else 0.0}


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Backward-compatible function name for sentence-level micro metrics."""
    return sentence_micro_metrics(rows)


def _mean_patient_metrics(patient_rows: list[list[dict[str, Any]]]) -> dict[str, Any]:
    if not patient_rows:
        raise ValueError("At least one patient cluster is required")
    per_patient = [sentence_micro_metrics(cluster) for cluster in patient_rows]
    return {
        "n_patients": len(per_patient),
        **{name: sum(result[name] for result in per_patient) / len(per_patient) for name in METRIC_NAMES},
    }


def patient_macro_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score each patient independently, then take the unweighted patient mean."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["patient_id"])].append(row)
    return _mean_patient_metrics([grouped[patient] for patient in sorted(grouped)])


def sampled_cluster_metrics(
    grouped: dict[str, list[dict[str, Any]]], sampled_patients: list[str]
) -> dict[str, dict[str, Any]]:
    """Score one cluster sample, preserving every sampled patient occurrence."""
    sampled_clusters = [grouped[patient] for patient in sampled_patients]
    sampled_rows = [row for cluster in sampled_clusters for row in cluster]
    return {
        "sentence_micro": sentence_micro_metrics(sampled_rows),
        "patient_macro": _mean_patient_metrics(sampled_clusters),
    }


def percentile(values: list[float], probability: float) -> float:
    """Linear-interpolated percentile, equivalent to the common type-7 definition."""
    if not values:
        raise ValueError("Cannot compute a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def patient_cluster_bootstrap(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["patient_id"]].append(row)
    patients = sorted(grouped)
    if len(patients) < 2:
        raise ValueError("At least two patient clusters are required")
    rng = random.Random(seed)
    samples = {
        aggregation: {name: [] for name in METRIC_NAMES}
        for aggregation in ("sentence_micro", "patient_macro")
    }
    for _ in range(replicates):
        sampled_patients = rng.choices(patients, k=len(patients))
        result = sampled_cluster_metrics(grouped, sampled_patients)
        for aggregation in samples:
            for name in METRIC_NAMES:
                samples[aggregation][name].append(result[aggregation][name])
    intervals = {
        aggregation: {
            name: [percentile(values, 0.025), percentile(values, 0.975)]
            for name, values in aggregation_samples.items()
        }
        for aggregation, aggregation_samples in samples.items()
    }
    return {
        "cluster_unit": "patient_id",
        "sampling": "patients sampled with replacement; repeated draws retain multiplicity",
        "patient_clusters": len(patients),
        "replicates": replicates,
        "seed": seed,
        "percentile_95_ci": intervals,
        "valid_replicates": {
            aggregation: {name: len(values) for name, values in aggregation_samples.items()}
            for aggregation, aggregation_samples in samples.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--id-col", default="sentence_id")
    parser.add_argument("--truth-col", default="y_true")
    parser.add_argument("--pred-col", default="y_pred")
    parser.add_argument("--expected-rows", type=int, default=1103)
    parser.add_argument("--expected-patient-start", type=int, default=201)
    parser.add_argument("--expected-patient-end", type=int, default=250)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise SystemExit("PREFLIGHT_BLOCKED: bootstrap replicates must be positive")

    output_path = Path(args.out)
    if output_path.exists():
        raise SystemExit(f"PREFLIGHT_BLOCKED: output already exists; refusing overwrite: {output_path}")

    manifest = load_rows(Path(args.manifest))
    predictions = load_rows(Path(args.predictions))
    pairs = join_pairs(manifest, predictions, args.patient_col, args.id_col, args.truth_col, args.pred_col)
    if len(pairs) != args.expected_rows:
        raise SystemExit(f"PREFLIGHT_BLOCKED: expected {args.expected_rows} paired rows, found {len(pairs)}")
    for row in pairs:
        row['patient_id'] = row['patient_id'].removeprefix('patient_')
    expected_patients = {str(i) for i in range(args.expected_patient_start, args.expected_patient_end + 1)}
    observed = {row["patient_id"] for row in pairs}
    if observed != expected_patients:
        raise SystemExit(f"PREFLIGHT_BLOCKED: patient coverage does not match {args.expected_patient_start}-{args.expected_patient_end}")
    normalized_keys = {(row['patient_id'], row['sentence_id']) for row in pairs}
    if len(normalized_keys) != len(pairs):
        raise SystemExit('PREFLIGHT_BLOCKED: duplicate normalized patient/sentence keys')
    sources = {name: {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for name, path in [('manifest', args.manifest), ('predictions', args.predictions)]}
    report = {
        "sources": sources,
        "preflight": {"paired_rows": len(pairs), "patients": len(observed)},
        "metrics": {
            "sentence_micro": sentence_micro_metrics(pairs),
            "patient_macro": patient_macro_metrics(pairs),
        },
        "patient_cluster_bootstrap": patient_cluster_bootstrap(pairs, args.bootstrap_replicates, args.seed),
    }
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as exc:
        raise SystemExit(f"PREFLIGHT_BLOCKED: output already exists; refusing overwrite: {output_path}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
