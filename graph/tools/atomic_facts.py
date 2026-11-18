"""Compare final graph facts with a fixed, complete source reference.

This module does not extract or invent gold facts. Summary facts must be an
audited structured projection of actual description/community contents; passing
IDs or a model's self-reported retention is insufficient. Text-to-fact extraction
correctness remains an upstream review obligation. Reference obligations are
fixed before candidate inspection and must be nonempty for every patient.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json

ATOMIC_FACT_VERSION = "source-atomic-facts-v1"
FIELDS = ("patient", "event", "subject", "predicate", "value", "unit", "polarity", "time")
_UNKNOWN = {"unknown", "n/a", "na", "?", "未确定", "未知"}


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    text = " ".join(value.split())
    if text.casefold() in _UNKNOWN:
        raise ValueError(f"{label} is unknown")
    if label not in ('patient', 'document patient', 'document_id', 'event', 'time', 'unit', 'source anchor'):
        text = text.casefold()
    return text


def _rows(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an explicit list")
    return value


def _fact(raw, documents, patients, aliases):
    if not isinstance(raw, Mapping):
        raise ValueError("fact must be a mapping")
    fact = dict(raw)
    if "value" not in fact and "object" in fact:
        fact["value"] = fact["object"]
    if "object" in fact and "value" in fact and fact["object"] != fact["value"]:
        raise ValueError("object/value conflict")
    normalized = {}
    for field in FIELDS:
        if field not in fact:
            raise ValueError(f"fact missing {field}")
        value = fact[field]
        # Explicit None means unitless / no time stated, never an inferred value.
        if value is None and field in ("unit", "time"):
            normalized[field] = None
        else:
            value = _text(value, field)
            normalized[field] = aliases.get(field, {}).get(value, value)
    if normalized["patient"] not in patients:
        raise ValueError("fact belongs to an undeclared patient")
    if normalized["polarity"] not in ("positive", "negative"):
        raise ValueError("polarity must be positive or negative")
    anchor = fact.get("source_anchor")
    if not isinstance(anchor, Mapping):
        raise ValueError("source_anchor must contain document_id and anchor")
    document = _text(anchor.get("document_id"), "document_id")
    if (normalized["patient"], document) not in documents:
        raise ValueError("source document is absent or belongs to another patient")
    location = _text(anchor.get("anchor"), "source anchor")
    return tuple(normalized[field] for field in FIELDS), (document, location)


def _facts(rows, documents, patients, aliases):
    # Same-event duplicates do not multiply either matches or denominators.
    result = {}
    for raw in _rows(rows, "facts"):
        key, anchor = _fact(raw, documents, patients, aliases)
        result.setdefault(key, set()).add(anchor)
    return result


def _overlap(reference, candidate, indices):
    # Counter intersection gives a one-to-one matching for each projection.
    project = lambda facts: Counter(tuple(key[i] for i in indices) for key in facts)
    return sum((project(reference) & project(candidate)).values())


def _f1(tp, predicted, expected):
    return 2 * tp / (predicted + expected) if predicted + expected else 0.0


def evaluate_atomic_facts(reference: Mapping, candidate: Mapping, *, aliases: Mapping | None = None) -> dict:
    """Return six patient-macro graph metrics, micro F1 and eligibility.

    Reference schema: complete=True, patients=[...], documents=[{patient,
    document_id}], facts=[...], description_obligations=[facts],
    community_obligations=[facts]. Candidate schema: facts=[...],
    description_facts=[...], community_facts=[...]. Every fact contains FIELDS
    (object may replace value), plus source_anchor={document_id, anchor}.
    All fields use strings except explicit unit/time=None. Alias dictionaries
    are field-specific, explicit equivalences; patient/event/time/anchor identity
    cannot be aliased. No inferred units, dates, ontology or LLM matching.

    Invalid/incomplete inputs return metrics=None, eligible=False; they must not
    become successful zero-denominator or N/A measurements.
    """
    result = dict(version=ATOMIC_FACT_VERSION, eligible=False, metrics=None, errors=[], per_patient={}, micro=None,
                  reference_digest=None)
    try:
        if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
            raise ValueError("reference and candidate must be mappings")
        if reference.get("complete") is not True:
            raise ValueError("source reference is not explicitly complete")
        aliases = {} if aliases is None else aliases
        if not isinstance(aliases, Mapping):
            raise ValueError("aliases must be a field-specific mapping")
        normalized_aliases = {}
        for field, mapping in aliases.items():
            if field not in ("subject", "predicate", "value", "unit") or not isinstance(mapping, Mapping):
                raise ValueError("aliases may only map subject/predicate/value/unit")
            normalized_aliases[field] = {_text(k, field): _text(v, field) for k, v in mapping.items()}
            # Require direct, idempotent equivalences; no order-dependent chains.
            if any(normalized_aliases[field].get(v, v) != v for v in normalized_aliases[field].values()):
                raise ValueError("alias chains are not permitted")
        patient_rows = _rows(reference.get("patients"), "patients")
        patients = {_text(p, "patient") for p in patient_rows}
        if not patients or len(patients) != len(patient_rows):
            raise ValueError("patients must be nonempty and unique")
        documents = set()
        for row in _rows(reference.get("documents"), "documents"):
            if not isinstance(row, Mapping):
                raise ValueError("document must be a mapping")
            patient = _text(row.get("patient"), "document patient")
            if patient not in patients:
                raise ValueError("document belongs to undeclared patient")
            documents.add((patient, _text(row.get("document_id"), "document_id")))
        if any(not any(p == patient for p, _ in documents) for patient in patients):
            raise ValueError("every patient requires a source document")
        groups = {}
        for name, rows in (("reference", reference.get("facts")), ("candidate", candidate.get("facts")),
                           ("description_obligations", reference.get("description_obligations")),
                           ("community_obligations", reference.get("community_obligations")),
                           ("description_facts", candidate.get("description_facts")),
                           ("community_facts", candidate.get("community_facts"))):
            groups[name] = _facts(rows, documents, patients, normalized_aliases)
        for name in ("description_obligations", "community_obligations"):
            for fact, anchors in groups[name].items():
                if fact not in groups["reference"] or not anchors <= groups["reference"][fact]:
                    raise ValueError("summary obligations must be anchored source reference facts")
        # Bind arithmetic to the exact normalized reference, obligations and alias
        # dictionary; callers freeze/compare this digest before candidate trials.
        frozen = {name: sorted((json.dumps([key, sorted(anchors)], ensure_ascii=False) for key, anchors in groups[name].items()))
                  for name in ("reference", "description_obligations", "community_obligations")}
        frozen.update(patients=sorted(patients), documents=sorted(documents), aliases=normalized_aliases)
        result["reference_digest"] = hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        totals = Counter()
        for patient in sorted(patients):
            local = {name: {key: anchors for key, anchors in facts.items() if key[0] == patient}
                     for name, facts in groups.items()}
            ref, cand = local["reference"], local["candidate"]
            if not ref or not local["description_obligations"] or not local["community_obligations"]:
                raise ValueError("each patient needs nonempty source facts and both fixed summary obligations")
            exact = len(ref.keys() & cand.keys())
            count = len(ref)
            retention = {}
            for kind in ("description", "community"):
                obligations, observed = local[f"{kind}_obligations"], local[f"{kind}_facts"]
                retained = sum(key in observed and anchors <= observed[key] for key, anchors in obligations.items())
                retention[f"{kind}_retention"] = retained / len(obligations)
            scores = dict(fact_f1=_f1(exact, len(cand), count),
                          event_binding=_overlap(ref, cand, range(6)) / count,
                          time_correctness=_overlap(ref, cand, (0, 1, 2, 3, 4, 5, 7)) / count,
                          negation_preservation=_overlap(ref, cand, range(7)) / count,
                          source_anchor_completeness=sum(key in cand and anchors <= cand[key] for key, anchors in ref.items()) / count,
                          summary_retention=min(retention.values()))
            result["per_patient"][patient] = dict(metrics=scores, **retention, true_positive=exact,
                                                   predicted=len(cand), expected=count)
            totals.update(true_positive=exact, predicted=len(cand), expected=count)
        result["metrics"] = {name: sum(row["metrics"][name] for row in result["per_patient"].values()) / len(patients)
                             for name in next(iter(result["per_patient"].values()))["metrics"]}
        result["micro"] = dict(totals, fact_f1=_f1(totals["true_positive"], totals["predicted"], totals["expected"]))
        result["eligible"] = True
    except (ValueError, TypeError) as exc:
        result.update(metrics=None, per_patient={}, micro=None)
        result["errors"].append(str(exc))
    return result



def main():
    import argparse
    from pathlib import Path

    from graph.io import read_json, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate_atomic_facts(
        json.loads(read_json(args.reference)), json.loads(read_json(args.candidate)),
        aliases=json.loads(read_json(args.aliases)) if args.aliases else None,
    )
    write_json(args.out, report)
    return 0 if report["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
