"""Run one predeclared Flat, Generic and Domain comparison on a shared corpus."""
import argparse
import hashlib
import json
from pathlib import Path

from graph.io import read_json, write_json
from graph.native import ARMS as BUILD_ARMS, load_graph, native_context
from graph.provider import Transport, chat_body, parse_completion
from graph.retrieval import _cosine, anchor_query, retrieve

from .detection import citation_diagnostic, detection_messages, quote_proxy, render_detection_case
from .output_contract import GRADES, parse_strict_detection

ARMS = ("flat_dense", "generic_graph", "domain_graph")


def material(corpus, tables):
    units = [{
        "evidence_id": row.id, "source_evidence_id": row.document_id,
        "patient_id": corpus["patient_id"], "text": row.text,
    } for row in tables["text_units"].itertuples()]
    vectors = {row.id: [float(value) for value in row.embedding]
               for row in tables["embeddings.text_unit_text"].itertuples()}
    for vector in vectors.values():
        _cosine(vector, vector)
    return units, vectors


def load_cases(path, sources):
    cases = json.loads(read_json(path))
    if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
        raise ValueError("supply 1..12 cases per run")
    sample_ids = set()
    for case in cases:
        if (not isinstance(case, dict)
                or any(not isinstance(case.get(key), str) or not case[key].strip()
                       for key in ("sample_id", "sentence"))
                or case["sample_id"] in sample_ids):
            raise ValueError("each case needs a unique nonempty sample_id and sentence")
        sample_ids.add(case["sample_id"])
        ref = case.get("reference")
        if ref is None:
            continue
        if not isinstance(ref, dict) or ref.get("source_grade") not in GRADES:
            raise ValueError("optional full-source reference must use source_grade E1 through E4")
        quotes = ref.get("source_quotes", [])
        if not isinstance(quotes, list) or any(
            not isinstance(q, dict) or not isinstance(q.get("evidence_id"), str)
            or q["evidence_id"] not in sources
            or not isinstance(q.get("quote"), str) or not q["quote"].strip()
            or q["quote"] not in sources[q["evidence_id"]]
            for q in quotes
        ):
            raise ValueError("reference quotations must occur verbatim in their source documents")
    return cases


def run_paired(generic, domain, cases_path, strategy_path, out):
    builds = [load_graph(root) for root in (generic, domain)]
    (generic_build, corpus, generic_tables), (domain_build, domain_corpus, domain_tables) = builds
    if (generic_build["arm"], domain_build["arm"]) != ("generic", "domain"):
        raise ValueError("paired run requires Generic then Domain graph")
    if corpus != domain_corpus or generic_build["provider_base_url"] != domain_build["provider_base_url"]:
        raise ValueError("both arms require the same complete source corpus and provider endpoint")
    ceilings = [build["completion_ceiling"] for build in (generic_build, domain_build)]
    if ceilings[0] != ceilings[1] and ceilings != list(BUILD_ARMS.values()):
        raise ValueError("use matched ceilings or the historical 8192/32768 configuration pair")
    units, vectors = material(corpus, generic_tables)
    other_units, other_vectors = material(corpus, domain_tables)
    if units != other_units or vectors != other_vectors:
        raise ValueError("native source units and dense vectors must match exactly")
    sources = {doc["id"]: doc["text"] for doc in corpus["documents"]}
    cases = load_cases(cases_path, sources)
    strategy = read_json(strategy_path)
    if not strategy.strip():
        raise ValueError("fixed strategy file is empty")
    source_evidence = [{"evidence_id": key, "text": value} for key, value in sources.items()]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "inputs.json", {
        "corpus": corpus, "cases": cases, "strategy": strategy,
        "fixed_build_receipts": {arm: hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
                                 for arm, root in zip(("generic", "domain"), (generic, domain))},
        "completion_output_limits": ceilings, "top_k": 4,
        "max_chars": 6000, "max_native_context_tokens": 6000, "planned_calls": 5 * len(cases),
    })
    transport = Transport(out / "requests")
    rows = []
    try:
        if transport.endpoint != generic_build["provider_base_url"]:
            raise ValueError("build and paired query require the same provider endpoint")
        for number, case in enumerate(cases, 1):
            sentence = case["sentence"]
            anchor = anchor_query(sentence) or sentence
            natives = []
            for arm, root, (build, _, tables) in zip(ARMS[1:], (generic, domain), builds):
                native = native_context(root, build, tables, anchor, transport)
                write_json(out / f"case-{number:02d}-{arm}-native.json", native)
                _cosine(native["query_embedding"], next(iter(vectors.values())))
                natives.append(native)
            ranked = {"flat_dense": retrieve(sentence, units, patient_id=corpus["patient_id"],
                       mode="dense", vectors=vectors, query_vector=natives[0]["query_embedding"],
                       top_k=4, max_chars=6000)}
            for arm, native in zip(ARMS[1:], natives):
                ranked[arm] = retrieve(sentence, units, patient_id=corpus["patient_id"],
                                       mode="graph", graph_ranking=native["text_unit_ranking"],
                                       top_k=4, max_chars=6000)
                ranked[arm].update(ranking_origin="observed_native_local_search_pre_budget_source_order",
                                   native_internal_ranking_observed=True)
            for arm, result in ranked.items():
                runtime, mapping = render_detection_case({
                    **case, "source_evidence": source_evidence, "provided_evidence": result["selected"],
                }, citation_aliases=True)
                messages = detection_messages(runtime, strategy)
                raw = transport.request("chat/completions", chat_body(messages, max_tokens=4096, structured=True))
                completion = parse_completion(raw)
                pred = parse_strict_detection(completion.content)
                citation = citation_diagnostic(pred["reasoning"] or "", mapping["citation_to_original"], pred["evidence_grade"])
                pred.update(citation_diagnostic=citation, actual_model=completion.model)
                reference = case.get("reference") or {}
                grade = reference.get("source_grade")
                row = {"sample_id": case["sample_id"], "arm": arm, "retrieval": result,
                       "prediction": pred, "citation_mapping": mapping,
                       "reference_proxy": quote_proxy(result, reference),
                       "full_source_reference_grade": grade,
                       "full_source_grade_agreement": pred["evidence_grade"] == grade
                       if grade and pred["status"] == "valid" else None,
                       "provided_context_grade_adjudicated": False}
                rows.append(row)
                write_json(out / f"case-{number:02d}-{arm}-judgment.json", row)
                if pred["status"] != "valid" or citation["citation_failed"]:
                    raise ValueError("invalid detection or missing/unknown citation identifier")
        summary = {}
        for arm in ARMS:
            arm_rows = [row for row in rows if row["arm"] == arm]
            grades = [row["full_source_grade_agreement"] for row in arm_rows
                      if row["full_source_grade_agreement"] is not None]
            coverage = [row["reference_proxy"]["quoted_fact_coverage"] for row in arm_rows
                        if row["reference_proxy"]["quoted_fact_coverage"] is not None]
            summary[arm] = {
                "planned": len(cases), "valid": len(arm_rows),
                "reference_grade_agreements": sum(grades), "reference_grade_denominator": len(grades),
                "quote_proxy_mean": sum(coverage) / len(coverage) if coverage else None,
                "quote_proxy_denominator": len(coverage),
            }
        write_json(out / "report.json", {
            "status": "completed", "planned_calls": 5 * len(cases), "actual_calls": transport.calls,
            "completion_output_limits": ceilings, "prompt_only_causal_attribution": False,
            "native_graph_answer_generation": "Not Run", "clinical_accuracy": None,
            "summary": summary, "rows": rows,
        })
    except Exception as error:
        write_json(out / "failure.json", {
            "status": "failed", "error_type": type(error).__name__,
            "actual_calls": transport.calls, "retained_rows": len(rows),
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generic", "domain", "cases", "strategy", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run_paired(args.generic, args.domain, args.cases, args.strategy, args.out)


if __name__ == "__main__":
    main()

