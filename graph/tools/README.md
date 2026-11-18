# Offline graph tools

These tools read caller-supplied files. Run them from the repository root. Their outputs can contain source text, generated graph content, or caller labels. Save outputs in a private directory.

## Prepare one patient corpus

Use CSV inputs with a decimal `subject_id` column and at least one evidence field. A `text` field is preserved verbatim when supplied. Other fields are serialized as labeled values when `text` is absent. Sources containing target or label fields are rejected.

```bash
python -m graph.tools.prepare_ehr --csv source-a.csv source-b.csv --patient 42 --out corpus.json
```

For nested JSONL input, patient identity comes from `universal.subject_id`, then `patient_id`, `id`, or `name`. The default `discharge-radiology` mode uses `discharge.text` and `radiology.text`. Use `--mode discharge` for discharge text alone or `--mode structured` for triage, ED stay, diagnosis, and radiology fields. `discharge_target` stays outside the graph corpus in every mode.

```bash
python -m graph.tools.prepare_ehr --jsonl records.jsonl --mode structured --patient 42 --out corpus.json
```

Pass `corpus.json` to the [2.7 or 3.3 graph builder](../README.md).

## Inspect the native Parquet tables

The graph and community commands read a native GraphRAG `output` directory and export HTML plus JSON. The entity/relationship view reads `entities.parquet` and `relationships.parquet`. Place a trusted local D3 v7 distribution beside its output HTML as `d3.v7.min.js`. For example, install `d3@7.9.0` outside the private output directory and copy `dist/d3.min.js` there under that name. The community view reads `communities.parquet` and `community_reports.parquet` and needs no D3 file.

```bash
python -m graph.tools.generate_graph_viz --data-dir runs/domain/output --out private-views/SYNTHETIC-P1/graph_visualization.html
python -m graph.tools.generate_community_viz --data-dir runs/domain/output --out private-views/SYNTHETIC-P1/community_visualization.html
python -m graph.tools.generate_index --visual-dir private-views
```

Repeat the first two commands for each patient directory as needed. Open `private-views/index.html` to navigate the generated views. The exported HTML loads no remote script.

## Score reviewed atomic facts

The evaluator takes already reviewed facts with explicit patient, event, subject, predicate, value, unit, polarity, time, and source anchor. Provide a complete reference with declared patients and documents, reference facts, and separate description and community obligations. The candidate file contains projected facts, description facts, and community facts. See the schema in [atomic_facts.py](atomic_facts.py).

```bash
python -m graph.tools.atomic_facts --reference reviewed-reference.json --candidate reviewed-candidate.json --out graph-score.json
```

The report includes patient-macro scores, micro fact F1, and a digest of the normalized reference. Inputs lacking the required reviewed evidence return an ineligible report and a nonzero exit status. Fact extraction and source review take place before this arithmetic.
