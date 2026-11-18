# Sentence-level hallucination detection

This release exports the sentence detector and CLI from the maintained CuraView source dated 2026-11-18. The workflow loads an existing GraphRAG 2.7 index, splits the input, retrieves evidence for each sentence, requests a structured judgment, validates the responses, and writes a report. The report tracks failed and invalid judgments separately and excludes them from the valid detection denominator.

## Set up and run

Use Python 3.12 in a separate environment for GraphRAG 2.7.

```sh
python -m pip install -r detection/requirements.txt
python detection/detect.py --help
python detection/smoke_test.py
```

1. Build a GraphRAG 2.7 index for a document you are authorized to use. See the [graph component](../graph/).
2. Set `graphrag.project_root` in [config.yaml](config.yaml), or set `CURAVIEW_DETECTION_CONFIG` to your own YAML. Paths in that setting are relative to the YAML's directory. The detector expects the GraphRAG workspace for each document at `<project_root>/output/index_output/patient_1/`, with its own settings and GraphRAG's Parquet files inside `output/`. Each ID must be `patient_` followed by an integer.
3. Set `GRAPHRAG_API_KEY` for the provider configured in the default YAML. Set your model and endpoint under `llm.api` and configure the graph workspace's separate model settings as needed.
4. Run the detector with the same ID as the index.

```sh
python detection/detect.py /path/to/input.txt --patient-id patient_1
```

For batch discovery, set `CURAVIEW_DETECTION_TEXT_ROOT` to a directory containing `patient_1.txt`, `patient_2.txt`, and so on, then launch `python detection/detect.py` to select IDs interactively. A headless batch can use `--background --all`, `--background --patient-ids patient_1 patient_2`, or `--background --start 1 --end 2`. The launcher reports the PID and output log.

For local detection, install `detection/requirements-local.txt` with a PyTorch build suitable for your hardware, set `llm.mode: local`, and point `llm.local.model_path` at your local model. GraphRAG still uses the configured graph runtime and may call its provider for retrieval. Sentence splitting defaults to rules. The optional SaT splitter uses `wtpsplit` and its model when explicitly selected through `SENTENCE_SPLITTER_MODE`.

The default report path is `detection/outputs/<model>/detection_results/` and the default trace path is `detection/outputs/traces/`. Set `CURAVIEW_LEGACY_TRACE_ROOT` to direct traces to private storage. Reports, logs, and traces can contain supplied text and model responses. Use your own private runtime directory and provider settings for authorized material.

## Evidence and source scope

| Grade | Interpretation |
|---|---|
| E1 | The evidence directly supports the statement. |
| E2 | The evidence supports the statement through an inference. |
| E3 | The evidence neither supports nor directly contradicts the statement. |
| E4 | The evidence directly contradicts the statement. |

E1 and E2 correspond to `hallucination_status=false` with a null type. E3 and E4 correspond to `hallucination_status=true` with a type. The [paper-aligned prompt](doc/prompts/detection_prompt_paper_aligned.md) and [strict parser](curaview/_03_detection/parsing/output_contract.py) are separately available for callers using the four-field paper-aligned JSON format. The batch CLI uses the structured workflow and the public copies of its detector prompts in [doc/prompts](doc/prompts/). We removed case-like examples while preserving the instructions, label definitions and output fields. These copies have different prompt bytes from the prior source.

The offline smoke test runs sentence splitting, calls the real detector with invented evidence and a substituted model, checks successful trace capture, verifies the failed-query behavior, and checks both positive and inconsistent strict parser replies. A fresh corpus, provider, and local weights are supplied by the reader for model-backed runs.
