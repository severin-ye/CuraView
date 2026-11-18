# CuraView inference framework

CuraView checks each sentence in a discharge summary against evidence from the same patient's records. This repository supplies the inference framework for the [CuraView paper](https://arxiv.org/abs/2605.03476). [中文说明](README.zh-CN.md)

[Table-by-table guide](supplement/Table_Guide.md) · [Executed example, field mapping and complete synthetic request](supplement/Executed_Example.md)

## Run the detector

1. Use Python 3.12. Follow the [GraphRAG 2.7 instructions](graph/README.md) to index records you are authorized to use. Use a patient identifier such as `patient_1` for both the index and the detector.
2. Install the [detector dependencies](detection/requirements.txt) and configure the index and model in the [detector settings](detection/config.yaml). Model-backed runs need an authorized endpoint or a compatible local model. Keep patient inputs and runtime traces private.
3. Run `python detection/detect.py /path/to/summary.txt --patient-id patient_1`. The [detector guide](detection/README.md) explains the evidence grades, prompts, output fields and handling of failed judgments.

## Check the implementation

| Paper component | Public implementation and check |
|---|---|
| Patient-specific graph and evidence retrieval | [Index, query and input formats](graph/README.md) |
| Sentence-level judgment | [Prompts](detection/doc/prompts/), [output validation](detection/curaview/_03_detection/parsing/output_contract.py), [offline detector example](detection/smoke_test.py) |
| Flat, Generic and Domain retrieval with a fixed detector | [Three-condition comparison](comparison/README.md) and [GraphRAG 3.3 loopback integration test](graph/test_native.py). Use a separate 3.3 environment for this workflow. |
| Scoring detector predictions | [Scorer and four-sentence example](evaluation/README.md) |
| Inspect local results | [Web interface](webui/README.md) |

Run `python evaluation/synthetic_check.py` for a standard-library check with four invented sentences. With the detector environment installed, run `python detection/smoke_test.py` to check evidence handling and structured output with a simulated model.

The paper describes the verifier's training method and hyperparameters. This release distributes inference code, settings, prompt templates and synthetic checks. Training and curation code, training logs, checkpoints, and study data are outside the release. Reproducing the clinical performance tables requires the corresponding historical inputs and model.

## Citation

```bibtex
@misc{ye2026curaview,
  title         = {CuraView: A Multi-Agent Framework for Medical Hallucination Detection with {GraphRAG}-Enhanced Knowledge Verification},
  author        = {Ye, Severin and Kong, Xiao and He, Xiaopeng and Yan, Guangsu and Oh, Dongsuk},
  year          = {2026},
  eprint        = {2605.03476},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2605.03476}
}
```

---

## License

The project code is distributed under the [MIT License](LICENSE). Separately obtained models, datasets and upstream dependencies retain their own terms.
