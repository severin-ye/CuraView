# CuraView

**A GraphRAG-Based Framework for Medical AI Hallucination Generation and Detection**

Official research code release for the paper:

> *"CuraView: A GraphRAG-Based Framework for Medical AI Hallucination Generation and Detection"*

[![Paper](https://img.shields.io/badge/Paper-coming%20soon-blue)](#citation)
[![Code](https://img.shields.io/badge/Code-coming%20soon-orange)](#code-status)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![GraphRAG](https://img.shields.io/badge/GraphRAG-Microsoft-orange)](https://github.com/microsoft/graphrag)
[![LangChain](https://img.shields.io/badge/LangChain-1.0-purple)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

---

## Overview

CuraView is a knowledge-based multi-agent framework for sentence-level medical hallucination detection in discharge summaries. It formulates hallucination detection as patient-grounded claim verification using GraphRAG-enhanced evidence retrieval and structured evidence grading (E1--E4).

### Key Features

- **Hallucination Generation Agent**: Controlled injection of 7 clinically meaningful hallucination types via LangChain
- **GraphRAG Knowledge Graph**: Patient-specific knowledge graphs built from heterogeneous EHR sources using Microsoft GraphRAG
- **Hallucination Detection Agent**: Evidence-grounded sentence-level verification with dual API/local model support
- **Structured Evidence Grading**: Four-level scheme (E1 strong support, E2 weak support, E3 no support, E4 direct contradiction)
- **Multi-Agent Curation Pipeline**: Generation--detection--curation workflow for producing quality-filtered training data
- **EHR Data Processing**: Toolchain for MIMIC-IV discharge-me dataset processing

---

## Code Status

**Code is being organized for public release.** Please check back later or watch this repository for updates.

---

---

## Hallucination Types

CuraView covers seven clinically meaningful hallucination categories:

| Type | Description | Example |
|------|-------------|---------|
| `diagnosis_error` | Diagnosis errors | Diabetes changed to hypertension |
| `medication_error` | Medication errors | Aspirin changed to penicillin |
| `exam_result_error` | Examination/lab result errors | Negative changed to positive |
| `time_error` | Temporal errors | 3 days changed to 7 days |
| `value_error` | Numerical value errors | 120/80 changed to 180/120 |
| `negation_error` | Negation/polarity errors | "No" changed to "Yes" |
| `invented_fact` | Completely fabricated events | Fabricated medical procedures |

---

---

---

## Evidence Grading Scheme

CuraView uses a four-level evidence grading scheme for sentence-level verification:

| Grade | Meaning | Description |
|-------|---------|-------------|
| **E1** | Strong Support | Directly supported by patient EHR evidence |
| **E2** | Weak Support | Partially supported, some ambiguity |
| **E3** | No Support | No supporting evidence found |
| **E4** | Direct Contradiction | Directly contradicted by patient EHR evidence |

---

---

## Related Repository

- **Dataset**: [CuraView-EVD](https://github.com/severin-ye/CuraView-EVD) - Evidence-annotated clinical hallucination dataset (coming soon)

---

## Results

Experimental results are not included in this repository. Results depend on local data, model weights or API versions, prompt revisions, and GraphRAG configuration. A future reproduction release may provide exact table and figure scripts.

---

## Citation

If you use CuraView, cite the repository and specify the exact commit hash used in your experiments.

```bibtex
@misc{curaview2026,
  title  = {CuraView: A GraphRAG-Based Framework for Medical AI Hallucination Generation and Detection},
  author = {Ye, Severin and Kong, Xiao and He, Xiaopeng and Yan, Guangsu and Peng, Limei and Oh, Dongsuk},
  year   = {2026},
  note   = {Official research code release},
  url    = {https://github.com/severin-ye/CuraView}
}
```

---

## License

- **Code**: MIT License, unless otherwise noted.
- **Clinical data**: Not redistributed; governed by the original data provider and data-use agreement.
- **Model weights and checkpoints**: Not redistributed; governed by the original model licenses.

---

## Contact

- **Repository maintainer**: Severin Ye
- **GitHub**: [@severin-ye](https://github.com/severin-ye)
- **Email**: [6severin9@gmail.com](mailto:6severin9@gmail.com)

For questions about the research code, open a GitHub issue or contact the repository maintainer.
