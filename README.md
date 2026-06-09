# CuraView

**A GraphRAG-Based Framework for Medical AI Hallucination Generation and Detection**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![GraphRAG](https://img.shields.io/badge/GraphRAG-Microsoft-orange.svg)](https://github.com/microsoft/graphrag)
[![LangChain](https://img.shields.io/badge/LangChain-1.0-purple.svg)](https://www.langchain.com/)

---

## Overview

**CuraView** is an innovative research platform for medical AI hallucination detection and correction, dedicated to addressing the reliability and safety issues of medical large language model outputs. By building a complete **Hallucination Generation -> Knowledge Graph Construction -> Intelligent Detection -> System Evaluation** pipeline, it provides technical safeguards for the clinical application of medical AI.

### Key Features

- **Hallucination Generation Agent**: Intelligent hallucination rewriting system based on LangChain 1.0, generating 7 types of medical error data
- **GraphRAG Knowledge Graph**: Medical knowledge graph built on Microsoft GraphRAG, supporting entity relationship extraction and vector retrieval
- **Hallucination Detection Agent**: Context-enhanced detection system combining GraphRAG, supporting both local model and API dual modes
- **System Integration & Evaluation**: Complete comparison and verification tools, supporting multi-model performance evaluation and recall analysis
- **EHR Data Processing**: Efficient electronic health record data processing toolchain, supporting MIMIC-IV dataset
- **Model Fine-tuning Framework**: Medical domain model adaptation training integrated with MS-SWIFT

---

## Code Status

**Code is being organized for public release.** Please check back later or watch this repository for updates.

---

## System Pipeline

The pipeline consists of five core stages:

1. **Evidence preparation**: Patient records are prepared from heterogeneous EHR sources
2. **Knowledge construction**: GraphRAG builds a patient-specific knowledge graph from multi-table EHR evidence
3. **Hallucination generation**: The generation agent injects controlled medical errors according to a seven-type clinical taxonomy
4. **Evidence-grounded detection**: The detection agent verifies each sentence against retrieved graph evidence and assigns one of four evidence grades (E1--E4)
5. **Data curation**: Quality-filtered supervision for improving local verification models

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

## Evidence Grading Scheme

CuraView uses a four-level evidence grading scheme for sentence-level verification:

| Grade | Meaning | Description |
|-------|---------|-------------|
| **E1** | Strong Support | Directly supported by patient EHR evidence |
| **E2** | Weak Support | Partially supported, some ambiguity |
| **E3** | No Support | No supporting evidence found |
| **E4** | Direct Contradiction | Directly contradicted by patient EHR evidence |

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
