# CuraView

**Code release for the paper "CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification"**

> Code is being organized for public release — please watch this repository for updates.

[![arXiv](https://img.shields.io/badge/arXiv-2605.03476-b31b1b.svg)](https://arxiv.org/abs/2605.03476)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)

---

## Overview

CuraView is a knowledge-based multi-agent framework for sentence-level medical hallucination detection in discharge summaries. It formulates hallucination detection as patient-grounded claim verification rather than generic factuality evaluation, using GraphRAG-enhanced evidence retrieval and structured evidence grading (E1--E4).

### Key Contributions

- **Patient-grounded formulation**: Sentence-level claim verification against patient-specific EHR evidence with E1--E4 evidence grading
- **Seven-type clinical hallucination taxonomy**: Diagnosis, medication, examination/laboratory, temporal, numerical, negation, and fabricated-fact errors
- **GraphRAG-based knowledge representation**: Per-patient knowledge graphs from heterogeneous EHR sources with domain-customized graph construction
- **Multi-agent pipeline**: Generation--detection--curation workflow producing evidence-annotated hallucination data for fine-tuning local verifiers

### System Pipeline

```
Evidence Preparation -> Knowledge Graph Construction -> Hallucination Generation
       -> Evidence-Grounded Detection -> Data Curation -> Curated Fine-Tuning
```

---

## Repository Status

**Code is being organized for public release.** The full source code will include:

- **LangChain-based multi-agent framework** for hallucination generation and detection
- **GraphRAG** indexing and medical prompt templates
- **Meditron-7B** generation and detection wrappers
- **MS-SWIFT** training, merging, quantization, and inference utilities
- **EHR data processing toolchain** for MIMIC-IV/Discharge-Me

---

## Dataset

The CuraView-EVD (Evidence-Annotated Dataset) is available at a separate repository:

- **[CuraView-EVD](https://github.com/severin-ye/CuraView-EVD)** — Evidence-annotated clinical hallucination dataset (coming soon)

---

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

MIT License

---

## Contact

- **Project Lead**: Severin Ye
- **GitHub**: [@severin-ye](https://github.com/severin-ye)
- **Email**: [6severin9@gmail.com](mailto:6severin9@gmail.com)
