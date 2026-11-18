# Scoring inference outputs

Run `python evaluation/synthetic_check.py` from the repository root with Python 3.12. The four invented sentences include two positive references and one positive prediction, which is correct. The check expects precision 1, recall 0.5 and F1 2/3.

[formal_metrics.py](formal_metrics.py) joins caller-supplied reference and prediction files by patient and sentence, then calculates sentence and patient metrics with patient-level resampling. Run `python evaluation/formal_metrics.py --help` for the input and cohort options. [official_baseline_inputs.py](official_baseline_inputs.py) supplies the input-format conversion used by the [example](synthetic_check.py). The Web interface uses the same scoring functions.
