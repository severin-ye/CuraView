"""Run a public detector job with an isolated input, report and trace directory."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

DETECTOR = Path(__file__).resolve().with_name("detect.py")


def valid_patient_id(value):
    return isinstance(value, str) and re.fullmatch(r"patient_[0-9]+", value) is not None


def run_detection(text, patient_id, run_dir, *, timeout=600):
    if not valid_patient_id(patient_id):
        raise ValueError("Patient ID must be patient_ followed by an integer")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    input_path = run_dir / "input.txt"
    input_path.write_text(text, encoding="utf-8")
    outputs = run_dir / "reports"
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "CURAVIEW_DETECTION_OUTPUT_ROOT": str(outputs),
        "CURAVIEW_DETECTION_LOG_ROOT": str(run_dir / "logs"),
        "CURAVIEW_LEGACY_TRACE_ROOT": str(run_dir / "trace"),
    }
    if config := env.get("CURAVIEW_DETECTION_CONFIG"):
        env["CURAVIEW_DETECTION_CONFIG"] = str(Path(config).resolve())
    log_path = run_dir / "detector.log"
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, str(DETECTOR), str(input_path), "--patient-id", patient_id],
            cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
            check=False, timeout=timeout,
        )
    candidates = list(outputs.glob(f"*/detection_results/{patient_id}_detection_report.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Detector exited with code {completed.returncode} without one report. See {log_path}")
    report = json.loads(candidates[0].read_text(encoding="utf-8"))
    if report.get("patient_id") != patient_id:
        raise RuntimeError("Detector report patient ID does not match the input")
    state = report.get("processing_status")
    if state not in {"completed", "partial", "failed"}:
        raise RuntimeError("Detector report has no recognized processing status")
    if (completed.returncode == 0) != (state == "completed"):
        raise RuntimeError(f"Detector exit code and report disagree. See {log_path}")
    return report
