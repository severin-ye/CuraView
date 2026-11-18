"""Run the saved sentence-level scoring pipeline on four invented records."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        records = [
            ("patient_1", 1, "Blue.", "Blue.", 0, 0),
            ("patient_1", 2, "Red.", "Blue.", 1, 1),
            ("patient_2", 1, "Gold.", "Green.", 1, 0),
            ("patient_2", 2, "Green.", "Green.", 0, 0),
        ]
        manifest = work / "invented.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["patient_id", "sentence_id", "sentence", "reference_text", "y_true", "y_pred"])
            writer.writerows(records)
        subprocess.run([sys.executable, str(here / "official_baseline_inputs.py"),
                        "--source-csv", str(manifest), "--output-dir", str(work / "prepared")],
                       check=True, stdout=subprocess.DEVNULL)
        report = work / "metrics.json"
        subprocess.run([sys.executable, str(here / "formal_metrics.py"),
                        "--manifest", str(manifest), "--predictions", str(manifest),
                        "--out", str(report), "--expected-rows", "4",
                        "--expected-patient-start", "1", "--expected-patient-end", "2",
                        "--bootstrap-replicates", "50"], check=True, stdout=subprocess.DEVNULL)
        metric = json.loads(report.read_text(encoding="utf-8"))["metrics"]["sentence_micro"]
        if (metric["precision"] != 1 or metric["recall"] != 0.5
                or abs(metric["f1"] - 2/3) > 1e-9):
            raise AssertionError(metric)
        print("Four synthetic sentences, two groups: precision 1, recall 0.5, F1 2/3.")


if __name__ == "__main__":
    main()
