"""Prepare a patient-scoped GraphRAG 2.7 project for the local detector."""
import argparse
from pathlib import Path
import re
import shutil

from .io import load_corpus, write_json


def prepare(corpus_path, project):
    corpus = load_corpus(corpus_path)
    patient = corpus["patient_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", patient):
        raise ValueError("patient_id must be a safe directory name using letters, digits, underscores or hyphens")
    root = Path(project).resolve() / "output" / "index_output" / patient
    root.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).parent / "v27"
    shutil.copyfile(template / "settings.yaml", root / "settings.yaml")
    shutil.copytree(template / "prompts", root / "prompts")
    (root / "input").mkdir()
    write_json(root / "input" / "documents.json", corpus["documents"])
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    print(prepare(args.corpus, args.project))


if __name__ == "__main__":
    main()
