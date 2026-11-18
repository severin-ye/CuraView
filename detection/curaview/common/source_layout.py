"""Resolve public detector files relative to this component's source tree."""
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[2]


def retained_path(relative):
    return project_root() / relative
