"""Select an isolated runtime config without replacing the shared source file."""
import os
from pathlib import Path
from curaview.common.source_layout import retained_path


def detection_config_path():
    return Path(os.environ.get("CURAVIEW_DETECTION_CONFIG", retained_path("config.yaml")))
