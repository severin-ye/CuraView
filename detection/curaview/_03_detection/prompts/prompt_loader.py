"""提示词加载工具"""

import os
from curaview.common.source_layout import retained_path


def load_detection_prompt() -> str:
    """
    加载幻觉检测提示词

    Returns:
        提示词内容
    """
    prompt_file = os.path.join(
        str(retained_path('doc/prompts')),
        'detection_prompt.md'
    )

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()
