"""
共享句子切分工具 - 统一两个agent的句子切分逻辑

这个模块提供了一个标准化的句子切分函数，确保：
1. hallucination_generation_medical_agent (幻觉生成系统)
2. hallucination_detection_graphrag_agent (幻觉检测系统)

使用完全相同的句子切分逻辑，避免索引不一致的问题。
"""

import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    将文本按句子分割（统一版本）

    切分规则：
    1. 按中英文句号、问号、感叹号后的空白字符分割
    2. 按连续2个或更多换行符分割（段落边界）
    3. 过滤空句子和过短句子（<10字符）

    Args:
        text: 完整文本

    Returns:
        句子列表（按原文顺序）

    Example:
        >>> text = "Hello world.\\n\\nThis is a test."
        >>> split_into_sentences(text)
        ['Hello world.', 'This is a test.']
    """
    # 先统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 按句号、问号、感叹号分割（支持中英文标点）
    # 保留段落边界作为分句点
    sentences = re.split(r'(?<=[.!?。！？])\s+|\n{2,}', text)

    # 过滤空句子并清理，排除过短的句子（少于10个字符）
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]

    return sentences
