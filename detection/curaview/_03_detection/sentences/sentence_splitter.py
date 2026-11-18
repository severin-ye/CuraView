"""句子分割工具 - 使用共享模块确保一致性"""

import sys
import os
from typing import List, Tuple
from langchain_core.tools import tool

# 导入共享的句子切分函数（使用新的模块化结构）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from curaview._03_detection.sentences.shared_splitter import split_into_sentences


@tool
def split_sentences(text: str) -> List[Tuple[int, str]]:
    """
    将文本分割为句子列表,返回(索引, 句子)元组

    **重要**: 此函数使用共享模块 (langchain/shared_utils/sentence_splitter.py)
    确保与幻觉生成系统使用完全相同的切分逻辑。

    Args:
        text: 原始文本

    Returns:
        [(0, "句子1"), (1, "句子2"), ...]
    """
    # 使用共享函数进行切分
    sentences = split_into_sentences(text)

    # 返回带索引的元组列表
    return [(i, s) for i, s in enumerate(sentences)]
