"""
共享句子切分工具 - 统一两个agent的句子切分逻辑（升级为基于模型的版本）

这个模块提供了一个标准化的句子切分函数，确保：
1. hallucination_generation_medical_agent (幻觉生成系统)
2. hallucination_detection_graphrag_agent (幻觉检测系统)

使用完全相同的句子切分逻辑，避免索引不一致的问题。

Version 2.0: 集成 HuggingFace sentence_boundary_detection_multilang 模型
- 支持规则模式（兼容旧版本）
- 支持模型模式（更智能的句子边界检测）
- 支持混合模式（规则+模型，工程级推荐）
"""

import re
import os
from typing import List, Optional
import warnings

# 延迟导入，避免不必要的依赖
_model = None
_model_loaded = False

# 配置
MODEL_NAME = "sat-3l-sm"  # 使用 wtpsplit 的 SaT 模型（轻量级版本）
DEFAULT_MODE = "rule"  # 默认使用规则模式，保持向后兼容


def ensure_model_loaded():
    """Compatibility entry used by detect.py; rule mode needs no model load."""
    if os.environ.get('SENTENCE_SPLITTER_MODE', DEFAULT_MODE).lower() == 'rule':
        return False
    return _load_model() is not None


def _load_model():
    """懒加载模型（仅在首次使用时加载）"""
    global _model, _model_loaded

    if _model_loaded:
        return _model

    try:
        from wtpsplit import SaT

        print("正在加载句子边界检测模型 (SaT - Segment any Text)...")
        # 直接使用模型名称，不指定hub_prefix（会自动处理）
        _model = SaT(MODEL_NAME)
        _model_loaded = True
        print("✓ 模型加载成功")

        return _model
    except Exception as e:
        raise RuntimeError("model/hybrid sentence splitting requires a working wtpsplit SaT model") from e


def _model_sentence_split_chunk(text: str) -> List[str]:
    """
    使用模型对文本进行句子分割

    Args:
        text: 文本

    Returns:
        句子列表
    """
    model = _load_model()

    if model is None:
        # 模型加载失败，回退到规则模式
        return _rule_based_split(text)

    try:
        # 使用 SaT 模型进行分割（不需要指定语言）
        sentences = model.split(text)

        # 过滤过短的句子（少于10个字符）
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]

        return sentences

    except Exception as e:
        raise RuntimeError("Sentence splitting model inference failed") from e


def _rule_based_split(text: str) -> List[str]:
    """
    基于规则的句子分割（原有逻辑）

    Args:
        text: 完整文本

    Returns:
        句子列表
    """
    # 先统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 按句号、问号、感叹号分割（支持中英文标点）
    # 保留段落边界作为分句点
    sentences = re.split(r'(?<=[.!?。！？])\s+|\n{2,}', text)

    # 过滤空句子并清理，排除过短的句子（少于10个字符）
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]

    return sentences


def _sliding_window_split(text: str, mode: str = "model") -> List[str]:
    """
    处理文本（SaT模型可以直接处理长文本）

    Args:
        text: 文本
        mode: 分割模式

    Returns:
        句子列表
    """
    if mode == "model":
        return _model_sentence_split_chunk(text)
    else:
        return _rule_based_split(text)


def _hybrid_split(text: str) -> List[str]:
    """
    混合模式：规则生成候选 + 模型精细判断（工程级推荐）

    策略：
    1. 先用规则快速分割出候选句子
    2. 对长句子（>100字符）或复杂句子使用模型进一步判断
    3. 短句子直接保留（规则足够准确）

    Args:
        text: 完整文本

    Returns:
        句子列表
    """
    # 第一步：规则分割
    rule_sentences = _rule_based_split(text)

    final_sentences = []

    for sent in rule_sentences:
        # 短句或简单句：直接保留
        if len(sent) <= 100 or sent.count('.') <= 1:
            final_sentences.append(sent)
        else:
            # 长句或复杂句：使用模型进一步分割
            model_sents = _model_sentence_split_chunk(sent)
            final_sentences.extend(model_sents)

    return final_sentences


def split_into_sentences(text: str, mode: Optional[str] = None) -> List[str]:
    """
    将文本按句子分割（统一接口，支持多种模式）

    模式说明：
    - "rule": 纯规则模式（快速，适合简单文本，默认）
    - "model": 纯模型模式（智能，适合复杂文本）
    - "hybrid": 混合模式（推荐，兼顾速度和准确性）

    Args:
        text: 完整文本
        mode: 分割模式，可选 "rule"/"model"/"hybrid"
              如果为None，使用环境变量 SENTENCE_SPLITTER_MODE 或默认值

    Returns:
        句子列表（按原文顺序）

    Example:
        >>> text = "Hello world.\\n\\nThis is a test."
        >>> split_into_sentences(text, mode="rule")
        ['Hello world.', 'This is a test.']

        >>> split_into_sentences(text, mode="model")
        ['Hello world.', 'This is a test.']
    """
    if not text or not text.strip():
        return []

    # 确定使用的模式
    if mode is None:
        mode = os.environ.get('SENTENCE_SPLITTER_MODE', DEFAULT_MODE)

    mode = mode.lower()

    if mode == "rule":
        return _rule_based_split(text)
    elif mode == "model":
        return _sliding_window_split(text, mode="model")
    elif mode == "hybrid":
        return _hybrid_split(text)
    else:
        warnings.warn(f"未知模式 '{mode}'，使用默认规则模式")
        return _rule_based_split(text)
