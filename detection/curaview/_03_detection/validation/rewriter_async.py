"""
检测结果重检测模块 - 异步版本

当发现幻觉标签与证据等级不一致时，异步并发重新执行检测流程
"""

import os
import sys
import asyncio
import yaml
from typing import Dict, Tuple, List

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from curaview._03_detection.validation.schemas import HallucinationDetectionResult, SentenceDetectionRecord
from curaview._03_detection.validation.validators import DetectionValidator
from curaview.common.logger import setup_logger
from curaview.common.config_loader import detection_config_path

logger = setup_logger()

# 加载配置
config_path = detection_config_path()
with open(config_path, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)


async def redetect_inconsistent_result_async(
    record: SentenceDetectionRecord,
    patient_id: str,
    detect_func,
    llm_structured,
    max_retries: int = 3
) -> Tuple[HallucinationDetectionResult, int]:
    """
    异步重新检测不一致的结果

    Args:
        record: 原始检测记录
        patient_id: 患者ID
        detect_func: detect_single_sentence函数
        llm_structured: 配置了结构化输出的LLM
        max_retries: 最大重试次数

    Returns:
        (修正后的检测结果, 尝试次数)
    """
    original_result = record.detection_result
    sentence = record.original_sentence

    logger.info(f"重新检测句子 {record.sentence_index}: {sentence[:50]}...")

    for attempt in range(1, max_retries + 1):
        try:
            # 在线程池中运行同步的detect_func
            loop = asyncio.get_event_loop()
            detection_result_dict = await loop.run_in_executor(
                None,
                detect_func,
                sentence,
                patient_id,
                llm_structured
            )

            redetected_result = detection_result_dict["detection_result"]

            # 验证修正后的结果
            is_consistent, errors = DetectionValidator.validate_consistency_with_context(
                redetected_result.hallucination_status,
                redetected_result.evidence_grade.value,
                redetected_result.hallucination_type.value if redetected_result.hallucination_type else None,
                redetected_result.reasoning
            )

            if is_consistent:
                logger.info(f"✅ 句子 {record.sentence_index} 重检成功 (尝试 {attempt} 次)")
                return redetected_result, attempt
            else:
                logger.warning(f"⚠️  句子 {record.sentence_index} 重检后仍不一致: {errors}")
                if attempt < max_retries:
                    await asyncio.sleep(1)  # 短暂延迟后重试

        except Exception as e:
            logger.error(f"❌ 句子 {record.sentence_index} 重检失败: {e}")
            if attempt < max_retries:
                await asyncio.sleep(2)
            else:
                raise RuntimeError("Redetection exhausted after execution failures") from e

    # 达到最大重试次数仍不一致，返回最后一次结果
    logger.warning(f"句子 {record.sentence_index} 达到最大重试次数，使用最后结果")
    raise ValueError("Redetection exhausted without a consistent result")


async def redetect_batch_async(
    records_to_redetect: List[Tuple[int, SentenceDetectionRecord]],  # [(position, record), ...]
    patient_id: str,
    detect_func,
    llm_structured,
    max_retries: int = 3
) -> Dict:
    """
    批量异步重检不一致的句子

    Args:
        records_to_redetect: [(record_position, record), ...] 需要重检的记录列表
        patient_id: 患者ID
        detect_func: detect_single_sentence函数
        llm_structured: LLM结构化输出对象
        max_retries: 每个句子的最大重试次数

    Returns:
        {
            'updated_results': {position: new_detection_result, ...},
            'total_attempts': int,
            'successful_count': int
        }
    """
    logger.info(f"开始批量异步重检: {len(records_to_redetect)} 个句子")

    async def process_one(position, record):
        new_result, attempts = await redetect_inconsistent_result_async(
            record=record,
            patient_id=patient_id,
            detect_func=detect_func,
            llm_structured=llm_structured,
            max_retries=max_retries
        )

        # 验证是否成功
        is_consistent, _ = DetectionValidator.validate_consistency_with_context(
            new_result.hallucination_status,
            new_result.evidence_grade.value,
            new_result.hallucination_type.value if new_result.hallucination_type else None,
            new_result.reasoning
        )

        return {
            'position': position,
            'new_result': new_result,
            'attempts': attempts,
            'is_consistent': is_consistent
        }

    # 并发执行所有重检任务
    tasks = [process_one(pos, rec) for pos, rec in records_to_redetect]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 汇总结果
    updated_results = {}
    total_attempts = 0
    successful_count = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"重检任务异常: {result}")
            continue

        position = result['position']
        new_result = result['new_result']
        attempts = result['attempts']
        is_consistent = result['is_consistent']

        updated_results[position] = new_result
        total_attempts += attempts

        if is_consistent:
            successful_count += 1

    logger.info(f"批量重检完成: 成功 {successful_count}/{len(records_to_redetect)}, 总尝试次数: {total_attempts}")

    return {
        'updated_results': updated_results,
        'total_attempts': total_attempts,
        'successful_count': successful_count
    }
