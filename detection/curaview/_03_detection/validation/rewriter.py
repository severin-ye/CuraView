"""
检测结果重检测模块

当发现幻觉标签与证据等级不一致时，重新执行完整的幻觉检测流程
"""

import os
import sys
import yaml
from typing import Dict, Tuple

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


def redetect_inconsistent_result(
    record: SentenceDetectionRecord,
    patient_id: str,
    detect_func,
    llm_structured,
    max_retries: int = 3
) -> Tuple[HallucinationDetectionResult, int]:
    """
    重新检测不一致的结果（执行完整检测流程）

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

    logger.info(f"\n{'='*80}")
    logger.info(f"重新检测句子 {record.sentence_index}: {sentence[:100]}...")
    logger.info(f"{'='*80}")
    logger.info(f"原始判断: status={original_result.hallucination_status}, "
                f"type={original_result.hallucination_type}, "
                f"grade={original_result.evidence_grade.value}")

    for attempt in range(1, max_retries + 1):
        logger.info(f"\n尝试 {attempt}/{max_retries}...")

        try:
            # 调用完整的检测流程
            logger.info("重新执行完整幻觉检测流程...")
            detection_result_dict = detect_func(
                sentence=sentence,
                patient_id=patient_id,
                llm_structured=llm_structured
            )

            redetected_result = detection_result_dict["detection_result"]

            logger.info(f"✓ 重新检测完成")
            logger.info(f"  新判断: status={redetected_result.hallucination_status}, "
                       f"type={redetected_result.hallucination_type}, "
                       f"grade={redetected_result.evidence_grade.value}")

            # 验证修正后的结果
            is_consistent, errors = DetectionValidator.validate_consistency_with_context(
                redetected_result.hallucination_status,
                redetected_result.evidence_grade.value,
                redetected_result.hallucination_type.value if redetected_result.hallucination_type else None,
                redetected_result.reasoning
            )

            if is_consistent:
                logger.info(f"✅ 重新检测成功，一致性验证通过")
                return redetected_result, attempt
            else:
                logger.warning(f"⚠️  重新检测后仍不一致: {errors}")
                if attempt < max_retries:
                    logger.info("继续重试...")

        except Exception as e:
            logger.error(f"❌ 重新检测失败: {e}")
            if attempt < max_retries:
                logger.info("继续重试...")
            else:
                raise RuntimeError("Redetection exhausted after execution failures") from e

    # 达到最大重试次数仍不一致
    logger.warning(f"⚠️  达到最大重试次数({max_retries})，使用最后一次结果")
    raise ValueError("Redetection exhausted without a consistent result")


def redetect_report_inconsistencies(
    report_data: Dict,
    patient_id: str,
    detect_func,
    llm_structured,
    max_retries: int = 3
) -> Dict:
    """
    重新检测报告中所有不一致的检测结果

    Args:
        report_data: 检测报告数据字典
        patient_id: 患者ID
        detect_func: detect_single_sentence函数
        llm_structured: 配置了结构化输出的LLM
        max_retries: 每个句子的最大重试次数

    Returns:
        修正后的报告数据字典
    """
    from curaview._03_detection.validation.schemas import DetectionReport, SentenceDetectionRecord

    # 解析报告
    report = DetectionReport(**report_data)

    logger.info("\n" + "=" * 80)
    logger.info(f"开始重新检测报告: {report.patient_id}")
    logger.info("=" * 80)

    # 找出所有不一致的记录
    inconsistent_indices = []
    for record in report.records:
        is_consistent, _ = DetectionValidator.validate_consistency_with_context(
            record.detection_result.hallucination_status,
            record.detection_result.evidence_grade.value,
            record.detection_result.hallucination_type.value if record.detection_result.hallucination_type else None,
            record.detection_result.reasoning
        )
        if not is_consistent:
            inconsistent_indices.append(record.sentence_index)

    if not inconsistent_indices:
        logger.info("✅ 未发现不一致的记录，无需重新检测")
        return report_data

    logger.info(f"发现 {len(inconsistent_indices)} 条不一致记录: {inconsistent_indices}")

    # 逐个重新检测
    redetect_attempts = 0
    successful_redetections = 0

    for i, record in enumerate(report.records):
        if record.sentence_index in inconsistent_indices:
            redetected_result, attempts = redetect_inconsistent_result(
                record,
                patient_id,
                detect_func,
                llm_structured,
                max_retries
            )

            # 更新记录
            report.records[i].detection_result = redetected_result
            redetect_attempts += attempts

            # 验证是否成功
            is_consistent, _ = DetectionValidator.validate_consistency_with_context(
                redetected_result.hallucination_status,
                redetected_result.evidence_grade.value,
                redetected_result.hallucination_type.value if redetected_result.hallucination_type else None,
                redetected_result.reasoning
            )
            if is_consistent:
                successful_redetections += 1

    # 更新报告统计
    hallucination_count = sum(
        1 for rec in report.records
        if rec.detection_result.hallucination_status
    )
    report.detected_hallucinations = hallucination_count
    report.hallucination_rate = hallucination_count / len(report.records) if report.records else 0
    report.hallucinated_indices = [
        rec.sentence_index for rec in report.records
        if rec.detection_result.hallucination_status
    ]

    logger.info("\n" + "=" * 80)
    logger.info("重新检测汇总")
    logger.info("=" * 80)
    logger.info(f"需要重新检测的记录: {len(inconsistent_indices)}")
    logger.info(f"成功修正: {successful_redetections}")
    logger.info(f"总重新检测尝试次数: {redetect_attempts}")
    logger.info("=" * 80)

    return report.model_dump()
