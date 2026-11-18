"""
三层验证日志生成模块

生成详细的JSON格式验证日志，记录：
1. 错误率统计
2. 具体哪些句子出错
3. 错误的原因和详情
4. 修正前后的对比
"""

import os
from curaview.common.source_layout import retained_path
import json
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

from curaview._03_detection.validation.schemas import DetectionReport, SentenceDetectionRecord
from curaview._03_detection.validation.validators import DetectionValidator
from curaview.common.logger import setup_logger

logger = setup_logger()


class ValidationError(BaseModel):
    """单个验证错误"""
    sentence_index: int
    error_type: str  # "format", "type", "consistency"
    error_message: str
    original_result: Dict
    corrected_result: Optional[Dict] = None
    correction_attempts: Optional[int] = None
    correction_success: bool = False


class ValidationLog(BaseModel):
    """三层验证日志"""
    patient_id: str
    total_sentences: int

    # 验证统计
    format_errors_count: int
    type_errors_count: int
    consistency_errors_count: int
    total_errors_count: int
    error_rate: float

    # 错误详情
    format_errors: List[ValidationError]
    type_errors: List[ValidationError]
    consistency_errors: List[ValidationError]

    # 修正统计
    redetection_attempts_total: int
    successfully_corrected: int
    correction_success_rate: float

    # 元数据
    validation_timestamp: str
    output_file_path: str


def extract_detection_result_dict(record: SentenceDetectionRecord) -> Dict:
    """提取检测结果为字典格式"""
    result = record.detection_result
    return {
        "sentence_index": record.sentence_index,
        "original_sentence": record.original_sentence,
        "hallucination_status": result.hallucination_status,
        "hallucination_type": result.hallucination_type.value if result.hallucination_type else None,
        "evidence_grade": result.evidence_grade.value,
        "reasoning": result.reasoning[:200] + "..." if len(result.reasoning) > 200 else result.reasoning
    }


def validate_report_and_generate_log(
    report_data: Dict,
    output_dir: str = None
) -> ValidationLog:
    """
    验证报告并生成详细日志

    Args:
        report_data: 检测报告数据字典
        output_dir: 日志输出目录，默认为 outputs/三层验证_output

    Returns:
        ValidationLog对象
    """
    report = DetectionReport(**report_data)
    patient_id = report.patient_id
    total_sentences = report.total_sentences

    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(
            os.environ.get('CURAVIEW_DETECTION_OUTPUT_ROOT', str(retained_path("."))),
            "outputs",
            "三层验证_output"
        )
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"开始三层验证: {patient_id} ({total_sentences} 个句子)")
    logger.info(f"{'='*80}")

    # 初始化错误列表
    format_errors: List[ValidationError] = []
    type_errors: List[ValidationError] = []
    consistency_errors: List[ValidationError] = []

    # 遍历每条记录进行验证
    for record in report.records:
        result = record.detection_result

        # Layer 1: JSON格式验证（已通过Pydantic验证，此处形式上检查）
        try:
            # 如果能解析说明格式正确
            _ = result.model_dump()
        except Exception as e:
            format_errors.append(ValidationError(
                sentence_index=record.sentence_index,
                error_type="format",
                error_message=f"JSON格式错误: {str(e)}",
                original_result=extract_detection_result_dict(record)
            ))
            continue

        # Layer 2: 幻觉类型验证
        is_valid_type, type_errors_list = DetectionValidator.validate_hallucination_type(
            result.hallucination_status,
            result.hallucination_type.value if result.hallucination_type else None
        )
        if not is_valid_type:
            type_errors.append(ValidationError(
                sentence_index=record.sentence_index,
                error_type="type",
                error_message="; ".join(type_errors_list),
                original_result=extract_detection_result_dict(record)
            ))

        # Layer 3: 一致性验证
        is_consistent, consistency_errors_list = DetectionValidator.validate_consistency_with_context(
            result.hallucination_status,
            result.evidence_grade.value,
            result.hallucination_type.value if result.hallucination_type else None,
            result.reasoning
        )
        if not is_consistent:
            consistency_errors.append(ValidationError(
                sentence_index=record.sentence_index,
                error_type="consistency",
                error_message="; ".join(consistency_errors_list),
                original_result=extract_detection_result_dict(record)
            ))

    # 统计错误
    total_errors = len(format_errors) + len(type_errors) + len(consistency_errors)
    error_rate = total_errors / total_sentences if total_sentences > 0 else 0

    # 生成日志对象
    log = ValidationLog(
        patient_id=patient_id,
        total_sentences=total_sentences,
        format_errors_count=len(format_errors),
        type_errors_count=len(type_errors),
        consistency_errors_count=len(consistency_errors),
        total_errors_count=total_errors,
        error_rate=error_rate,
        format_errors=format_errors,
        type_errors=type_errors,
        consistency_errors=consistency_errors,
        redetection_attempts_total=0,
        successfully_corrected=0,
        correction_success_rate=0.0,
        validation_timestamp=datetime.now().isoformat(),
        output_file_path=""
    )

    # 保存日志
    log_file = os.path.join(output_dir, f"{patient_id}_validation_log.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log.model_dump(), f, ensure_ascii=False, indent=2)

    log.output_file_path = log_file

    # 打印统计信息
    logger.info(f"\n{'='*80}")
    logger.info(f"三层验证结果统计")
    logger.info(f"{'='*80}")
    logger.info(f"患者ID: {patient_id}")
    logger.info(f"总句子数: {total_sentences}")
    logger.info(f"总错误数: {total_errors}")
    logger.info(f"错误率: {error_rate:.2%}")
    logger.info(f"  - 格式错误: {len(format_errors)}")
    logger.info(f"  - 类型错误: {len(type_errors)}")
    logger.info(f"  - 一致性错误: {len(consistency_errors)}")
    logger.info(f"日志已保存: {log_file}")
    logger.info(f"{'='*80}\n")

    return log


def update_log_with_correction_results(
    log: ValidationLog,
    corrected_report: Dict,
    redetection_attempts: int,
    output_dir: str = None
) -> ValidationLog:
    """
    更新日志，添加修正结果

    Args:
        log: 原始验证日志
        corrected_report: 修正后的报告数据
        redetection_attempts: 总重新检测尝试次数
        output_dir: 日志输出目录

    Returns:
        更新后的ValidationLog对象
    """
    report = DetectionReport(**corrected_report)

    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(
            os.environ.get('CURAVIEW_DETECTION_OUTPUT_ROOT', str(retained_path("."))),
            "outputs",
            "三层验证_output"
        )

    # 更新一致性错误的修正结果
    successfully_corrected = 0
    for error in log.consistency_errors:
        # 找到对应的修正后记录
        for record in report.records:
            if record.sentence_index == error.sentence_index:
                # 验证修正后是否一致
                result = record.detection_result
                is_consistent, _ = DetectionValidator.validate_consistency_with_context(
                    result.hallucination_status,
                    result.evidence_grade.value,
                    result.hallucination_type.value if result.hallucination_type else None,
                    result.reasoning
                )

                if is_consistent:
                    error.corrected_result = extract_detection_result_dict(record)
                    error.correction_success = True
                    successfully_corrected += 1
                break

    # 更新统计
    log.redetection_attempts_total = redetection_attempts
    log.successfully_corrected = successfully_corrected
    log.correction_success_rate = (
        successfully_corrected / len(log.consistency_errors)
        if len(log.consistency_errors) > 0 else 1.0
    )

    # 保存更新后的日志
    log_file = os.path.join(output_dir, f"{log.patient_id}_validation_log.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log.model_dump(), f, ensure_ascii=False, indent=2)

    log.output_file_path = log_file

    # 打印修正结果
    logger.info(f"\n{'='*80}")
    logger.info(f"修正结果统计")
    logger.info(f"{'='*80}")
    logger.info(f"患者ID: {log.patient_id}")
    logger.info(f"需要修正: {len(log.consistency_errors)} 个")
    logger.info(f"成功修正: {successfully_corrected} 个")
    logger.info(f"修正成功率: {log.correction_success_rate:.2%}")
    logger.info(f"总重新检测尝试次数: {redetection_attempts}")
    logger.info(f"日志已更新: {log_file}")
    logger.info(f"{'='*80}\n")

    return log
