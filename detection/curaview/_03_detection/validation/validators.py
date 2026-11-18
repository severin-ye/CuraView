"""
数据验证模块 - 对检测结果进行多层次验证

1. JSON格式验证
2. 幻觉类型验证
3. 证据等级与幻觉标签严格一致性验证（E1/E2=False，E3/E4=True，无例外）
   Strict evidence/status consistency: E1/E2=False; E3/E4=True, without exceptions.
"""

from typing import Dict, List, Tuple, Optional

from curaview._03_detection.validation.schemas import HallucinationDetectionResult, HallucinationType, EvidenceGrade, SentenceDetectionRecord, DetectionReport


class ValidationError_(Exception):
    """自定义验证错误"""
    pass


class DetectionValidator:
    """检测结果验证器"""

    VALID_HALLUCINATION_TYPES = {
        "diagnosis_error",
        "medication_error",
        "exam_result_error",
        "time_error",
        "value_error",
        "negation_error",
        "invented_fact",
    }

    VALID_EVIDENCE_GRADES = {"E1", "E2", "E3", "E4"}

    EVIDENCE_TO_HALLUCINATION = {
        "E1": False,
        "E2": False,
        "E3": True,
        "E4": True,
    }

    @classmethod
    def validate_json_format(cls, result_dict: Dict) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        required_fields = ["reasoning", "hallucination_status", "evidence_grade"]
        for field in required_fields:
            if field not in result_dict:
                errors.append(f"❌ 缺少必需字段: {field}")

        if "reasoning" in result_dict and not isinstance(result_dict["reasoning"], str):
            errors.append(
                f"❌ 字段 'reasoning' 应为字符串，实际为: {type(result_dict['reasoning']).__name__}"
            )

        if "hallucination_status" in result_dict and not isinstance(
            result_dict["hallucination_status"], bool
        ):
            errors.append(
                f"❌ 字段 'hallucination_status' 应为布尔值，实际为: {type(result_dict['hallucination_status']).__name__}"
            )

        if "hallucination_type" in result_dict:
            val = result_dict["hallucination_type"]
            if val is not None and not isinstance(val, str):
                errors.append(
                    f"❌ 字段 'hallucination_type' 应为字符串或 null，实际为: {type(val).__name__}"
                )

        if "evidence_grade" in result_dict and not isinstance(
            result_dict["evidence_grade"], str
        ):
            errors.append(
                f"❌ 字段 'evidence_grade' 应为字符串，实际为: {type(result_dict['evidence_grade']).__name__}"
            )

        return len(errors) == 0, errors

    @classmethod
    def validate_hallucination_type(
        cls, hallucination_status: bool, hallucination_type: Optional[str]
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        if hallucination_status:
            if hallucination_type is None:
                errors.append("❌ 当 hallucination_status=True 时，hallucination_type 不能为 null")
            elif hallucination_type not in cls.VALID_HALLUCINATION_TYPES:
                errors.append(
                    f"❌ 非法的幻觉类型: '{hallucination_type}'. 有效类型: {cls.VALID_HALLUCINATION_TYPES}"
                )
        else:
            if hallucination_type is not None:
                errors.append(
                    f"❌ 当 hallucination_status=False 时，hallucination_type 应为 null，但实际为: '{hallucination_type}'"
                )

        return len(errors) == 0, errors

    @classmethod
    def validate_evidence_grade(cls, evidence_grade: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if evidence_grade not in cls.VALID_EVIDENCE_GRADES:
            errors.append(
                f"❌ 非法的证据等级: '{evidence_grade}'. 有效等级: {cls.VALID_EVIDENCE_GRADES}"
            )
        return len(errors) == 0, errors

    @classmethod
    def validate_consistency(
        cls, hallucination_status: bool, evidence_grade: str
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        expected_status = cls.EVIDENCE_TO_HALLUCINATION.get(evidence_grade)

        if expected_status is not None and hallucination_status != expected_status:
            expected_str = "True(有幻觉)" if expected_status else "False(无幻觉)"
            actual_str = "True(有幻觉)" if hallucination_status else "False(无幻觉)"
            errors.append(
                "❌ 幻觉标签与证据等级不一致:\n"
                f"   证据等级 {evidence_grade} 对应期望: {expected_str}\n"
                f"   但实际为: {actual_str}\n"
                "   → 需要重新检查上下文进行修正"
            )

        return len(errors) == 0, errors

    @classmethod
    def validate_consistency_with_context(
        cls,
        hallucination_status: bool,
        evidence_grade: str,
        hallucination_type: Optional[str],
        reasoning: str,
    ) -> Tuple[bool, List[str]]:
        """验证幻觉标签与证据等级的一致性

        规则：
        - E1/E2 → False (无幻觉)
        - E3/E4 → True (有幻觉)

        不再有任何例外情况
        """
        errors: List[str] = []

        expected_status = cls.EVIDENCE_TO_HALLUCINATION.get(evidence_grade)

        if expected_status is not None and hallucination_status != expected_status:
            expected_str = "True(有幻觉)" if expected_status else "False(无幻觉)"
            actual_str = "True(有幻觉)" if hallucination_status else "False(无幻觉)"
            errors.append(
                "❌ 幻觉标签与证据等级不一致:\n"
                f"   证据等级 {evidence_grade} 对应期望: {expected_str}\n"
                f"   但实际为: {actual_str}"
            )

        return len(errors) == 0, errors

    @classmethod
    def validate_detection_result(cls, result_dict: Dict) -> Tuple[bool, Dict]:
        report = {
            "passed": False,
            "json_format_valid": False,
            "type_valid": False,
            "grade_valid": False,
            "consistency_valid": False,
            "errors": [],
            "warnings": [],
        }

        json_valid, json_errors = cls.validate_json_format(result_dict)
        report["json_format_valid"] = json_valid
        report["errors"].extend(json_errors)
        if not json_valid:
            return False, report

        halluc_status = result_dict.get("hallucination_status")
        halluc_type = result_dict.get("hallucination_type")
        type_valid, type_errors = cls.validate_hallucination_type(
            halluc_status, halluc_type
        )
        report["type_valid"] = type_valid
        report["errors"].extend(type_errors)

        evidence_grade = result_dict.get("evidence_grade")
        grade_valid, grade_errors = cls.validate_evidence_grade(evidence_grade)
        report["grade_valid"] = grade_valid
        report["errors"].extend(grade_errors)

        if halluc_status is not None and evidence_grade:
            consistency_valid, consistency_errors = cls.validate_consistency(
                halluc_status, evidence_grade
            )
            report["consistency_valid"] = consistency_valid
            if not consistency_valid:
                report["warnings"].extend(consistency_errors)

        report["passed"] = (
            report["json_format_valid"]
            and report["type_valid"]
            and report["grade_valid"]
            and report["consistency_valid"]
        )
        return report["passed"], report

    @classmethod
    def validate_detection_record(cls, record: SentenceDetectionRecord) -> Tuple[bool, Dict]:
        result_dict = record.detection_result.model_dump()
        return cls.validate_detection_result(result_dict)

    @classmethod
    def validate_detection_report(cls, report: DetectionReport) -> Dict:
        summary = {
            "total_records": len(report.records),
            "passed": 0,
            "failed": 0,
            "failed_indices": [],
            "inconsistency_warnings": [],
            "details": [],
        }

        for record in report.records:
            passed, validation_report = cls.validate_detection_record(record)
            if passed:
                summary["passed"] += 1
            else:
                summary["failed"] += 1
                summary["failed_indices"].append(record.sentence_index)
                summary["details"].append(
                    {"sentence_index": record.sentence_index, "errors": validation_report["errors"]}
                )

            if not validation_report.get("consistency_valid", True):
                summary["inconsistency_warnings"].extend(validation_report["warnings"])

        return summary


class ConsistencyChecker:
    """幻觉标签与证据等级一致性检查器"""

    @staticmethod
    def get_expected_status(evidence_grade: str) -> Optional[bool]:
        mapping = {
            "E1": False,
            "E2": False,
            "E3": True,
            "E4": True,
        }
        return mapping.get(evidence_grade)

    @staticmethod
    def check_sentence(record: SentenceDetectionRecord) -> Dict:
        result = record.detection_result
        expected = ConsistencyChecker.get_expected_status(result.evidence_grade.value)
        actual = result.hallucination_status
        is_consistent = expected == actual
        return {
            "sentence_index": record.sentence_index,
            "sentence": record.original_sentence[:100],
            "evidence_grade": result.evidence_grade.value,
            "expected_status": expected,
            "actual_status": actual,
            "consistent": is_consistent,
            "hallucination_type": result.hallucination_type.value if result.hallucination_type else None,
            "reasoning": result.reasoning[:200] if result.reasoning else "",
        }
