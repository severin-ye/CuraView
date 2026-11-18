"""Pydantic数据模型定义"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime


class HallucinationType(str, Enum):
    """幻觉类型枚举"""
    DIAGNOSIS_ERROR = "diagnosis_error"
    MEDICATION_ERROR = "medication_error"
    EXAM_RESULT_ERROR = "exam_result_error"
    TIME_ERROR = "time_error"
    VALUE_ERROR = "value_error"
    NEGATION_ERROR = "negation_error"
    INVENTED_FACT = "invented_fact"


class EvidenceGrade(str, Enum):
    """证据等级枚举"""
    E1 = "E1"  # 患者证据明确直接支持（结构化或非结构化） / explicit direct patient-evidence support
    E2 = "E2"  # 由患者证据间接推断支持 / indirect inference from patient evidence
    E3 = "E3"  # 检索证据无支持且无直接矛盾 / no retrieved support and no direct contradiction
    E4 = "E4"  # 与明确患者事实直接矛盾，非伤害等级 / direct contradiction, not harm severity


class HallucinationDetectionResult(BaseModel):
    """幻觉检测结果 - AI输出部分"""
    reasoning: str = Field(..., description="有无幻觉的原因和思考过程")
    hallucination_status: bool = Field(..., description="是否存在幻觉,True表示存在")
    hallucination_type: Optional[HallucinationType] = Field(None, description="幻觉类型,仅当存在幻觉时填写")
    evidence_grade: EvidenceGrade = Field(..., description="证据等级")


class SentenceDetectionRecord(BaseModel):
    """单个句子的完整检测记录"""
    sentence_index: int = Field(..., description="句子编号")
    original_sentence: str = Field(..., description="原始句子")
    detection_result: HallucinationDetectionResult = Field(..., description="AI检测结果")
    graphrag_evidence: str = Field(..., description="GraphRAG提供的原始证据")
    distillation_status: Optional[str] = Field(None, description="saved / disabled；旧记录未提供时为空")


class DetectionReport(BaseModel):
    """完整的检测报告"""
    patient_id: str = Field(..., description="患者ID")
    total_sentences: int = Field(..., description="总句子数")
    successful_sentences: Optional[int] = Field(None, description="有效检测分母；旧报告未记录时为空")
    failed_sentences: List[dict] = Field(default_factory=list, description="失败记录，无医学标签")
    invalid_sentences: List[dict] = Field(default_factory=list, description="无效结果保留原判定，不进入有效分母")
    redetection_failures: List[dict] = Field(default_factory=list, description="重检失败或仍不一致；不覆盖原判定")
    processing_status: str = Field(default="unknown", description="completed / partial / failed；旧报告未记录时unknown")

    # 总体幻觉统计
    detected_hallucinations: int = Field(..., description="检测到的幻觉数量")
    hallucination_rate: Optional[float] = Field(..., description="有效检测中幻觉比例；无有效检测时为空")
    hallucinated_indices: List[int] = Field(default_factory=list, description="检测到幻觉的句子索引列表")

    # E3等级幻觉统计（无证据支持）
    e3_hallucinations: int = Field(default=0, description="E3等级幻觉数量")
    e3_hallucination_rate: Optional[float] = Field(default=0.0, description="E3幻觉比例")
    e3_hallucinated_indices: List[int] = Field(default_factory=list, description="E3幻觉句子索引列表")

    # E4等级幻觉统计（与证据冲突）
    e4_hallucinations: int = Field(default=0, description="E4等级幻觉数量")
    e4_hallucination_rate: Optional[float] = Field(default=0.0, description="E4幻觉比例")
    e4_hallucinated_indices: List[int] = Field(default_factory=list, description="E4幻觉句子索引列表")

    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat(), description="报告最后更新时间")
    records: List[SentenceDetectionRecord] = Field(..., description="所有检测记录")


class GraphRAGContext(BaseModel):
    """GraphRAG返回的上下文数据"""
    entities: List[str] = Field(default_factory=list, description="相关实体列表")
    relationships: List[str] = Field(default_factory=list, description="相关关系列表")
    community_reports: List[str] = Field(default_factory=list, description="社区报告列表")
    raw_response: str = Field(..., description="GraphRAG的原始响应文本")
