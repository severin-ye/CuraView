"""GraphRAG查询工具"""

import os
import sys
import asyncio
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.tools import tool

# 添加GraphRAG路径

from graphrag.config.load_config import load_config
import graphrag.api as api
import pandas as pd

# 添加父目录到路径以导入utils
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from curaview.common.logger import setup_logger
from curaview.common.config_loader import detection_config_path
from curaview._99_tracing.capture.detection_distillation import trace_query_call, trace_async_query_call, record_native_query_result

logger = setup_logger()

# 加载配置以获取当前模式
config_path = detection_config_path()
with open(config_path, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)


@trace_async_query_call
async def _async_query_graphrag(
    sentence: str,
    patient_id: str,
    method: str = "local"
) -> Dict[str, Any]:
    """
    异步查询GraphRAG (内部函数)

    Args:
        sentence: 待查询的句子
        patient_id: 患者ID
        method: 查询方法 (local/global/drift)

    Returns:
        {
            "response": "GraphRAG的文本回答",
            "context": {
                "entities": [...],
                "relationships": [...],
                "reports": [...]
            }
        }
    """
    project_root = Path(CONFIG['graphrag']['project_root'])
    if not project_root.is_absolute():
        project_root = (Path(config_path).parent / project_root).resolve()
    if not patient_id.startswith('patient_') or not patient_id.removeprefix('patient_').isascii() or not patient_id.removeprefix('patient_').isdigit():
        raise ValueError('patient_id must be patient_ followed by an ASCII integer')
    index_root = project_root / "output" / "index_output" / patient_id

    if not index_root.exists():
        raise ValueError(f"患者索引不存在: {patient_id}")

    # 加载配置和数据
    logger.debug(f"加载GraphRAG索引: {index_root}")
    config = load_config(index_root)
    output_dir = index_root / "output"

    entities = pd.read_parquet(output_dir / "entities.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")
    text_units = pd.read_parquet(output_dir / "text_units.parquet")
    communities = pd.read_parquet(output_dir / "communities.parquet")
    community_reports = pd.read_parquet(output_dir / "community_reports.parquet")

    # 检查covariates
    covariates_file = output_dir / "covariates.parquet"
    covariates = pd.read_parquet(covariates_file) if covariates_file.exists() else None

    # 设置环境变量
    os.environ['GRAPHRAG_PATIENT_ID'] = patient_id

    # 根据LLM模式设置query_log输出路径
    mode = CONFIG['llm'].get('mode', 'api')
    if mode == 'local':
        # 本地模式：输出到实验目录
        query_log_dir = os.environ.get('CURAVIEW_DETECTION_LOG_ROOT', str(Path(__file__).resolve().parents[3] / 'outputs' / 'query-logs'))
        os.environ['GRAPHRAG_QUERY_LOG_DIR'] = query_log_dir
        logger.info(f"📁 本地模式：query_log → {query_log_dir}")
    else:
        # API模式：输出到默认目录
        query_log_dir = os.environ.get('CURAVIEW_DETECTION_LOG_ROOT', str(Path(__file__).resolve().parents[3] / 'outputs' / 'query-logs'))
        os.environ['GRAPHRAG_QUERY_LOG_DIR'] = query_log_dir
        logger.debug(f"📁 API模式：query_log → {query_log_dir}")

    # 构造查询提示词
    query_prompt = f"""
请提取与以下句子相关的所有医学信息:

句子: {sentence}

请重点关注:
1. 句子中提到的实体(疾病、药物、检查、时间等)
2. 这些实体之间的关系
3. 相关的医学事实和数据
"""

    # 执行查询
    logger.debug(f"执行GraphRAG {method} 查询...")

    if method == "local":
        response, context_data = await api.local_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            covariates=covariates,
            community_level=2,
            response_type="Multiple Paragraphs",
            query=query_prompt,
        )
    elif method == "global":
        response, context_data = await api.global_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            community_level=2,
            dynamic_community_selection=False,
            response_type="Multiple Paragraphs",
            query=query_prompt,
        )
    elif method == "drift":
        response, context_data = await api.drift_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            community_level=2,
            response_type="Multiple Paragraphs",
            query=query_prompt,
        )
    else:
        raise ValueError(f"不支持的查询方法: {method}")

    record_native_query_result(response, context_data)
    # 提取关键信息
    context = {
        "entities": _extract_entities(context_data),
        "relationships": _extract_relationships(context_data),
        "reports": _extract_reports(context_data),
    }

    logger.debug(f"查询成功,提取到 {len(context['entities'])} 个实体")

    return {
        "response": response,
        "context": context
    }


def _extract_entities(context_data) -> list:
    """从context_data提取实体信息"""
    entities = []

    try:
        # 尝试从不同可能的属性提取实体
        if hasattr(context_data, 'entities'):
            for e in context_data.entities[:15]:  # 限制数量
                entities.append(str(e))
        elif hasattr(context_data, 'data') and 'entities' in context_data.data:
            for e in context_data.data['entities'][:15]:
                entities.append(str(e))

        # 如果是DataFrame
        if isinstance(context_data, dict) and 'entities' in context_data:
            df = context_data['entities']
            if isinstance(df, pd.DataFrame) and not df.empty:
                for _, row in df.head(15).iterrows():
                    entity_info = f"{row.get('name', 'Unknown')}: {row.get('description', '')}"
                    entities.append(entity_info)
    except Exception as e:
        logger.warning(f"提取实体失败: {e}")
        raise

    return entities


def _extract_relationships(context_data) -> list:
    """从context_data提取关系信息"""
    relationships = []

    try:
        if hasattr(context_data, 'relationships'):
            for r in context_data.relationships[:15]:
                relationships.append(str(r))
        elif hasattr(context_data, 'data') and 'relationships' in context_data.data:
            for r in context_data.data['relationships'][:15]:
                relationships.append(str(r))

        # 如果是DataFrame
        if isinstance(context_data, dict) and 'relationships' in context_data:
            df = context_data['relationships']
            if isinstance(df, pd.DataFrame) and not df.empty:
                for _, row in df.head(15).iterrows():
                    rel_info = f"{row.get('source', '')} -> {row.get('target', '')}: {row.get('description', '')}"
                    relationships.append(rel_info)
    except Exception as e:
        logger.warning(f"提取关系失败: {e}")
        raise

    return relationships


def _extract_reports(context_data) -> list:
    """从context_data提取报告摘要"""
    reports = []

    try:
        if hasattr(context_data, 'reports'):
            for r in context_data.reports[:5]:  # 报告内容较长,限制数量
                reports.append(str(r)[:300])  # 截断到300字符
        elif hasattr(context_data, 'data') and 'reports' in context_data.data:
            for r in context_data.data['reports'][:5]:
                reports.append(str(r)[:300])

        # 如果是DataFrame
        if isinstance(context_data, dict) and 'community_reports' in context_data:
            df = context_data['community_reports']
            if isinstance(df, pd.DataFrame) and not df.empty:
                for _, row in df.head(5).iterrows():
                    report_text = row.get('full_content', row.get('summary', ''))
                    reports.append(str(report_text)[:300])
    except Exception as e:
        logger.warning(f"提取报告失败: {e}")
        raise

    return reports


@tool
@trace_query_call
def query_graphrag_for_sentence(sentence: str, patient_id: str = "patient_2") -> str:
    """
    使用GraphRAG查询与句子相关的医学知识

    这是幻觉检测的核心工具。它会调用GraphRAG知识图谱,检索与待检测句子相关的医学实体、关系和报告。

    Args:
        sentence: 待检测的句子
        patient_id: 患者ID,默认为patient_2

    Returns:
        JSON格式的查询结果,包含:
        - response: GraphRAG的文本回答
        - context: 结构化的上下文信息
            - entities: 相关实体列表
            - relationships: 相关关系列表
            - reports: 社区报告摘要列表

    Example:
        >>> result = query_graphrag_for_sentence.invoke({
        ...     "sentence": "The synthetic marker is blue.",
        ...     "patient_id": "patient_2"
        ... })
        >>> data = json.loads(result)
        >>> print(data['response'])
    """
    try:
        # 创建新的事件循环 (解决asyncio冲突)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_async_query_graphrag(sentence, patient_id))
        finally:
            # 清理所有待处理的任务
            try:
                # 取消所有待处理的任务
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                # 等待任务取消完成
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()

        # 序列化为JSON
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"GraphRAG查询失败: {e}")

        # Execution errors are not empty successful evidence.
        raise
