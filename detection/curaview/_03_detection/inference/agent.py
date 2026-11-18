"""GraphRAG幻觉检测Agent定义"""

import os
import sys
import yaml
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from curaview._03_detection.sentences.sentence_splitter import split_sentences
from curaview._02_graph.retrieval.graphrag_query_tool import query_graphrag_for_sentence
from curaview._03_detection.validation.schemas import HallucinationDetectionResult
from curaview._03_detection.validation.detection_errors import require_query_evidence
from curaview._03_detection.prompts.prompt_loader import load_detection_prompt
from curaview.common.logger import setup_logger
from curaview._99_tracing.capture.detection_distillation import get_trace_store, trace_detection_call
from curaview.common.config_loader import detection_config_path

logger = setup_logger()

# 加载配置
config_path = detection_config_path()
with open(config_path, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

# 全局HTTP客户端，用于清理
_global_http_client = None


def create_detection_agent():
    """
    创建幻觉检测Agent

    Returns:
        (agent, llm_structured) 元组
    """
    llm_config = CONFIG['llm']
    mode = llm_config.get('mode', 'api')

    logger.info(f"🔧 LLM模式: {mode}")

    if mode == 'local':
        # 本地模型模式
        logger.info("📦 使用本地模型")
        from curaview.backends.local.local_llm_json import create_local_llm
        local_config = llm_config['local']
        llm = create_local_llm(local_config)
        llm.trace_store = get_trace_store()
        logger.info(f"✅ 本地模型加载完成: {local_config['model_path']}")

    else:
        # API模式
        logger.info("🌐 使用API模式")
        api_config = llm_config['api']

        # 从环境变量读取API Key
        api_key_env = api_config['api_key'].strip('${}')
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(f"请设置环境变量: {api_key_env}")

        # 配置HTTP客户端参数以避免连接泄漏
        global _global_http_client
        import httpx
        _global_http_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )

        llm = ChatOpenAI(
            model=api_config['model'],
            temperature=api_config['temperature'],
            max_tokens=api_config['max_tokens'],
            api_key=api_key,
            base_url=api_config.get('base_url'),
            http_client=_global_http_client
        )
        logger.info(f"✅ API配置完成: {api_config['model']}")

    # 所有模式统一使用with_structured_output方法
    # API模式：LangChain提供
    # 本地模式：我们自己实现
    llm_structured = llm.with_structured_output(HallucinationDetectionResult)
    logger.info("✅ 结构化输出已配置")

    # 加载系统提示词
    system_prompt = load_detection_prompt()

    # 定义工具列表（仅API模式需要agent）
    tools = [
        split_sentences,
        query_graphrag_for_sentence
    ]

    # 创建Agent（仅API模式）
    agent = None
    if mode == 'api':
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=InMemorySaver()
        )
        logger.info("✅ Agent创建成功")
    else:
        logger.info(f"ℹ️  本地模式不使用Agent架构，直接调用模型")

    return agent, llm_structured


def _get_distillation_writer():
    """Load the optional legacy writer before making any query/model call.

    Explicitly setting output.save_detection_distillation=false disables this
    derived record, not actual model trace capture. Missing source is an error
    when enabled; no replacement writer is invented.
    """
    enabled = CONFIG.get('output', {}).get('save_detection_distillation', True)
    if not isinstance(enabled, bool):
        raise ValueError("output.save_detection_distillation must be a boolean")
    if not enabled:
        return None
    try:
        from curaview._99_tracing.capture.detection_distillation import save_detection_distillation
    except ImportError as exc:
        raise RuntimeError(
            "Detection distillation is enabled but utils.detection_distillation is unavailable; "
            "recover its source or explicitly set output.save_detection_distillation=false"
        ) from exc
    return save_detection_distillation


@trace_detection_call
def detect_single_sentence(
    sentence: str,
    patient_id: str,
    llm_structured
) -> dict:
    """
    检测单个句子的幻觉

    Args:
        sentence: 待检测的句子
        patient_id: 患者ID
        llm_structured: 配置了结构化输出的LLM

    Returns:
        {
            "detection_result": HallucinationDetectionResult对象,
            "graphrag_evidence": GraphRAG原始证据字符串
        }
    """
    distillation_writer = _get_distillation_writer()
    # Step 1: 调用GraphRAG查询
    logger.info(f"查询GraphRAG: {sentence[:50]}...")
    graphrag_result = query_graphrag_for_sentence.invoke({
        "sentence": sentence,
        "patient_id": patient_id
    })
    graphrag_result = require_query_evidence(graphrag_result)

    # Step 2: 构造检测提示词
    detection_system_prompt = load_detection_prompt()
    user_prompt = f"""
待检测句子: "{sentence}"

GraphRAG检索到的相关证据:
{graphrag_result}

请基于以上证据,检测该句子是否存在医学幻觉。
"""

    # Step 3: 调用LLM进行检测
    logger.info("正在进行幻觉检测...")
    try:
        detection_result = llm_structured.invoke([
            {"role": "system", "content": detection_system_prompt},
            {"role": "user", "content": user_prompt}
        ])
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        raise

    # Step 4: 保存检测阶段的蒸馏记录
    try:
        if distillation_writer is not None:
            distillation_writer(
                patient_id=patient_id,
                sentence=sentence,
                system_prompt=detection_system_prompt,
                user_prompt=user_prompt,
                detection_result=detection_result,
                graphrag_evidence=graphrag_result
            )
    except Exception as e:
        logger.warning(f"保存detection distillation失败: {e}")
        raise

    # Step 5: 组装结果
    return {
        "detection_result": detection_result,
        "graphrag_evidence": graphrag_result,
        "distillation_status": "saved" if distillation_writer is not None else "disabled",
    }



def cleanup_agent_resources():
    """清理Agent相关的HTTP资源"""
    global _global_http_client

    if _global_http_client is not None:
        try:
            _global_http_client.close()
            logger.info("✅ HTTP客户端已关闭")
            _global_http_client = None
        except Exception as e:
            logger.debug(f"关闭HTTP客户端失败: {e}")
