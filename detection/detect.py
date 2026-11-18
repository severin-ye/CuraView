#!/usr/bin/env python3
"""
GraphRAG幻觉检测 - 主程序

本模块包含幻觉检测系统的核心功能实现，包括：
- 患者病历检测
- 批量处理
- 句子分割和验证
- 结果报告生成
- 进度跟踪
- 后台运行支持

使用方式:
    1. 交互式模式（推荐）:
       python detect.py

    2. 指定文本文件:
       python detect.py path/to/text.txt --patient-id patient_2

    3. 显示详细日志:
       python detect.py --verbose

    4. 查看运行状态:
       python detect.py --status

环境要求:
    - 已激活虚拟环境 .env-RAG
    - 已设置环境变量 GRAPHRAG_API_KEY (API模式)
    - 已配置 config.yaml
"""

import os
import sys
import json
import yaml
import argparse
import asyncio
import gc
import time
import warnings
import subprocess
import atexit
from typing import List, Tuple
from pathlib import Path
_SOURCE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SOURCE_ROOT/'src'))
sys.path.insert(0, str(_SOURCE_ROOT))
from curaview.common.source_layout import retained_path, project_root as layout_project_root
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))
# 添加shared_utils到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from curaview._03_detection.inference.agent import create_detection_agent, detect_single_sentence
from curaview._03_detection.sentences.sentence_splitter import split_sentences
from curaview._03_detection.validation.schemas import SentenceDetectionRecord, DetectionReport
from curaview.common.logger import setup_logger
from curaview.common.config_loader import detection_config_path

from curaview._03_detection.validation.validators import DetectionValidator, ConsistencyChecker
# 导入进度跟踪器
from curaview.common.progress_tracker import ProgressTracker

logger = setup_logger()


def format_time(seconds):
    """Duration display without the absent historical time_maker package."""
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_rate(rate):
    return f'{rate:.2%}' if rate is not None else '未确定'


def partition_valid_records(records):
    """Apply the existing validation contract without changing model labels."""
    valid_records, invalid_records = [], []
    for record in records:
        result = record.detection_result
        type_valid, type_errors = DetectionValidator.validate_hallucination_type(
            result.hallucination_status,
            result.hallucination_type.value if result.hallucination_type is not None else None)
        consistent, consistency_errors = DetectionValidator.validate_consistency_with_context(
            result.hallucination_status, result.evidence_grade.value,
            result.hallucination_type.value if result.hallucination_type is not None else None,
            result.reasoning)
        if type_valid and consistent:
            valid_records.append(record)
        else:
            invalid_records.append({'sentence_index': record.sentence_index, 'status': 'invalid',
                'errors': type_errors + consistency_errors,
                'original_record': record.model_dump(mode='json')})
    return valid_records, invalid_records

# 加载配置
config_path = detection_config_path()
with open(config_path, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

# 注册退出时的资源清理钩子
def _cleanup_on_exit():
    """程序退出时自动清理资源"""
    try:
        # 静默清理，避免退出时的日志干扰
        import litellm
        if hasattr(litellm, 'module_level_aclient') and litellm.module_level_aclient:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(litellm.module_level_aclient.aclose())
                loop.close()
            except:
                pass
    except:
        pass

atexit.register(_cleanup_on_exit)


def get_output_config():
    """根据当前LLM模式和模型获取输出配置"""
    mode = CONFIG['llm'].get('mode', 'api')
    base_dir_config = os.environ.get('CURAVIEW_DETECTION_OUTPUT_ROOT', CONFIG['output']['base_dir'])

    # 处理相对路径：相对于配置文件所在目录
    if not os.path.isabs(base_dir_config):
        config_dir = os.path.dirname(config_path)
        base_dir = Path(config_dir) / base_dir_config
    else:
        base_dir = Path(base_dir_config)

    # 获取模型名称
    if mode == 'api':
        model_name = CONFIG['llm']['api']['model']  # 如: qwen-plus, qwen-max
    else:
        # 本地模式: 从路径提取模型名称
        model_path = CONFIG['llm']['local']['model_path']
        # 提取最后一部分并转小写，如: Qwen3-14B-Base -> qwen3-14b-base
        model_name = Path(model_path).name.lower()

    # 创建模型专属子目录
    model_output_dir = base_dir / model_name

    return {
        'output_dir': str(model_output_dir / 'detection_results'),
        'validation_log_dir': str(model_output_dir / '三层验证_output'),
        'model_name': model_name
    }


def get_patient_text_file(patient_id: str) -> str:
    """
    获取患者幻觉生成文本文件路径（支持根目录和模型子目录）

    Args:
        patient_id: 患者ID，如 "patient_15"

    Returns:
        文本文件的完整路径

    Raises:
        FileNotFoundError: 如果找不到文件
    """
    runtime_text_root = os.environ.get('CURAVIEW_DETECTION_TEXT_ROOT')
    if runtime_text_root:
        root = Path(runtime_text_root).resolve()
        candidate = (root / f'{patient_id}.txt').resolve()
        if candidate.parent != root:
            raise ValueError('Patient text path must stay within runtime text root')
        if not candidate.is_file():
            raise FileNotFoundError(f'找不到 {patient_id} 的运行输入文本')
        return str(candidate)
    # 获取项目根目录(此文件在 ./)
    project_root = layout_project_root()
    base_dir = project_root.parent / 'generation' / 'outputs'
    filename = "rewritten_brief_hospital_course.txt"

    # 优先检查根目录
    root_path = base_dir / patient_id / filename
    if root_path.exists():
        return str(root_path)

    # 检查模型子目录（如 qwen-plus/, qwen-max/, 等）
    for model_dir in base_dir.iterdir() if base_dir.is_dir() else ():
        if model_dir.is_dir() and not model_dir.name.startswith("patient_"):
            model_path = model_dir / patient_id / filename
            if model_path.exists():
                return str(model_path)

    raise FileNotFoundError(f"找不到 {patient_id} 的文本文件")


def find_available_patients() -> List[str]:
    """
    查找同时满足两个条件的患者:
    1. 有GraphRAG知识图谱索引
    2. 有output输出

    Returns:
        患者ID列表,按数字排序
    """
    # 读取配置文件获取graphrag项目路径
    config_path = detection_config_path()
    with open(config_path, encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    # 获取graphrag项目根目录
    graphrag_project_root = Path(config_data['graphrag']['project_root'])
    if not graphrag_project_root.is_absolute():
        graphrag_project_root = (config_path.parent / graphrag_project_root).resolve()

    graphrag_index_dir = graphrag_project_root / "output" / "index_output"

    # 获取幻觉生成输出目录
    project_root = layout_project_root()
    hallucination_output_base = project_root.parent / 'generation' / 'outputs'

    # 查找有索引的患者
    patients_with_index = set()
    if graphrag_index_dir.exists():
        for path in graphrag_index_dir.iterdir():
            suffix = path.name.removeprefix('patient_')
            if path.is_dir() and path.name.startswith('patient_') and suffix.isascii() and suffix.isdigit():
                # 检查是否有必要的索引文件
                output_dir = path / "output"
                if output_dir.exists() and (output_dir / "entities.parquet").exists():
                    patients_with_index.add(path.name)

    # 查找有幻觉输出的患者 (支持模型子目录，如qwen-plus/)
    patients_with_output = set()
    if hallucination_output_base.exists():
        # 先检查根目录
        for path in hallucination_output_base.iterdir():
            if path.is_dir() and path.name.startswith("patient_"):
                rewritten_file = path / "rewritten_brief_hospital_course.txt"
                if rewritten_file.exists():
                    patients_with_output.add(path.name)

        # 再检查模型子目录 (qwen-plus/, qwen-max/, 等)
        for model_dir in hallucination_output_base.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("patient_"):
                for path in model_dir.iterdir():
                    if path.is_dir() and path.name.startswith("patient_"):
                        rewritten_file = path / "rewritten_brief_hospital_course.txt"
                        if rewritten_file.exists():
                            patients_with_output.add(path.name)

    runtime_text_root = os.environ.get('CURAVIEW_DETECTION_TEXT_ROOT')
    if runtime_text_root:
        patients_with_output = {path.stem for path in Path(runtime_text_root).glob('patient_*.txt')
                                if path.stem.removeprefix('patient_').isascii() and path.stem.removeprefix('patient_').isdigit()}

    # 取交集并排序
    available_patients = sorted(
        patients_with_index & patients_with_output,
        key=lambda x: int(x.split('_')[1])
    )

    return available_patients


def parse_patient_selection(user_input: str, available_patients: List[str]) -> List[str]:
    """
    解析用户输入的患者选择

    Args:
        user_input: 用户输入 (空/单个数字/逗号分隔列表/范围)
        available_patients: 可用患者列表

    Returns:
        选中的患者ID列表
    """
    user_input = user_input.strip()

    # 空输入 - 处理所有患者
    if not user_input:
        return available_patients

    # 优先检测逗号分隔的列表: "15,22,64,69" 或 "15, 22, 64" 或 "191-199" 或混合 "15,22,191-199"
    if ',' in user_input or '-' in user_input:
        try:
            selected = []
            # 如果没有逗号,将整个输入作为一个元素处理
            parts = user_input.split(',') if ',' in user_input else [user_input]
            for num_str in parts:
                num_str = num_str.strip()
                if '-' in num_str:
                    # 支持范围: "191-199"
                    start_str, end_str = num_str.split('-', 1)
                    start_num = int(start_str.strip())
                    end_num = int(end_str.strip())
                    if start_num > end_num:
                        start_num, end_num = end_num, start_num
                    for patient_id in available_patients:
                        patient_num = int(patient_id.split('_')[1])
                        if start_num <= patient_num <= end_num:
                            if patient_id not in selected:
                                selected.append(patient_id)
                else:
                    # 单个数字
                    patient_num = int(num_str)
                    patient_id = f"patient_{patient_num}"
                    if patient_id in available_patients:
                        if patient_id not in selected:
                            selected.append(patient_id)
                    else:
                        print(f"⚠️  患者 {patient_id} 不可用(缺少索引或输出文件)")
            return sorted(selected, key=lambda x: int(x.split('_')[1]))
        except ValueError:
            print(f"❌ 无效的逗号分隔列表: {user_input}")
            return []

    # 解析空格分隔的输入
    parts = user_input.split()

    if len(parts) == 1:
        # 单个患者: "2" -> patient_2
        try:
            patient_num = int(parts[0])
            patient_id = f"patient_{patient_num}"
            if patient_id in available_patients:
                return [patient_id]
            else:
                print(f"⚠️  患者 {patient_id} 不可用(缺少索引或输出文件)")
                return []
        except ValueError:
            print(f"❌ 无效输入: {user_input}")
            return []

    elif len(parts) == 2:
        # 范围: "10 35" -> patient_10 到 patient_35
        try:
            start_num = int(parts[0])
            end_num = int(parts[1])

            if start_num > end_num:
                start_num, end_num = end_num, start_num

            selected = []
            for patient_id in available_patients:
                patient_num = int(patient_id.split('_')[1])
                if start_num <= patient_num <= end_num:
                    selected.append(patient_id)

            return selected
        except ValueError:
            print(f"❌ 无效范围: {user_input}")
            return []

    else:
        print(f"❌ 无效输入格式: {user_input}")
        return []


def interactive_select_patients() -> List[str]:
    """
    交互式选择患者

    Returns:
        选中的患者ID列表
    """
    print("\n" + "=" * 80)
    print("GraphRAG幻觉检测 - 患者选择")
    print("=" * 80)

    # 查找可用患者
    available_patients = find_available_patients()

    if not available_patients:
        print("\n❌ 未找到可用患者")
        print("\n需要同时满足:")
        print("  1. GraphRAG索引: graphrag/output/index_output/patient_X/")
        print("  2. 幻觉输出: langchain/hallucination_generation_medical_agent/output/patient_X/rewritten_brief_hospital_course.txt")
        return []

    # 显示可用患者
    print(f"\n✓ 找到 {len(available_patients)} 个可用患者:")
    patient_nums = [int(p.split('_')[1]) for p in available_patients]
    print(f"  {', '.join(map(str, patient_nums))}")

    # 提示输入(循环直到获得有效输入或用户取消)
    while True:
        print("\n请选择要检测的患者:")
        print("  - 直接回车: 处理所有患者")
        print("  - 输入单个数字: 如 '2' (处理patient_2)")
        print("  - 输入逗号分隔: 如 '15,22,64,69' (处理这些患者)")
        print("  - 输入逗号+范围: 如 '15,22,64,69,191-199' (混合使用)")
        print("  - 输入范围: 如 '10 35' (处理patient_10到patient_35)")
        print()

        try:
            user_input = input("请输入: ").strip()
            selected = parse_patient_selection(user_input, available_patients)

            if selected:
                print(f"\n✓ 已选择 {len(selected)} 个患者: {', '.join([p.split('_')[1] for p in selected])}")
                return selected
            else:
                # 如果没有选中任何患者，继续循环重新提示
                print("请重新输入有效的患者编号或范围\n")
                continue

        except KeyboardInterrupt:
            print("\n\n操作已取消")
            return []


def detect_sentence_wrapper(
    sentence_data: Tuple[int, str],
    patient_id: str,
    llm_structured
) -> Tuple[int, object, bool]:
    """
    句子检测包装函数，用于并行处理

    Args:
        sentence_data: (索引, 句子) 元组
        patient_id: 患者ID
        llm_structured: 结构化输出的LLM

    Returns:
        (索引, 检测记录, 是否成功) 元组
    """
    idx, sentence = sentence_data

    try:
        result = detect_single_sentence(sentence, patient_id, llm_structured)

        record = SentenceDetectionRecord(
            sentence_index=idx,
            original_sentence=sentence,
            detection_result=result['detection_result'],
            graphrag_evidence=result['graphrag_evidence'],
            distillation_status=result.get('distillation_status')
        )

        return (idx, record, True)

    except Exception as e:
        import traceback
        logger.error(f"句子 {idx} 检测失败: {e}")
        logger.error(f"详细错误: {traceback.format_exc()}")
        return (idx, {
            "sentence_index": idx,
            "original_sentence": sentence,
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
        }, False)


def process_text_file(
    text_file_path: str,
    patient_id: str = None,
    parallel: bool = None,
    max_workers: int = None,
    shared_agent=None,
    shared_llm_structured=None,
    precomputed_sentences=None,
    batch_progress_info: dict = None
) -> DetectionReport:
    """
    处理整个文本文件,检测所有句子的幻觉（带进度跟踪）

    Args:
        text_file_path: 文本文件路径
        patient_id: 患者ID,默认从配置读取
        parallel: 是否使用并行检测,默认从配置读取
        max_workers: 并行线程数,默认从配置读取
        batch_progress_info: 批量处理进度信息 {"current": 1, "total": 10, "elapsed": 600, "avg_time": 120}
        shared_agent: 共享的agent实例(批量处理时避免重复加载模型)
        shared_llm_structured: 共享的llm_structured实例
        precomputed_sentences: 预先计算的句子列表(批量处理时避免重复分割)

    Returns:
        DetectionReport对象
    """
    if patient_id is None:
        patient_id = CONFIG['graphrag']['default_patient_id']
    suffix = patient_id.removeprefix('patient_')
    if not patient_id.startswith('patient_') or not suffix.isascii() or not suffix.isdigit():
        raise ValueError('patient_id must be patient_ followed by an ASCII integer')

    # 从配置读取并行设置
    if parallel is None:
        parallel = CONFIG['detection'].get('parallel_detection', True)
    if max_workers is None:
        max_workers = CONFIG['detection'].get('max_workers', 4)

    # 定义步骤
    step_names = [
        "验证GraphRAG索引",
        "读取文本文件",
        "分割句子",
        "初始化检测Agent",
        "批量幻觉检测",
        "三层验证",
        "自动修正不一致",
        "生成并保存报告",
        "打印摘要"
    ]

    # 获取输出配置
    output_config = get_output_config()

    # 创建进度跟踪器
    stats_file = os.path.join(
        output_config['output_dir'],
        "..",
        ".progress_stats",
        "detection_stats.json"
    )

    tracker = ProgressTracker(
        task_name=f"{patient_id}幻觉检测",
        total_steps=len(step_names),
        step_names=step_names,
        stats_file=stats_file,
        show_progress_bar=CONFIG.get('display', {}).get('show_progress_bar', True)
    )

    logger.info("=" * 80)
    logger.info(f"GraphRAG幻觉检测任务启动")
    logger.info("=" * 80)
    logger.info(f"患者ID: {patient_id}")
    logger.info(f"输入文件: {text_file_path}")
    logger.info(f"当前模型: {output_config['model_name']}")
    logger.info(f"输出目录: {output_config['output_dir']}")
    logger.info("")

    # Step 0: 验证GraphRAG索引是否存在
    tracker.start_step("验证GraphRAG索引")
    logger.info("Step 0: 验证GraphRAG索引...")

    # 读取配置获取graphrag项目路径
    config_path = detection_config_path()
    with open(config_path, encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    graphrag_project_root = Path(config_data['graphrag']['project_root'])
    if not graphrag_project_root.is_absolute():
        graphrag_project_root = (config_path.parent / graphrag_project_root).resolve()

    index_root = graphrag_project_root / "output" / "index_output" / patient_id

    if not index_root.exists():
        tracker.complete_step("failed")
        error_msg = f"❌ 患者索引目录不存在: {index_root}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    output_dir = index_root / "output"
    entities_file = output_dir / "entities.parquet"

    if not entities_file.exists():
        tracker.complete_step("failed")
        error_msg = f"❌ GraphRAG索引未完成或损坏: 缺少 {entities_file}\n请先运行GraphRAG索引构建"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"✓ GraphRAG索引验证通过: {index_root}")
    tracker.complete_step("completed")

    # Step 1: 读取文件
    tracker.start_step("读取文本文件")
    logger.info("\nStep 1: 读取文本文件...")
    if not os.path.exists(text_file_path):
        tracker.complete_step("failed")
        raise FileNotFoundError(f"文件不存在: {text_file_path}")

    with open(text_file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    logger.info(f"文本长度: {len(full_text)} 字符")
    tracker.complete_step("completed")

    # Step 2: 分割句子 (如果已预先分割则跳过)
    tracker.start_step("分割句子")
    if precomputed_sentences is not None:
        logger.info("\nStep 2: 使用预分割的句子 (跳过分割)...")
        sentence_list = precomputed_sentences
        total_sentences = len(sentence_list)
        logger.info(f"✓ 使用预分割句子: {total_sentences} 个")
    else:
        logger.info("\nStep 2: 分割句子...")

        # 检测句子分割模型状态
        try:
            from curaview._03_detection.sentences.shared_splitter import _model_loaded, _model
            if _model_loaded and _model is not None:
                logger.info("✓ 句子分割模型: 已加载 (SaT模型)")
            else:
                logger.info("✓ 句子分割模型: 使用规则模式")
        except:
            logger.info("✓ 句子分割模型: 使用规则模式")

        sentence_list = split_sentences.invoke({"text": full_text})
        total_sentences = len(sentence_list)
        logger.info(f"总句子数: {total_sentences}")

    if total_sentences == 0:
        tracker.complete_step("failed")
        logger.error("文本中没有有效句子")
        raise ValueError("文本为空或无法分割")
    tracker.complete_step("completed")

    # Step 3: 创建Agent (如果没有共享实例,则创建新的)
    tracker.start_step("初始化检测Agent")
    if shared_llm_structured is None:
        logger.info("\nStep 3: 初始化检测Agent...")
        agent, llm_structured = create_detection_agent()
        logger.info("Agent初始化完成")
    else:
        logger.info("\nStep 3: 使用共享Agent (跳过模型加载)...")
        agent = shared_agent  # 本地模式下可能为None,但没关系
        llm_structured = shared_llm_structured
        logger.info("✓ 共享Agent就绪")
    tracker.complete_step("completed")

    # Step 4: 幻觉检测（支持并行和串行）
    tracker.start_step("批量幻觉检测")
    logger.info(f"\nStep 4: 开始幻觉检测 ({'并行' if parallel else '串行'}模式)...")
    if parallel:
        logger.info(f"并行线程数: {max_workers}")
    logger.info("=" * 80)

    records: List[SentenceDetectionRecord] = []
    hallucination_count = 0
    failed_count = 0
    failed_sentences = []
    redetection_failures = []

    if parallel:
        # 并行检测
        completed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(detect_sentence_wrapper, sentence_data, patient_id, llm_structured): sentence_data[0]
                for sentence_data in sentence_list
            }

            # 收集结果（按完成顺序）
            for future in as_completed(future_to_idx):
                idx, record, success = future.result()
                completed_count += 1

                # 刷新进度显示
                tracker.refresh()

                if success and record:
                    records.append(record)

                    # 显示检测结果
                    sentence_preview = record.original_sentence[:80] + ('...' if len(record.original_sentence) > 80 else '')

                    if record.detection_result.hallucination_status:
                        hallucination_count += 1
                        type_label = record.detection_result.hallucination_type.value if record.detection_result.hallucination_type is not None else '未提供'
                        logger.warning(f"[{completed_count}/{total_sentences}] 句子{idx}: ⚠️ 幻觉 - {type_label}")
                    else:
                        logger.info(f"[{completed_count}/{total_sentences}] 句子{idx}: ✓ 正常")
                else:
                    failed_count += 1
                    failed_sentences.append(record)
                    logger.error(f"[{completed_count}/{total_sentences}] 句子{idx}: ✗ 失败")

        # 按索引排序
        records.sort(key=lambda r: r.sentence_index)

    else:
        # 串行检测（原有逻辑）
        sentence_start_time = time.time()
        for idx, sentence in sentence_list:
            logger.info(f"\n[{idx+1}/{total_sentences}] 句子: {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
            logger.info("-" * 80)

            # 计算句子级别的进度
            completed_sentences = len(records)
            if completed_sentences > 0:
                sentence_elapsed = time.time() - sentence_start_time
                avg_time_per_sentence = sentence_elapsed / completed_sentences
                remaining_sentences = total_sentences - completed_sentences
                sentence_remaining = avg_time_per_sentence * remaining_sentences

                # 格式化时间并显示句子进度
                sentence_info = f"\033[36m句子进度: {completed_sentences}/{total_sentences}\033[0m, \033[33m已用: {format_time(sentence_elapsed)}\033[0m, \033[32m剩余: {format_time(sentence_remaining)}\033[0m"
                logger.info(sentence_info)

            try:
                # 检测句子
                result = detect_single_sentence(sentence, patient_id, llm_structured)

                # 构建记录
                record = SentenceDetectionRecord(
                    sentence_index=idx,
                    original_sentence=sentence,
                    detection_result=result['detection_result'],
                    graphrag_evidence=result['graphrag_evidence'],
                    distillation_status=result.get('distillation_status')
                )

                records.append(record)

                # 显示检测结果
                if result['detection_result'].hallucination_status:
                    hallucination_count += 1
                    logger.warning(f"⚠️  检测到幻觉")
                    type_label = result['detection_result'].hallucination_type.value if result['detection_result'].hallucination_type is not None else '未提供'
                    logger.warning(f"   类型: {type_label}")
                    logger.warning(f"   证据等级: {result['detection_result'].evidence_grade.value}")
                else:
                    logger.info(f"✓ 无幻觉 (证据等级: {result['detection_result'].evidence_grade.value})")

                # 显示推理过程摘要
                reasoning_preview = result['detection_result'].reasoning[:150]
                logger.info(f"   推理: {reasoning_preview}{'...' if len(result['detection_result'].reasoning) > 150 else ''}")

                # 显示批量处理进度（如果有的话）
                if batch_progress_info:
                    remaining_patients = batch_progress_info['total'] - batch_progress_info['current']
                    batch_remaining = batch_progress_info['avg_time'] * remaining_patients
                    patient_progress_str = f"\033[35m📊 患者进度: {batch_progress_info['current']}/{batch_progress_info['total']}\033[0m, \033[33m平均用时: {format_time(batch_progress_info['avg_time'])}/患者\033[0m, \033[32m预计剩余: {format_time(batch_remaining)}\033[0m"
                    logger.info(patient_progress_str)

            except Exception as e:
                failed_count += 1
                failed_sentences.append({
                    "sentence_index": idx,
                    "original_sentence": sentence,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e),
                })
                logger.error(f"❌ 检测失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # 继续处理下一个句子
                continue

    # 检查是否全部失败
    if len(records) == 0 and total_sentences > 0:
        error_msg = f"所有句子检测都失败！0/{total_sentences} 成功。请检查GraphRAG配置和LLM连接。"
        logger.error(error_msg)
        tracker.complete_step("failed")
        raise RuntimeError(error_msg)

    # 记录统计信息
    logger.info(f"\n检测完成: 成功 {len(records)}/{total_sentences}, 失败 {failed_count}")
    if failed_count > 0:
        logger.warning(f"警告: {failed_count} 个句子检测失败")

    tracker.complete_step("completed")

    # Step 5: 三层验证与自动修正 (在生成报告前执行)
    tracker.start_step("三层验证")
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: 三层验证与自动修正...")
    logger.info("=" * 80)

    # 导入验证模块
    from curaview._03_detection.validation.validation_logger import validate_report_and_generate_log, update_log_with_correction_results

    # 5.1 构造临时报告用于验证
    logger.info("\n[5.1] 构造检测结果...")

    # 总体幻觉统计
    hallucinated_indices = [
        rec.sentence_index for rec in records
        if rec.detection_result.hallucination_status
    ]

    # E3等级幻觉统计（无证据支持）
    e3_hallucinated_indices = [
        rec.sentence_index for rec in records
        if rec.detection_result.hallucination_status and rec.detection_result.evidence_grade.value == "E3"
    ]
    e3_hallucinations = len(e3_hallucinated_indices)
    e3_hallucination_rate = e3_hallucinations / len(records) if records else 0

    # E4等级幻觉统计（与证据冲突）
    e4_hallucinated_indices = [
        rec.sentence_index for rec in records
        if rec.detection_result.hallucination_status and rec.detection_result.evidence_grade.value == "E4"
    ]
    e4_hallucinations = len(e4_hallucinated_indices)
    e4_hallucination_rate = e4_hallucinations / len(records) if records else 0

    temp_report = DetectionReport(
        patient_id=patient_id,
        total_sentences=total_sentences,
        successful_sentences=len(records),
        failed_sentences=sorted(failed_sentences, key=lambda item: item['sentence_index']),
        processing_status='partial' if failed_count else 'completed',
        detected_hallucinations=hallucination_count,
        hallucination_rate=hallucination_count / len(records) if records else 0,
        hallucinated_indices=hallucinated_indices,
        e3_hallucinations=e3_hallucinations,
        e3_hallucination_rate=e3_hallucination_rate,
        e3_hallucinated_indices=e3_hallucinated_indices,
        e4_hallucinations=e4_hallucinations,
        e4_hallucination_rate=e4_hallucination_rate,
        e4_hallucinated_indices=e4_hallucinated_indices,
        records=records
    )

    # 5.2 执行三层验证
    logger.info("\n[5.2] 执行三层验证...")
    output_config = get_output_config()
    validation_log = validate_report_and_generate_log(
        report_data=temp_report.model_dump(),
        output_dir=output_config['validation_log_dir']
    )

    tracker.complete_step("completed")

    # 5.3 如果有一致性错误，自动重新检测
    if validation_log.consistency_errors_count > 0:
        tracker.start_step("自动修正不一致")
        logger.info("\n[5.3] 检测到不一致，执行自动重新检测...")
        logger.info(f"需要修正的句子数: {validation_log.consistency_errors_count}")

        # 提前提取不一致索引（供后续日志使用）
        inconsistent_indices = [err.sentence_index for err in validation_log.consistency_errors]

        # 检查是否启用并发重检
        parallel_redetection = CONFIG['detection'].get('parallel_redetection', False)

        if parallel_redetection:
            # 使用异步并发重检
            logger.info("使用异步并发重检模式")
            from curaview._03_detection.validation.rewriter_async import redetect_batch_async

            # 准备需要重检的记录
            inconsistent_records = []
            for err in validation_log.consistency_errors:
                idx = err.sentence_index
                for i, rec in enumerate(records):
                    if rec.sentence_index == idx:
                        inconsistent_records.append((i, rec))
                        break

            # 异步批量重检
            redetect_result = asyncio.run(redetect_batch_async(
                records_to_redetect=inconsistent_records,
                patient_id=patient_id,
                detect_func=detect_single_sentence,
                llm_structured=llm_structured,
                max_retries=3
            ))

            # 更新记录
            for position, new_result in redetect_result['updated_results'].items():
                records[position].detection_result = new_result
            updated_positions = set(redetect_result['updated_results'])
            for position, record in inconsistent_records:
                is_consistent, _ = DetectionValidator.validate_consistency_with_context(
                    record.detection_result.hallucination_status,
                    record.detection_result.evidence_grade.value,
                    record.detection_result.hallucination_type.value if record.detection_result.hallucination_type else None,
                    record.detection_result.reasoning
                )
                if position not in updated_positions or not is_consistent:
                    redetection_failures.append({
                        'sentence_index': record.sentence_index,
                        'status': 'invalid' if position in updated_positions else 'error',
                        'stage': 'redetection',
                    })

            successful_corrections = redetect_result['successful_count']
            redetect_attempts = redetect_result['total_attempts']

        else:
            # 使用原有串行重检
            logger.info("使用串行重检模式")
            from curaview._03_detection.validation.rewriter import redetect_inconsistent_result

            redetect_attempts = 0
            successful_corrections = 0

            for idx in inconsistent_indices:
                # 找到对应的记录
                record = None
                record_position = None
                for i, rec in enumerate(records):
                    if rec.sentence_index == idx:
                        record = rec
                        record_position = i
                        break

                if record is None:
                    logger.error(f"❌ 未找到句子 {idx} 的记录")
                    continue

                logger.info(f"\n重新检测句子 {idx}...")

                try:
                    redetected_result, attempts = redetect_inconsistent_result(
                        record,
                        patient_id,
                        detect_single_sentence,
                        llm_structured,
                        max_retries=3
                    )

                    # 更新记录
                    records[record_position].detection_result = redetected_result
                    redetect_attempts += attempts

                    # 验证是否成功
                    is_consistent, _ = DetectionValidator.validate_consistency_with_context(
                        redetected_result.hallucination_status,
                        redetected_result.evidence_grade.value,
                        redetected_result.hallucination_type.value if redetected_result.hallucination_type else None,
                        redetected_result.reasoning
                    )
                    if is_consistent:
                        successful_corrections += 1
                        logger.info(f"✅ 句子 {idx} 修正成功")
                    else:
                        logger.warning(f"⚠️  句子 {idx} 修正后仍不一致")
                        redetection_failures.append({'sentence_index': idx, 'status': 'invalid', 'stage': 'redetection'})

                except Exception as e:
                    logger.error(f"❌ 句子 {idx} 重新检测失败: {e}")
                    redetection_failures.append({
                        'sentence_index': idx, 'status': 'error', 'stage': 'redetection',
                        'error_type': type(e).__name__, 'error': str(e),
                    })

        # 重新统计幻觉数量
        hallucination_count = sum(
            1 for rec in records
            if rec.detection_result.hallucination_status
        )
        hallucinated_indices = [
            rec.sentence_index for rec in records
            if rec.detection_result.hallucination_status
        ]

        # 重新统计E3和E4等级幻觉
        e3_hallucinated_indices = [
            rec.sentence_index for rec in records
            if rec.detection_result.hallucination_status and rec.detection_result.evidence_grade.value == "E3"
        ]
        e3_hallucinations = len(e3_hallucinated_indices)
        e3_hallucination_rate = e3_hallucinations / len(records) if records else 0

        e4_hallucinated_indices = [
            rec.sentence_index for rec in records
            if rec.detection_result.hallucination_status and rec.detection_result.evidence_grade.value == "E4"
        ]
        e4_hallucinations = len(e4_hallucinated_indices)
        e4_hallucination_rate = e4_hallucinations / len(records) if records else 0

        # 更新临时报告
        temp_report.detected_hallucinations = hallucination_count
        temp_report.hallucination_rate = hallucination_count / len(records) if records else 0
        temp_report.hallucinated_indices = hallucinated_indices
        temp_report.e3_hallucinations = e3_hallucinations
        temp_report.e3_hallucination_rate = e3_hallucination_rate
        temp_report.e3_hallucinated_indices = e3_hallucinated_indices
        temp_report.e4_hallucinations = e4_hallucinations
        temp_report.e4_hallucination_rate = e4_hallucination_rate
        temp_report.e4_hallucinated_indices = e4_hallucinated_indices
        temp_report.records = records

        # 更新验证日志
        validation_log = update_log_with_correction_results(
            log=validation_log,
            corrected_report=temp_report.model_dump(),
            redetection_attempts=redetect_attempts,
            output_dir=output_config['validation_log_dir']
        )

        logger.info("\n" + "-" * 80)
        logger.info(f"修正统计: 需要修正 {len(inconsistent_indices)} 个，成功修正 {successful_corrections} 个")
        logger.info("-" * 80)
        tracker.complete_step("completed")
    else:
        logger.info("\n✅ 三层验证通过，无需修正")
        # 跳过修正步骤
        tracker.start_step("自动修正不一致")
        tracker.complete_step("skipped")

    records, invalid_sentences = partition_valid_records(records)
    hallucinated_indices = [rec.sentence_index for rec in records if rec.detection_result.hallucination_status]
    hallucination_count = len(hallucinated_indices)
    e3_hallucinated_indices = [rec.sentence_index for rec in records if rec.detection_result.hallucination_status and rec.detection_result.evidence_grade.value == 'E3']
    e4_hallucinated_indices = [rec.sentence_index for rec in records if rec.detection_result.hallucination_status and rec.detection_result.evidence_grade.value == 'E4']
    # Step 6: 生成并保存最终报告 (已经过验证和修正)
    tracker.start_step("生成并保存报告")
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: 生成并保存最终报告...")
    logger.info("=" * 80)

    # 使用修正后的数据生成最终报告（包含E3/E4统计）
    report = DetectionReport(
        patient_id=patient_id,
        total_sentences=total_sentences,
        successful_sentences=len(records),
        failed_sentences=sorted(failed_sentences, key=lambda item: item['sentence_index']),
        invalid_sentences=invalid_sentences,
        redetection_failures=redetection_failures,
        processing_status=('partial' if records else 'failed') if failed_count or redetection_failures or invalid_sentences else 'completed',
        detected_hallucinations=hallucination_count,
        hallucination_rate=hallucination_count / len(records) if records else None,
        hallucinated_indices=hallucinated_indices,
        e3_hallucinations=len(e3_hallucinated_indices),
        e3_hallucination_rate=len(e3_hallucinated_indices) / len(records) if records else None,
        e3_hallucinated_indices=e3_hallucinated_indices,
        e4_hallucinations=len(e4_hallucinated_indices),
        e4_hallucination_rate=len(e4_hallucinated_indices) / len(records) if records else None,
        e4_hallucinated_indices=e4_hallucinated_indices,
        records=records
    )

    # 保存最终报告
    if CONFIG['output']['save_to_file']:
        output_config = get_output_config()
        output_dir = output_config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{patient_id}_detection_report.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report.model_dump_json(indent=2, ensure_ascii=False))

        logger.info(f"✅ 最终报告已保存: {output_file}")
        logger.info(f"✅ 验证日志已保存: {validation_log.output_file_path}")

    tracker.complete_step("completed")

    # Step 7: 打印最终摘要
    tracker.start_step("打印摘要")
    logger.info("\n" + "=" * 80)
    logger.info("最终检测摘要")
    logger.info("=" * 80)
    logger.info(f"总句子数: {total_sentences}")
    logger.info(f"成功检测: {len(records)}")
    logger.info(f"检测失败: {failed_count}")
    logger.info(f"")
    logger.info(f"总体幻觉统计:")
    logger.info(f"  检测到幻觉: {hallucination_count} 个")
    logger.info(f"  幻觉比例: {format_rate(report.hallucination_rate)}")
    logger.info(f"  幻觉句子索引: {hallucinated_indices}")
    logger.info(f"")
    logger.info(f"E3等级幻觉统计 (无证据支持):")
    logger.info(f"  E3幻觉数量: {report.e3_hallucinations} 个")
    logger.info(f"  E3幻觉比例: {format_rate(report.e3_hallucination_rate)}")
    logger.info(f"  E3幻觉句子索引: {report.e3_hallucinated_indices}")
    logger.info(f"")
    logger.info(f"E4等级幻觉统计 (与证据冲突):")
    logger.info(f"  E4幻觉数量: {report.e4_hallucinations} 个")
    logger.info(f"  E4幻觉比例: {format_rate(report.e4_hallucination_rate)}")
    logger.info(f"  E4幻觉句子索引: {report.e4_hallucinated_indices}")

    # 按类型统计
    type_counts = {}
    for record in records:
        if record.detection_result.hallucination_status:
            halluc_type = record.detection_result.hallucination_type.value if record.detection_result.hallucination_type is not None else '未提供'
            type_counts[halluc_type] = type_counts.get(halluc_type, 0) + 1

    if type_counts:
        logger.info("\n幻觉类型分布:")
        for halluc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {halluc_type}: {count}")

    # 按证据等级统计
    evidence_counts = {}
    for record in records:
        grade = record.detection_result.evidence_grade.value
        evidence_counts[grade] = evidence_counts.get(grade, 0) + 1

    logger.info("\n证据等级分布:")
    for grade in ['E1', 'E2', 'E3', 'E4']:
        count = evidence_counts.get(grade, 0)
        logger.info(f"  {grade}: {count}")

    logger.info("=" * 80)
    tracker.complete_step("completed")

    # 完成并保存进度摘要
    tracker.finish()
    if CONFIG['output']['save_to_file']:
        output_config = get_output_config()
        progress_summary_file = os.path.join(
            output_config['output_dir'],
            f"{patient_id}_progress_summary.json"
        )
        tracker.save_summary(progress_summary_file)

    return report


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='GraphRAG幻觉检测Agent - 基于知识图谱的医学文本幻觉检测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式选择患者
  python detect.py

  # 检测指定文本文件
  python detect.py discharge.txt --patient-id patient_2

  # 查看详细日志
  python detect.py --verbose
        """
    )

    parser.add_argument(
        'text_file',
        nargs='?',
        help='待检测的文本文件路径 (可选,不指定则交互式选择患者)'
    )

    parser.add_argument(
        '--patient-id',
        default=None,
        help='患者ID (仅在指定text_file时使用)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )

    args = parser.parse_args()

    # 检查是否是后台模式（通过环境变量传递）
    background_mode = os.getenv('HD_BACKGROUND_MODE') == '1'
    selected_patients_env = os.getenv('HD_SELECTED_PATIENTS')

    # 设置日志级别
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # 检查环境变量（仅API模式需要）
    mode = CONFIG['llm'].get('mode', 'api')
    if mode == 'api':
        api_key_env = CONFIG['llm']['api']['api_key'].strip('${}')
        if not os.getenv(api_key_env):
            print(f"错误: 请设置环境变量 {api_key_env}")
            print(f"示例: export {api_key_env}='your_api_key'")
            sys.exit(1)

    try:
        # 模式1: 指定了文本文件 - 直接处理
        if args.text_file:
            report = process_text_file(args.text_file, args.patient_id)

            print("\n" + "=" * 80)
            print("✅ 检测完成!" if report.processing_status == 'completed' else "⚠️ 部分句子检测失败；见报告 failed_sentences")
            print("=" * 80)
            output_config = get_output_config()
            project_root = str(retained_path("."))
            log_path = os.path.join(project_root, "logs", "agent_logs")
            print(f"查看详细结果: {output_config['output_dir']}/{report.patient_id}_detection_report.json")
            print(f"查看日志: {log_path}/")

            # 清理LiteLLM异步客户端
            try:
                import litellm
                if hasattr(litellm, 'close_all_clients'):
                    litellm.close_all_clients()
            except Exception:
                pass

            sys.exit(0 if report.processing_status == 'completed' else 1)

        # 模式2: 交互式选择患者
        else:
            # 如果是后台模式，直接从环境变量读取患者列表
            if background_mode and selected_patients_env:
                selected_patients = selected_patients_env.split(',')
                print(f"\n📋 后台模式: 处理 {len(selected_patients)} 个患者")
                print(f"   患者列表: {', '.join([p.split('_')[1] for p in selected_patients])}")
            else:
                selected_patients = interactive_select_patients()

            if not selected_patients:
                print("\n未选择任何患者,退出")
                sys.exit(0)

            # 如果是后台模式，跳过运行方式询问
            if background_mode:
                print("\n后台运行模式已启用")
            else:
                # 询问运行方式
                print("\n" + "=" * 80)
                print("请选择运行方式:")
                print("  1. 前台运行（实时查看日志）")
                print("  2. 后台运行（输出到日志文件）")
                print("  3. 退出")
                print("=" * 80)

                run_mode = input("\n请选择 (1-3): ").strip()

                if run_mode == '3':
                    print("\n退出")
                    sys.exit(0)
                elif run_mode == '2':
                    # 后台运行
                    print("\n切换到后台运行模式...")

                    # 构建患者范围描述（用于日志命名）
                    patient_nums = [int(p.split('_')[1]) for p in selected_patients]
                    if len(patient_nums) == 1:
                        range_desc = str(patient_nums[0])
                    else:
                        range_desc = f"{min(patient_nums)}-{max(patient_nums)}"

                    # 生成日志文件（使用模型专属目录）
                    output_config = get_output_config()
                    model_name = output_config['model_name']
                    log_dir = retained_path("outputs") / model_name / 'logs'
                    log_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = time.strftime("%Y-%m-%d_%H-%M")
                    log_file = log_dir / f"{timestamp}__{range_desc}.log"

                    # 重启为后台进程，传递患者参数
                    cmd = [sys.executable, __file__]
                    # 将患者列表通过环境变量传递
                    env = os.environ.copy()
                    env['HD_SELECTED_PATIENTS'] = ','.join(selected_patients)
                    env['HD_BACKGROUND_MODE'] = '1'

                    print(f"📝 日志文件: {log_file}")
                    print(f"🚀 启动后台任务...")

                    with open(log_file, 'w') as f:
                        process = subprocess.Popen(
                            cmd,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            env=env
                        )

                    print(f"✅ 后台任务已启动 (PID: {process.pid})")
                    print(f"   患者范围: {range_desc}")
                    print(f"   查看日志: tail -f {log_file}")
                    print(f"   检查状态: python detect.py --status")
                    print(f"   终止任务: kill {process.pid}")
                    sys.exit(0)
                elif run_mode != '1':
                    print("\n无效选择，默认使用前台运行")

            print()

            # 批量处理 - 一次性加载模型,所有患者共享
            print("=" * 80)
            print(f"开始批量检测 {len(selected_patients)} 个患者")
            print("=" * 80)

            # 预加载模型 (所有患者共享,避免重复加载)
            print("\n🔧 预加载检测模型 (所有患者共享)...")
            try:
                agent, llm_structured = create_detection_agent()
                print("✅ 模型加载完成\n")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                sys.exit(1)

            # 预加载句子分割模型并批量分割所有患者的文本
            print("🔧 预加载句子分割模型并批量分割文本...")
            try:
                # 导入句子分割工具
                from curaview._03_detection.sentences.shared_splitter import ensure_model_loaded, split_into_sentences
                from curaview._03_detection.sentences.sentence_splitter import split_sentences

                # 加载句子分割模型
                model_loaded = ensure_model_loaded()
                if model_loaded:
                    print("✅ 句子分割模型加载成功 (SaT模型)")
                else:
                    print("⚠️  使用规则模式进行句子分割")

                # 批量分割所有患者的句子
                patient_sentences = {}
                total_sentences = 0

                for patient_id in selected_patients:
                    try:
                        text_file = get_patient_text_file(patient_id)
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read()
                        sentences = split_sentences.invoke({"text": text})
                        patient_sentences[patient_id] = sentences
                        total_sentences += len(sentences)
                    except Exception as e:
                        logger.warning(f"⚠️  {patient_id} 文本读取失败: {e}")
                        patient_sentences[patient_id] = []

                print(f"✅ 批量分割完成: {len(selected_patients)}个患者, 共{total_sentences}个句子\n")
            except Exception as e:
                logger.warning(f"⚠️  批量分割失败,将使用逐个分割模式: {e}")
                patient_sentences = {}

            success_count = 0
            failed_patients = []
            batch_start_time = time.time()

            for idx, patient_id in enumerate(selected_patients, 1):
                # 计算患者级别的进度
                batch_progress_info = None
                if idx > 1:
                    batch_elapsed = time.time() - batch_start_time
                    avg_time_per_patient = batch_elapsed / (idx - 1)
                    remaining_patients = len(selected_patients) - (idx - 1)
                    batch_remaining = avg_time_per_patient * remaining_patients
                    patient_progress_info = f"\033[35m患者进度: {idx-1}/{len(selected_patients)}\033[0m, \033[33m平均用时: {format_time(avg_time_per_patient)}/患者\033[0m, \033[32m预计剩余: {format_time(batch_remaining)}\033[0m"
                    print(f"\n📊 {patient_progress_info}")

                    # 构造进度信息字典
                    batch_progress_info = {
                        'current': idx - 1,
                        'total': len(selected_patients),
                        'elapsed': batch_elapsed,
                        'avg_time': avg_time_per_patient
                    }

                print(f"\n[{idx}/{len(selected_patients)}] 正在处理: {patient_id}")
                print("-" * 80)

                # 获取文本文件路径（支持根目录和模型子目录）
                try:
                    text_file = get_patient_text_file(patient_id)
                except FileNotFoundError as e:
                    logger.error(f"❌ {patient_id} 检测失败: {e}")
                    failed_patients.append((patient_id, str(e)))
                    continue

                try:
                    # 传入共享的agent、llm_structured和预分割的句子
                    report = process_text_file(
                        text_file,
                        patient_id,
                        shared_agent=agent,
                        shared_llm_structured=llm_structured,
                        precomputed_sentences=patient_sentences.get(patient_id),
                        batch_progress_info=batch_progress_info
                    )
                    if report.processing_status == 'completed':
                        success_count += 1
                        print(f"✓ {patient_id} 检测完成 (幻觉率: {report.hallucination_rate:.2%})")
                    else:
                        failed_patients.append((patient_id, '部分句子检测失败；见报告 failed_sentences'))
                        print(f"⚠️ {patient_id} 部分完成；有效检测 {report.successful_sentences}/{report.total_sentences}")

                    # 每个患者处理完后都清理未关闭的连接
                    try:
                        import aiohttp
                        closed = 0
                        for obj in gc.get_objects():
                            if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
                                try:
                                    temp_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(temp_loop)
                                    temp_loop.run_until_complete(obj.close())
                                    temp_loop.close()
                                    closed += 1
                                except:
                                    pass
                        if closed > 0:
                            logger.info(f"🧹 患者完成后清理: 关闭了 {closed} 个aiohttp会话")
                    except:
                        pass

                    gc.collect()

                except Exception as e:
                    failed_patients.append((patient_id, str(e)))
                    logger.error(f"❌ {patient_id} 检测失败: {e}")
                    gc.collect()
                    continue

            # 显示批量处理摘要
            print("\n" + "=" * 80)
            print("批量检测完成")
            print("=" * 80)
            print(f"总患者数: {len(selected_patients)}")
            print(f"成功: {success_count}")
            print(f"失败: {len(failed_patients)}")

            if failed_patients:
                print("\n失败患者:")
                for patient_id, error in failed_patients:
                    print(f"  - {patient_id}: {error}")

            output_config = get_output_config()
            project_root = str(retained_path("."))
            log_path = os.path.join(project_root, "logs", "agent_logs")
            print(f"\n查看结果: {output_config['output_dir']}/")
            print(f"查看日志: {log_path}/")

            # 清理所有异步资源
            cleanup_async_resources(llm_structured)

            sys.exit(0 if not failed_patients else 1)

    except KeyboardInterrupt:
        print("\n\n操作已取消")
        # 尝试清理，但可能 llm_structured 未定义
        try:
            cleanup_async_resources(llm_structured)
        except NameError:
            cleanup_async_resources()
        sys.exit(1)

    except Exception as e:
        logger.error(f"程序异常终止: {e}")
        import traceback
        traceback.print_exc()
        # 尝试清理，但可能 llm_structured 未定义
        try:
            cleanup_async_resources(llm_structured)
        except NameError:
            cleanup_async_resources()
        sys.exit(1)


def cleanup_async_resources(llm=None):
    """清理所有异步HTTP资源

    Args:
        llm: ChatOpenAI 实例，用于关闭其 HTTP 客户端
    """
    try:
        # 1. 关闭 Agent 的 HTTP 客户端
        try:
            from curaview._03_detection.inference.agent import cleanup_agent_resources
            cleanup_agent_resources()
        except Exception as e:
            logger.debug(f"清理Agent资源失败: {e}")

        # 2. 清理 LiteLLM 异步客户端（正确处理所有协程）
        try:
            import litellm
            # 使用正确的方式关闭LiteLLM异步资源
            async def close_litellm_clients():
                """异步关闭所有LiteLLM客户端"""
                closed_items = []

                # 关闭模块级异步客户端
                if hasattr(litellm, 'module_level_aclient') and litellm.module_level_aclient:
                    try:
                        await litellm.module_level_aclient.aclose()
                        closed_items.append('module_level_aclient')
                    except Exception as e:
                        logger.debug(f"关闭module_level_aclient失败: {e}")

                # 关闭HTTP客户端（如果存在）
                if hasattr(litellm, 'client_session') and litellm.client_session:
                    try:
                        await litellm.client_session.close()
                        closed_items.append('client_session')
                    except Exception as e:
                        logger.debug(f"关闭client_session失败: {e}")

                # 等待所有待处理的任务完成
                await asyncio.sleep(0.1)
                return closed_items

            # 创建新的事件循环并运行清理
            try:
                # 获取或创建事件循环
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Loop is closed")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # 运行异步清理
                closed = loop.run_until_complete(close_litellm_clients())
                if closed:
                    logger.info(f"✅ LiteLLM客户端已关闭: {', '.join(closed)}")

                # 确保所有待处理任务完成
                loop.run_until_complete(asyncio.sleep(0))

            except Exception as loop_err:
                logger.debug(f"LiteLLM清理事件循环异常: {loop_err}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"清理LiteLLM失败: {e}")

        # 3. 清理 aiohttp 客户端会话
        try:
            import aiohttp
            # 获取所有未关闭的connector
            closed_count = 0
            for obj in gc.get_objects():
                if isinstance(obj, aiohttp.ClientSession):
                    try:
                        if not obj.closed:
                            # 创建临时事件循环关闭
                            temp_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(temp_loop)
                            try:
                                temp_loop.run_until_complete(obj.close())
                                closed_count += 1
                            finally:
                                temp_loop.close()
                    except:
                        pass
            if closed_count > 0:
                logger.info(f"✅ 关闭了 {closed_count} 个未关闭的aiohttp会话")
        except ImportError:
            pass

        # 4. 强制垃圾回收
        gc.collect()

        # 5. 清理未完成的异步任务（如果还有事件循环）
        try:
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                # 取消所有待处理任务
                try:
                    # 获取所有待处理任务
                    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]

                    if pending:
                        logger.debug(f"发现 {len(pending)} 个待处理的异步任务，正在取消...")

                        # 取消所有任务
                        for task in pending:
                            task.cancel()

                        # 等待所有取消完成，捕获CancelledError
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                    # 给事件循环额外时间完成清理回调
                    loop.run_until_complete(asyncio.sleep(0.1))

                    # 关闭事件循环
                    loop.close()
                    logger.debug("✅ 事件循环已关闭")

                except Exception as task_err:
                    logger.debug(f"清理异步任务异常: {task_err}")
        except RuntimeError as e:
            # 可能没有事件循环或已关闭
            logger.debug(f"获取事件循环失败: {e}")

        # 6. 再次垃圾回收确保清理
        gc.collect()

    except Exception as e:
        # 清理失败不影响程序退出
        logger.debug(f"资源清理异常: {e}")


def run_background(args):
    """后台运行检测任务"""
    cmd = [sys.executable, __file__]

    available = find_available_patients()
    if args.all:
        selected = available
    elif args.start is not None and args.end is not None:
        selected = [p for p in available if args.start <= int(p.removeprefix('patient_')) <= args.end]
    elif args.patient_ids:
        selected = [p if p.startswith('patient_') else f'patient_{p}' for p in args.patient_ids]
        if not set(selected) <= set(available):
            raise ValueError('Selected IDs must have both text and a GraphRAG index')
    else:
        raise ValueError('Use --background with --all, --start/--end or --patient-ids')
    if not selected:
        raise ValueError('No selected patients have both text and a GraphRAG index')
    env = {**os.environ, 'HD_BACKGROUND_MODE': '1', 'HD_SELECTED_PATIENTS': ','.join(selected)}

    # 设置日志文件（使用模型专属目录）
    output_config = get_output_config()
    model_name = output_config['model_name']
    log_dir = retained_path("outputs") / model_name / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # 构建患者范围描述
    if args.all:
        range_desc = "all"
    elif args.start and args.end:
        range_desc = f"{args.start}-{args.end}"
    elif args.patient_ids:
        patient_nums = [int(p.split('_')[1]) if '_' in p else int(p) for p in args.patient_ids]
        range_desc = f"{min(patient_nums)}-{max(patient_nums)}"
    else:
        range_desc = "unknown"

    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    log_file = log_dir / f"{timestamp}__{range_desc}.log"

    print(f"📝 日志文件: {log_file}")
    print(f"🚀 启动后台任务...")

    # 启动后台进程
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    print(f"✅ 后台任务已启动 (PID: {process.pid})")
    print(f"   查看日志: tail -f {log_file}")
    print(f"   检查状态: python detect.py --status")


def show_status():
    """显示检测任务状态"""
    try:
        import psutil
    except ImportError:
        print("错误: 需要安装 psutil 模块")
        print("运行: pip install psutil")
        sys.exit(1)

    print("=" * 70)
    print("  幻觉检测任务状态")
    print("=" * 70)
    print()

    # 查找正在运行的检测进程
    found = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'detect.py' in ' '.join(cmdline) and proc.pid != os.getpid():
                found = True
                print(f"📊 进程 ID: {proc.info['pid']}")
                print(f"   CPU: {proc.info['cpu_percent']:.1f}%")
                print(f"   内存: {proc.info['memory_percent']:.1f}%")
                print()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not found:
        print("  ✗ 没有正在运行的检测任务")
        print()

    # 显示最近的日志
    print("=" * 70)
    print("📁 最近的日志文件:")
    print()

    # 获取当前模型的日志目录
    output_config = get_output_config()
    model_name = output_config['model_name']
    log_dir = retained_path("outputs") / model_name / 'logs'

    if log_dir.exists():
        logs = sorted(log_dir.glob('*.log'), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        for log in logs:
            size = log.stat().st_size / 1024  # KB
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(log.stat().st_mtime))
            print(f"  • {log.name}")
            print(f"    大小: {size:.1f} KB")
            print(f"    修改时间: {mtime}")
            print()
    else:
        print(f"  ✗ 日志目录不存在: {log_dir}")
        print()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GraphRAG幻觉检测系统', add_help=False)
    parser.add_argument('--background', '-bg', action='store_true', help='后台运行')
    parser.add_argument('--status', '-s', action='store_true', help='查看运行状态')
    parser.add_argument('--all', action='store_true', help='处理所有患者')
    parser.add_argument('--start', type=int, help='起始患者ID')
    parser.add_argument('--end', type=int, help='结束患者ID')
    parser.add_argument('--patient-ids', nargs='+', help='指定患者ID列表')

    args, remaining = parser.parse_known_args()

    # 处理特殊命令
    if args.status:
        show_status()
        sys.exit(0)

    if args.background:
        run_background(args)
        sys.exit(0)

    # 正常运行模式
    print("=" * 80)
    print("GraphRAG幻觉检测系统")
    print("=" * 80)
    print()

    # 运行主程序
    sys.argv[1:] = remaining
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户取消操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
