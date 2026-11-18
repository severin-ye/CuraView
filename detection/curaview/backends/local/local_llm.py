"""
本地LLM封装类
支持使用本地Qwen3模型进行推理，采用分步生成策略避免JSON格式问题
"""

import os
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置详细调试日志
DEBUG_LOG_DIR = Path("experiment/幻觉检测_qwen/output/排错日志")
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)

# 加载本地提示词
from curaview.common.source_layout import retained_path
PROMPT_DIR = retained_path("doc/prompts")
LOCAL_PROMPT_FILE = PROMPT_DIR / "detection_prompt_local_stepwise.md"

def load_local_prompt() -> str:
    """加载本地分步生成提示词"""
    text = LOCAL_PROMPT_FILE.read_text(encoding='utf-8')
    if not text.strip():
        raise ValueError('configured local stepwise prompt is empty')
    return text

def get_debug_logger(name: str) -> logging.Logger:
    """获取调试日志记录器（每次运行创建新日志文件）"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # 文件处理器 - 详细日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = DEBUG_LOG_DIR / f"debug_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        # 强制刷新，确保日志及时写入文件
        file_handler.setFormatter(logging.Formatter('%(asctime)s - [线程%(thread)d] - %(message)s'))
        logger.addHandler(file_handler)
        logger.propagate = False  # 避免重复输出

        logger.info(f"=" * 80)
        logger.info(f"调试日志开始 - {timestamp}")
        logger.info(f"=" * 80)

        # 强制刷新
        for handler in logger.handlers:
            handler.flush()

    return logger

# 创建全局调试logger
debug_logger = get_debug_logger("local_llm_debug")
from pydantic import BaseModel


class LocalLLM:
    """本地大语言模型封装类，兼容langchain接口"""

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False
    ):
        """
        初始化本地模型

        Args:
            model_path: 模型路径
            temperature: 采样温度
            max_tokens: 最大生成token数
            device_map: 设备映射策略
            torch_dtype: 数据类型
            trust_remote_code: 是否信任远程代码
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        print(f"🔄 正在加载本地模型: {model_path}")

        # 转换torch_dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype_obj,
            device_map=device_map,
            trust_remote_code=trust_remote_code
        )

        print(f"✅ 模型加载成功")

        # 显示设备信息
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"💾 GPU显存占用: {allocated:.2f}GB")

    def invoke(self, messages: List[Dict[str, str]]) -> Any:
        """
        调用模型生成回复（兼容langchain接口）

        Args:
            messages: 消息列表，格式 [{"role": "system/user/assistant", "content": "..."}]

        Returns:
            生成的文本内容
        """
        # **线程安全：使用临时temperature参数，不修改self.temperature**
        # 检查是否在结构化输出模式（通过消息内容判断）
        use_low_temp = any("JSON" in str(msg.get("content", "")) for msg in messages)
        current_temp = 0.1 if use_low_temp else self.temperature

        from curaview._99_tracing.capture.detection_distillation import get_trace_store
        store = getattr(self, 'trace_store', None) or get_trace_store()
        parameters = dict(max_new_tokens=self.max_tokens, temperature=current_temp,
                          do_sample=current_temp > 0, pad_token_id=self.tokenizer.eos_token_id,
                          eos_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.1,
                          no_repeat_ngram_size=3)
        # Each step opens its own call BEFORE rendering/tokenization can fail.
        with store.call('legacy.stepwise.generate', {'messages': messages,
                        'generation_parameters': parameters}) as call:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            call.record('rendered_input', {'text': text})
            inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
            call.record('tokenized', {key: value.tolist() for key, value in inputs.items()})
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **parameters)
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            call.record('generated', {'output_tensor_ids': outputs.tolist(),
                                     'generated_token_ids': generated_ids.tolist()})
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
            call.record('decoded', {'raw_generated_text': response_text})
            call.finish({'input_width': input_length})

        # 返回类似langchain的响应对象
        class SimpleResponse:
            def __init__(self, content):
                self.content = content

        return SimpleResponse(response_text)

    def with_structured_output(self, schema: type[BaseModel]) -> 'StructuredLocalLLM':
        """
        配置结构化输出（兼容langchain接口）

        Args:
            schema: Pydantic模型类

        Returns:
            配置了结构化输出的LLM实例
        """
        return StructuredLocalLLM(self, schema)


class StructuredLocalLLM:
    """支持结构化输出的本地LLM"""

    def __init__(self, base_llm: LocalLLM, schema: type[BaseModel]):
        """
        初始化结构化输出LLM

        Args:
            base_llm: 基础LLM实例
            schema: 输出的Pydantic模型类
        """
        self.base_llm = base_llm
        self.schema = schema

    def invoke(self, messages: List[Dict[str, str]]) -> BaseModel:
        """
        调用模型并返回结构化输出（分步生成策略）

        Args:
            messages: 消息列表

        Returns:
            解析后的Pydantic模型实例
        """
        # **新策略：分步生成，避免JSON格式错误**
        return self._invoke_step_by_step(messages)

    def _invoke_step_by_step(self, messages: List[Dict[str, str]]) -> BaseModel:
        """
        分步生成策略：一步一步生成每个字段，避免JSON格式错误

        步骤：
        1. 生成 reasoning（推理依据）
        2. 生成 hallucination_status（是否有幻觉）
        3. 生成 hallucination_type（幻觉类型）
        4. 生成 evidence_grade（证据等级）
        5. 组装成完整结果

        Args:
            messages: 原始消息列表

        Returns:
            完整的Pydantic模型实例
        """
        import re

        thread_id = id(messages) % 10000
        debug_logger.info(f"=" * 60)
        debug_logger.info(f"[线程{thread_id}] 🔄 开始分步生成")

        for handler in debug_logger.handlers:
            handler.flush()

        original_sentence = ""
        graphrag_evidence = ""
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            evidence_marker = next((marker for marker in (
                "GraphRAG证据:", "GraphRAG检索到的相关证据:"
            ) if marker in content), None)
            if "句子:" in content:
                sentence_part = content.split("句子:", 1)[1]
                original_sentence = (sentence_part.split(evidence_marker, 1)[0]
                                     if evidence_marker else sentence_part).strip()
                if original_sentence.startswith('"') and original_sentence.endswith('"'):
                    original_sentence = original_sentence[1:-1]
            if evidence_marker:
                graphrag_evidence = content.split(evidence_marker, 1)[1].split(
                    "\n\n请基于以上证据", 1)[0].strip()

        debug_logger.info(f"[线程{thread_id}] 原始句子: {original_sentence[:50]}...")
        debug_logger.info(f"[线程{thread_id}] 证据长度: {len(graphrag_evidence)} 字符")

        system_prompt = load_local_prompt()

        step1_prompt = (
            "**检测任务**\n\n"
            f"**句子:** {original_sentence}\n\n"
            f"**GraphRAG证据:**\n{graphrag_evidence}\n\n"
            "请用50字以内概括句子与证据的关键比对结果，不要输出JSON。"
        )
        step1_messages = [
            {"role": "system", "content": system_prompt + "\n\n你现在处于Step 1: 生成推理过程。请用简短中文回答，不要输出JSON。"},
            {"role": "user", "content": step1_prompt},
        ]
        reasoning_response = self.base_llm.invoke(step1_messages)
        reasoning = reasoning_response.content.strip().replace("```json", "").replace("```", "")
        if reasoning.startswith("{"):
            match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', reasoning)
            if match:
                reasoning = match.group(1)
        reasoning = reasoning[:150]

        debug_logger.info(f"[线程{thread_id}] ✅ Step 1 - reasoning: {reasoning}")
        print(f"📝 [{thread_id}] Step 1/4: Reasoning完成")

        step2_prompt = (
            "**判断任务**\n\n"
            f"**推理:** {reasoning}\n\n"
            "如果存在医学幻觉，只输出 true；如果不存在医学幻觉，只输出 false。"
        )
        step2_messages = [
            {"role": "system", "content": system_prompt + "\n\n你现在处于Step 2: 判断是否存在幻觉。只输出true或false。"},
            {"role": "user", "content": step2_prompt},
        ]
        status_response = self.base_llm.invoke(step2_messages)
        status_text = status_response.content.strip().replace("```", "").replace("json", "")
        if status_text.lower() not in {"true", "false"}:
            raise ValueError("Step 2 did not return exactly true or false")
        hallucination_status = status_text.lower() == "true"

        debug_logger.info(f"[线程{thread_id}] ✅ Step 2 - status: {hallucination_status} (原文: {status_text[:30]})")
        print(f"🔍 [{thread_id}] Step 2/4: Status={hallucination_status}")

        if hallucination_status:
            step3_prompt = (
                "**分类任务**\n\n"
                f"**推理:** {reasoning}\n\n"
                "请只输出一个主标签：value_error、time_error、negation_error、medication_error、diagnosis_error、exam_result_error、invented_fact。\n"
                "优先级：value_error > time_error > negation_error > 领域标签 > invented_fact。"
            )
            step3_messages = [
                {"role": "system", "content": system_prompt + "\n\n你现在处于Step 3: 判断幻觉类型。只输出一个类型名，不要解释。"},
                {"role": "user", "content": step3_prompt},
            ]
            type_response = self.base_llm.invoke(step3_messages)
            type_text = type_response.content.strip().lower().replace("```", "").replace("json", "")

            valid_types = [
                "value_error",
                "time_error",
                "negation_error",
                "medication_error",
                "diagnosis_error",
                "exam_result_error",
                "invented_fact",
            ]
            hallucination_type = type_text if type_text in valid_types else None
            if hallucination_type is None:
                if type_text in {"否定", "场景"}:
                    hallucination_type = "negation_error"
                elif type_text == "数值":
                    hallucination_type = "value_error"
                elif type_text == "时间":
                    hallucination_type = "time_error"
                elif type_text in {"药物", "用药"}:
                    hallucination_type = "medication_error"
                elif type_text == "诊断":
                    hallucination_type = "diagnosis_error"
                elif type_text == "检查":
                    hallucination_type = "exam_result_error"
                else:
                    raise ValueError("Step 3 did not return one recognized hallucination type")

            debug_logger.info(f"[线程{thread_id}] ✅ Step 3 - hallucination_type: {hallucination_type} (原文: {type_response.content.strip()[:30]})")
            print(f"🏷️  [{thread_id}] Step 3/4: Type={hallucination_type}")
        else:
            hallucination_type = None
            debug_logger.info(f"[线程{thread_id}] ⊘ Step 3 - 无幻觉，跳过类型判断")
            print(f"⊘ [{thread_id}] Step 3/4: Type=null")

        step4_prompt = (
            "评估以下错误片段的证据等级。\n\n"
            f"**证据片段:**\n{graphrag_evidence[:300]}...\n\n"
            "标准：E1=患者结构化或非结构化证据直接明确支持；E2=需从患者证据间接推断；E3=检索证据不支持且不矛盾，不限新增片段；E4=与患者事实直接矛盾，不表示危害程度。优先级：E4 > E1 > E2 > E3。E1/E2对应无幻觉且类型为null，E3/E4对应有幻觉。\n"
            "只输出E1/E2/E3/E4。"
        )
        step4_messages = [
            {"role": "system", "content": "你是证据等级评估专家。只输出E1/E2/E3/E4，不要解释。"},
            {"role": "user", "content": step4_prompt},
        ]
        grade_response = self.base_llm.invoke(step4_messages)
        evidence_grade_text = grade_response.content.strip().upper()
        match = re.fullmatch(r"E[1-4]", evidence_grade_text)
        if match is None:
            raise ValueError("Step 4 did not return exactly one evidence grade")
        evidence_grade = match.group(0)

        debug_logger.info(f"[线程{thread_id}] ✅ Step 4 - evidence_grade: {evidence_grade}")
        print(f"📊 [{thread_id}] Step 4/4: Grade={evidence_grade}")

        result_data = {
            "reasoning": reasoning,
            "hallucination_status": hallucination_status,
            "hallucination_type": hallucination_type,
            "evidence_grade": evidence_grade,
        }

        debug_logger.info(f"[线程{thread_id}] 🎯 分步生成完成")
        debug_logger.debug(f"[线程{thread_id}] 最终数据: {result_data}")
        print(f"✅ [{thread_id}] 分步生成完成")

        for handler in debug_logger.handlers:
            handler.flush()

        try:
            return self.schema(**result_data)
        except Exception as e:
            debug_logger.error(f"[线程{thread_id}] Pydantic验证失败: {e}")
            debug_logger.error(f"[线程{thread_id}] 数据: {result_data}")
            raise

    def _invoke_json_mode(self, messages: List[Dict[str, str]]) -> BaseModel:
        """
        原始的JSON模式（已废弃，保留备用）

        Args:
            messages: 消息列表

        Returns:
            解析后的Pydantic模型实例
        """
        # 获取schema定义
        schema_json = self.schema.model_json_schema()

        # **关键修复：构建更严格的JSON输出提示词**
        json_schema_str = json.dumps(schema_json, indent=2, ensure_ascii=False)

        # 获取必需字段列表
        required_fields = schema_json.get("required", [])
        field_descriptions = []
        for field_name, field_info in schema_json.get("properties", {}).items():
            field_type = field_info.get("type", "string")
            field_desc = field_info.get("description", "")
            required_mark = " (必填)" if field_name in required_fields else ""
            field_descriptions.append(f"  - {field_name}: {field_type}{required_mark} - {field_desc}")

        # 构建严格的JSON指令
        json_instruction = f"""

你必须以纯JSON格式输出，严格遵循以下要求：

**JSON Schema:**
{json_schema_str}

**字段说明:**
{chr(10).join(field_descriptions)}

**输出规则（必须严格遵守）：**
1. 只输出一个有效的JSON对象，不要任何额外文字
2. 禁止使用<think>、<reasoning>等XML标签包裹内容
3. 不要用markdown代码块（```json）包裹
4. **reasoning字段必须控制在50字以内**（超长会导致JSON截断）
5. 使用英文标点符号(逗号,冒号:)，不要用中文标点（，：）
6. 布尔值必须小写(true/false)，不是Python的True/False
7. null必须小写，不是None
8. 字段名必须精确匹配（全部小写+下划线）

**字段名拼写（必须精确，不要拼错）：**
- ✅ "hallucination_status" ❌ hallucination status, Hallucination_Status
- ✅ "hallucination_type" ❌ halucination_type, Hallucination_Type
- ✅ "evidence_grade" ❌ evidence_grage, Evidence_Grade, hvidence_grade
- ✅ "reasoning" ❌ Reasoning, reasioning

**Few-shot示例（严格模仿这个格式和长度）：**

示例1（无幻觉，reasoning仅28字）：
{{
  "reasoning": "证据显示患者腹水蛋白1.2g/dL，符合SBP预防标准",
  "hallucination_status": false,
  "hallucination_type": null,
  "evidence_grade": "E1"
}}

示例2（有幻觉，reasoning仅31字）：
{{
  "reasoning": "证据显示PMN仅15.7/mm³，远低于SBP诊断阈值250",
  "hallucination_status": true,
  "hallucination_type": "exam_result_error",
  "evidence_grade": "E1"
}}

**关键提醒：reasoning必须≤50字，超过会导致JSON截断！**

现在开始分析，直接输出JSON：
"""

        # 在system消息中添加JSON格式要求
        enhanced_messages = []
        for msg in messages:
            if msg["role"] == "system":
                enhanced_content = msg["content"] + json_instruction
                enhanced_messages.append({"role": "system", "content": enhanced_content})
            else:
                enhanced_messages.append(msg)

        # **线程安全：不修改实例属性，LocalLLM.invoke会根据消息内容自动使用0.1温度**
        try:
            response = self.base_llm.invoke(enhanced_messages)
            response_text = response.content.strip()

            thread_id = id(response) % 10000

            # 详细日志写入文件
            debug_logger.info(f"=" * 60)
            debug_logger.info(f"[线程{thread_id}] 开始处理")
            debug_logger.info(f"[线程{thread_id}] 原始输出长度: {len(response_text)} 字符")
            debug_logger.debug(f"[线程{thread_id}] 原始输出内容:\n{response_text}")

            # 强制刷新日志
            for handler in debug_logger.handlers:
                handler.flush()

            # 简化的终端输出
            print(f"🔍 [{thread_id}] 原始:{len(response_text)}字符")

            # 提取JSON
            json_text = self._extract_json(response_text)
            debug_logger.info(f"[线程{thread_id}] 提取JSON后长度: {len(json_text)} 字符")
            debug_logger.debug(f"[线程{thread_id}] 提取的JSON:\n{json_text}")

            # **关键修复：激进的JSON修复**
            json_text_before = json_text
            json_text = self._aggressive_json_repair(json_text)
            if json_text != json_text_before:
                debug_logger.info(f"[线程{thread_id}] JSON已修复")
                debug_logger.debug(f"[线程{thread_id}] 修复前:\n{json_text_before}")
                debug_logger.debug(f"[线程{thread_id}] 修复后:\n{json_text}")
                print(f"🔧 [{thread_id}] 已修复")

            # 清理字段名拼写错误
            json_text = self._clean_json_text(json_text)
            debug_logger.info(f"[线程{thread_id}] 清理后JSON长度: {len(json_text)} 字符")
            debug_logger.debug(f"[线程{thread_id}] 清理后JSON:\n{json_text}")

            # 解析并验证
            data = json.loads(json_text)
            debug_logger.info(f"[线程{thread_id}] JSON解析成功，字段: {list(data.keys())}")

            # Missing model fields remain missing; schema failures must not
            # manufacture an evidence grade or a reasoning statement.

            print(f"✅ [{thread_id}] 解析成功")

            result = self.schema(**data)
            debug_logger.info(f"[线程{thread_id}] Pydantic验证成功")

            # 强制刷新日志
            for handler in debug_logger.handlers:
                handler.flush()

            return result

        except json.JSONDecodeError as e:
            thread_id = id(e) % 10000

            # 详细错误日志
            debug_logger.error(f"[线程{thread_id}] JSON解析失败: {e}")
            debug_logger.error(f"[线程{thread_id}] 原始输出:\n{response_text}")
            debug_logger.error(f"[线程{thread_id}] 提取的JSON:\n{json_text}")

            # 简化终端输出
            print(f"❌ [{thread_id}] 解析失败: {str(e)[:50]}")

            # **尝试使用json-repair自动修复**
            try:
                import json_repair
                debug_logger.info(f"[线程{thread_id}] 尝试使用json-repair修复...")
                repaired_json = json_repair.repair_json(json_text)
                debug_logger.debug(f"[线程{thread_id}] json-repair修复结果:\n{repaired_json}")

                # 修复后再次清理字段名
                repaired_json = self._clean_json_text(repaired_json)

                data = json.loads(repaired_json)
                result = self.schema(**data)
                debug_logger.info(f"[线程{thread_id}] json-repair修复成功")
                print(f"✅ [{thread_id}] 修复后成功")
                return result
            except Exception as repair_error:
                debug_logger.error(f"[线程{thread_id}] json-repair修复失败: {repair_error}")
                print(f"❌ [{thread_id}] 修复失败")
                raise ValueError(f"模型输出不是有效的JSON格式: {e}")

        except ValueError as e:
            thread_id = id(e) % 10000
            debug_logger.error(f"[线程{thread_id}] Pydantic验证失败: {e}")
            if 'data' in locals():
                debug_logger.error(f"[线程{thread_id}] 解析的数据: {data}")
            print(f"❌ [{thread_id}] 验证失败")

            # 强制刷新日志
            for handler in debug_logger.handlers:
                handler.flush()

            raise
        # 注意：移除了finally块中的temperature恢复（已改为线程安全方式）

    def _aggressive_json_repair(self, json_text: str) -> str:
        """
        激进的JSON修复：处理常见语法错误

        Args:
            json_text: 待修复的JSON文本

        Returns:
            修复后的JSON文本
        """
        import re

        # **步骤0：最优先 - 统一引号格式（核心修复！）**
        # 先把所有单引号字段名和值转成双引号
        # 'reasoning': 'value' → "reasoning": "value"
        # 注意顺序很重要：先处理字段名，再处理值

        # 0.1 转换字段名的单引号：'field_name': → "field_name":
        json_text = re.sub(r"'(\w+)'\s*:", r'"\1":', json_text)

        # 0.2 **修复缺少冒号的字段（最常见）**
        # "hallucination_typenull → "hallucination_type":null
        # "evidence_gradeE1" → "evidence_grade":"E1"
        json_text = re.sub(r'"(hallucination_type|evidence_grade)"(\w)', r'"\1":"\2', json_text)

        # 0.2 转换字段值的单引号：: 'value' → : "value"
        # 但要小心reasoning字段内可能有合法的单引号
        json_text = re.sub(r":\s*'([^']*)'", r': "\1"', json_text)

        # **步骤1：检测未闭合的JSON（截断问题）**
        if json_text.count('{') > json_text.count('}'):
            debug_logger.warning(f"检测到未闭合JSON，左括号{json_text.count('{')}，右括号{json_text.count('}')}")
            last_quote = json_text.rfind('"')
            if last_quote > 0:
                json_text = json_text[:last_quote] + '"}}'
                debug_logger.info(f"已补全JSON结尾")

        # **步骤2：修复缺失的逗号**
        json_text = re.sub(r'"\s*\n\s*"', '",\n"', json_text)
        json_text = re.sub(r'"\s+"', '", "', json_text)  # 两个字段名之间缺逗号

        # **步骤3：修复末尾多余逗号**
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)

        # **步骤4：修复字段名后缺失冒号**
        json_text = re.sub(r'"(\w+)"\s+"', r'"\1": "', json_text)

        return json_text

    def _extract_json(self, text: str) -> str:
        """
        从文本中提取JSON内容（增强版，处理本地模型的各种输出问题）

        Args:
            text: 模型输出的文本

        Returns:
            提取的JSON字符串
        """
        import re

        text = text.strip()

        # **关键修复：移除<think>思维链标签（Qwen3-32B特性）**
        if '<think>' in text:
            # 移除整个<think>...</think>块
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = text.strip()

        # 移除markdown代码块
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()

        # **关键：立即清理字段名中的空格**
        # "evidence grade" → "evidence_grade"
        text = re.sub(r'"(\w+)\s+(\w+)"\s*:', r'"\1_\2":', text)

        # 查找JSON对象边界
        if "{" in text and "}" in text:
            start = text.find("{")
            # 查找匹配的右括号（处理嵌套）
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_text = text[start:i+1]

                        # 清理常见的JSON错误
                        json_text = self._clean_json_text(json_text)
                        return json_text

        # 如果都失败，返回清理后的原文本
        return self._clean_json_text(text.strip())

    def _clean_json_text(self, json_text: str) -> str:
        """
        清理JSON文本中的常见错误

        Args:
            json_text: 原始JSON文本

        Returns:
            清理后的JSON文本
        """
        import re

        # **关键-1：最优先 - 清理字段名中的空格**
        # "evidence grade":"E1" → "evidence_grade":"E1"
        # "hallucination status":false → "hallucination_status":false
        json_text = re.sub(r'"(\w+)\s+(\w+)"\s*:', r'"\1_\2":', json_text)
        json_text = re.sub(r'"(\w+)\s+(\w+)\s+(\w+)"\s*:', r'"\1_\2_\3":', json_text)  # 三个单词

        # **关键0：先清理单引号字段值（最常见错误！）**
        # 'hallucination_status': → "hallucination_status":
        # 'evidence_grade':'E1' → "evidence_grade":"E1"
        json_text = re.sub(r"'(\w+)'\s*:", r'"\1":', json_text)  # 字段名的单引号
        json_text = re.sub(r":\s*'([^']*)'", r': "\1"', json_text)  # 字段值的单引号

        # **关键1：清理字符串值中的实际控制字符**
        json_text = json_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        # **关键2：修复中文标点符号**
        json_text = json_text.replace('，', ',')  # 中文逗号 → 英文逗号
        json_text = json_text.replace('：', ':')  # 中文冒号 → 英文冒号
        json_text = json_text.replace('"', '"').replace('"', '"')  # 中文引号 → 英文引号

        # **关键3：修复Python布尔值**
        json_text = re.sub(r':\s*True\b', ': true', json_text)  # True → true
        json_text = re.sub(r':\s*False\b', ': false', json_text)  # False → false
        json_text = re.sub(r':\s*None\b', ': null', json_text)  # None → null

        # **关键4：修复字段名拼写错误（从实际运行日志中提取）**
        replacements = {
            # hallucination_status 各种错误拼写
            '"hallucimation_status"': '"hallucination_status"',
            '"hallucitation_status"': '"hallucination_status"',
            '"halluciation_status"': '"hallucination_status"',
            '"halucination_status"': '"hallucination_status"',
            '"halucination status"': '"hallucination_status"',
            '"hallucination status"': '"hallucination_status"',
            '"hallucinations_status"': '"hallucination_status"',
            '"hallucinations_tatus"': '"hallucination_status"',  # 实际错误
            '"Hallucinations_Status"': '"hallucination_status"',  # 大写错误
            '"halluciation_status"': '"hallucination_status"',  # 新错误
            '"is_hallucinated"': '"hallucination_status"',

            # hallucination_type 各种错误拼写
            '"hallucimation_type"': '"hallucination_type"',
            '"hallucitation_type"': '"hallucination_type"',
            '"hallucinaion_type"': '"hallucination_type"',
            '"halluculation_type"': '"hallucination_type"',
            '"halucination_type"': '"hallucination_type"',
            '"halucination _ type"': '"hallucination_type"',  # 带空格
            '"hal l uc i n ation _ t ype"': '"hallucination_type"',  # 多空格
            '"Halucination_Type"': '"hallucination_type"',
            '"hulusion type"': '"hallucination_type"',
            '"halucination type"': '"hallucination_type"',
            '"hallucination type"': '"hallucination_type"',
            '"type_of_halluci nation"': '"hallucination_type"',
            '"type_of_hallucination"': '"hallucination_type"',

            # evidence_grade 各种错误拼写（最高频错误！）
            '"evidence_grage"': '"evidence_grade"',  # 最常见
            '"evidence_garde"': '"evidence_grade"',  # 新错误：字母顺序错误（d<->e互换）
            '"evidence_graade"': '"evidence_grade"',
            '"evidence_grad"': '"evidence_grade"',  # 新错误：少了e
            '"evidence_ade"': '"evidence_grade"',  # 新错误：少了gr
            '"evidence_ grade"': '"evidence_grade"',  # 新错误：字段名内有空格
            '"evidence _grade"': '"evidence_grade"',
            '"hvidence_grade"': '"evidence_grade"',  # 实际错误
            '"evidencegrade"': '"evidence_grade"',
            '"evidence grade"': '"evidence_grade"',
            '"Evidence_Grade"': '"evidence_grade"',  # 实际错误
            '"Evidence_Grage"': '"evidence_grade"',
            '"evidenc e_grade"': '"evidence_grade"',
            '"viden ce_Grade"': '"evidence_grade"',  # 实际错误（多空格）
            '"ev idence_g rade"': '"evidence_grade"',
            '"evidence _ grade"': '"evidence_grade"',
            '"evidenceGrade"': '"evidence_grade"',

            # reasoning 各种错误拼写
            '"reasioning"': '"reasoning"',
            '"reasning"': '"reasoning"',
        }

        for old, new in replacements.items():
            json_text = json_text.replace(old, new)

        # 移除不存在的额外字段（模型自创的）
        extra_fields = [
            r'"hallucinations"\s*:\s*\[[^\]]*\]\s*,?',
            r'"hallucinated_entities"\s*:\s*\[[^\]]*\]\s*,?',
            r'"confidence_level"\s*:\s*"[^"]*"\s*,?',
            r'"confidence_score"\s*:\s*\d+\s*,?',
            r'"source"\s*:\s*"[^"]*"\s*,?',
        ]
        for pattern in extra_fields:
            json_text = re.sub(pattern, '', json_text)

        # 移除非法的转义序列（保留合法的）
        # 只保留 \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        def fix_escapes(match):
            escaped = match.group(1)
            if escaped in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't'] or (escaped.startswith('u') and len(escaped) == 5):
                return '\\' + escaped
            else:
                # 移除非法的转义符
                return escaped

        json_text = re.sub(r'\\(.)', fix_escapes, json_text)

        # 移除可能的行内注释（某些模型会生成）
        json_text = re.sub(r'//.*?$', '', json_text, flags=re.MULTILINE)

        # 移除JSON中的换行符和回车符（在字符串值内的）
        # 这需要小心处理，只在字符串值内替换
        lines = json_text.split('\n')
        cleaned_lines = []
        in_string = False
        for line in lines:
            # 简单处理：如果不在合法的JSON结构中，跳过
            if not line.strip():
                continue
            cleaned_lines.append(line)

        json_text = '\n'.join(cleaned_lines)

        return json_text


def create_local_llm(config: Dict[str, Any]) -> LocalLLM:
    """
    根据配置创建本地LLM实例

    Args:
        config: 配置字典，包含model_path等参数

    Returns:
        LocalLLM实例
    """
    return LocalLLM(
        model_path=config['model_path'],
        temperature=config.get('temperature', 0.3),
        max_tokens=config.get('max_tokens', 2000),
        device_map=config.get('device_map', 'auto'),
        torch_dtype=config.get('torch_dtype', 'bfloat16'),
        trust_remote_code=config.get('trust_remote_code', False)
    )
