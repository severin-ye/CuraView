"""
本地LLM封装类 - JSON输出模式（支持with_structured_output）
支持使用本地Qwen3模型进行推理，一次性生成完整JSON输出
模仿LangChain的API模式，支持with_structured_output方法
使用constrained generation强制JSON格式输出
"""

import os
import json
import re
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Type
from contextlib import contextmanager
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# 配置详细调试日志
DEBUG_LOG_DIR = Path("experiment/幻觉检测_qwen/output/排错日志")
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)

# 加载本地提示词
from curaview.common.source_layout import retained_path
PROMPT_DIR = retained_path("doc/prompts")
LOCAL_PROMPT_FILE = PROMPT_DIR / "detection_prompt_local_json.md"

def load_local_prompt() -> str:
    """加载本地JSON输出提示词"""
    text = LOCAL_PROMPT_FILE.read_text(encoding='utf-8')
    if not text.strip():
        raise ValueError('configured local JSON prompt is empty')
    return text

def get_debug_logger(name: str) -> logging.Logger:
    """获取调试日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = DEBUG_LOG_DIR / f"debug_json_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - [线程%(thread)d] - %(message)s'))
        logger.addHandler(file_handler)
        logger.propagate = False

        logger.info(f"=" * 80)
        logger.info(f"JSON模式调试日志 - {timestamp}")
        logger.info(f"=" * 80)

    return logger

debug_logger = get_debug_logger("local_llm_json_debug")


def normalize_messages(messages):
    """Preserve caller message order/content; reject unsupported content explicitly."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise TypeError("messages must be a nonempty sequence of role/content messages")
    normalized = []
    for message in messages:
        if isinstance(message, dict):
            role, content = message.get('role'), message.get('content')
        else:
            role = getattr(message, 'role', None) or getattr(message, 'type', None)
            content = getattr(message, 'content', None)
        role = {'human': 'user', 'ai': 'assistant'}.get(role, role)
        if role not in ('system', 'user', 'assistant', 'tool', 'function', 'developer'):
            raise TypeError("unsupported message role")
        if not isinstance(content, str):
            raise TypeError("local JSON messages require text content")
        # Rendering metadata we cannot represent must never be silently discarded.
        for key in ('tool_calls', 'function_call', 'tool_call_id', 'name', 'additional_kwargs'):
            value = message.get(key) if isinstance(message, dict) else getattr(message, key, None)
            if value:
                raise TypeError("local JSON messages do not support tool/function calls")
        normalized.append({'role': role, 'content': content})
    return normalized


class _UntracedCall:
    id = None
    def record(self, event, payload):
        pass
    def finish(self, payload=None):
        pass


@contextmanager
def optional_trace(store, stage, payload):
    if store is None:
        yield _UntracedCall()
    else:
        with store.call(stage, payload) as call:
            yield call


class JSONFormatEnforcer(LogitsProcessor):
    """强制JSON格式输出的Logit Processor - 简化版"""

    def __init__(self, tokenizer, eos_token_id):
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.json_started = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """在生成开始时强制输出 { """
        # 只在第一个生成的token时起作用
        if input_ids.shape[1] > 0 and not self.json_started:
            last_tokens = input_ids[0, -3:].tolist() if input_ids.shape[1] >= 3 else input_ids[0].tolist()
            text = self.tokenizer.decode(last_tokens)

            # 如果还没有看到 {，提升 { 的概率
            if '{' not in text:
                # 找到 { 对应的token id
                brace_tokens = self.tokenizer.encode('{', add_special_tokens=False)
                if brace_tokens:
                    for token_id in brace_tokens:
                        scores[0, token_id] += 10.0  # 大幅提升 { 的概率
            else:
                self.json_started = True

        return scores


class LocalLLM:
    """本地LLM基础类（JSON输出模式）"""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        trace_store: Any = None,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.trace_store = trace_store

        print("🔄 正在加载本地模型 (JSON模式): " + model_path, flush=True)
        print("  ⏳ 加载tokenizer...", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True
        )

        print("  ⏳ 加载模型权重...", flush=True)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype_map.get(torch_dtype, torch.bfloat16),
            trust_remote_code=trust_remote_code,
            local_files_only=True
        )

        print("✅ 模型加载成功", flush=True)

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 GPU显存占用: {memory_allocated:.2f}GB\n", flush=True)

    def invoke(self, messages: List[Dict[str, str]]) -> Any:
        """调用模型生成JSON响应"""
        # 使用局部变量存储temperature，避免修改实例属性
        local_temperature = self.temperature

        # 构建prompt
        prompt = self._build_prompt(messages)

        # 调试日志
        debug_logger.debug(f"=" * 80)
        debug_logger.debug(f"JSON模式生成请求")
        debug_logger.debug(f"Prompt长度: {len(prompt)} 字符")
        debug_logger.debug(f"Temperature: {local_temperature}")

        parameters = {'max_new_tokens': self.max_tokens, 'temperature': local_temperature,
                      'do_sample': local_temperature > 0, 'pad_token_id': self.tokenizer.eos_token_id}
        with optional_trace(getattr(self, 'trace_store', None), 'legacy.local.generate',
                            {'messages': normalize_messages(messages), 'rendered_input': prompt,
                             'generation_parameters': parameters}) as call:
            # Registration precedes tokenization, so tokenization failures remain observable.
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            call.record('tokenized', {key: value.tolist() for key, value in inputs.items()})
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **parameters)
            width = inputs['input_ids'].shape[1]
            generated_ids = outputs[0, width:].tolist()
            # Persist tensor output before decoding, which can itself fail.
            call.record('generated', {'output_tensor_ids': outputs.tolist(),
                                     'generated_token_ids': generated_ids})
            raw_generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            json_prefix = '{\n  "reasoning": "'
            response_text = json_prefix + raw_generated_text.strip()
            call.record('decoded', {'raw_generated_text': raw_generated_text,
                                    'json_prefill_prefix': json_prefix,
                                    'derived_content': response_text})
            class Response:
                def __init__(self):
                    self.content = response_text
                    self.raw_generated_text = raw_generated_text
                    self.generated_token_ids = generated_ids
                    self.trace_call_id = call.id
            response = Response()
            call.finish({'response_is_derived_prefilled_json': True})
            return response

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建prompt - 强制从{开始"""
        prompt_parts = []
        for msg in normalize_messages(messages):
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # 在assistant回复开始处添加{，强制JSON格式
        prompt_parts.append("<|im_start|>assistant\n{\n  \"reasoning\": \"")
        return "\n".join(prompt_parts)

    def with_structured_output(self, schema: Type[BaseModel]) -> 'StructuredLocalLLM':
        """
        模仿LangChain的with_structured_output方法
        返回一个支持结构化输出的包装器

        Args:
            schema: Pydantic模型类

        Returns:
            StructuredLocalLLM实例
        """
        return StructuredLocalLLM(self, schema)


class StructuredLocalLLM:
    """支持结构化输出的本地LLM（JSON模式）- 模仿LangChain API"""

    def __init__(self, base_llm: LocalLLM, schema: Type[BaseModel]):
        self.base_llm = base_llm
        self.schema = schema

        # 加载系统提示词
        self.system_prompt = load_local_prompt()

        # 生成JSON schema（像LangChain一样）
        self.json_schema = self._generate_json_schema()

        debug_logger.info("✅ JSON模式系统提示词已加载")
        debug_logger.info(f"✅ JSON Schema已生成:\n{json.dumps(self.json_schema, indent=2, ensure_ascii=False)}")

    def _generate_json_schema(self) -> dict:
        """从Pydantic模型生成JSON schema"""
        # 获取模型的schema
        schema_dict = self.schema.model_json_schema()
        return {
            "type": "object",
            "properties": schema_dict.get("properties", {}),
            "required": schema_dict.get("required", [])
        }

    def invoke(self, input_data):
        """执行JSON生成并解析为结构化对象"""
        import threading
        thread_id = threading.current_thread().ident

        debug_logger.info(f"\n{'='*80}")
        debug_logger.info(f"[线程{thread_id}] 开始JSON模式检测")

        if isinstance(input_data, (list, tuple)):
            messages = normalize_messages(input_data)
            return self._invoke_messages(messages, messages, 'caller_messages')
        if not isinstance(input_data, dict) or not {'sentence', 'evidence'}.issubset(input_data):
            raise TypeError('expected role/content messages or a sentence/evidence dict')
        # 提取sentence和evidence
        if isinstance(input_data, dict):
            sentence = input_data.get('sentence', '')
            evidence = input_data.get('evidence', '')

        if not isinstance(sentence, str) or not isinstance(evidence, str):
            raise TypeError('sentence and evidence must be text')
        debug_logger.info(f"[线程{thread_id}] 句子: {sentence[:100]}...")

        print(f"\n🔄 [线程{thread_id}] JSON模式检测", flush=True)
        print(f"  📌 句子: {sentence[:80]}...", flush=True)

        # 构建增强的系统提示（包含JSON schema）
        enhanced_system_prompt = f"""{self.system_prompt}

## JSON Schema要求

你的输出**必须**严格符合以下JSON schema:

```json
{json.dumps(self.json_schema, indent=2, ensure_ascii=False)}
```

**重要提醒：**
- 字段名必须与schema完全一致
- 字段类型必须符合schema定义
- 必须包含所有required字段
- 不要输出schema中未定义的额外字段
- 确保JSON格式正确（双引号、逗号、冒号）"""

        # 构建用户消息
        user_prompt = f"""**检测任务**

**句子:** {sentence}

**GraphRAG证据:**
{evidence}

**要求:** 请严格按照上述JSON schema输出检测结果。直接输出JSON，从{{开始。"""

        messages = [
            {"role": "system", "content": enhanced_system_prompt},  # 使用增强的系统提示
            {"role": "user", "content": user_prompt}
        ]

        return self._invoke_messages(messages, input_data, 'sentence_evidence_enhancement')

    def _invoke_messages(self, messages, original_input, input_mode):
        store = getattr(self.base_llm, 'trace_store', None)
        with optional_trace(store, 'legacy.local.structured',
                            {'original_input': original_input, 'used_messages': messages,
                             'input_mode': input_mode, 'schema': self.json_schema}) as call:
            response = self.base_llm.invoke(messages)
            raw_output = response.content
            parse_input = {'original_response_content': raw_output,
                           'raw_generated_text': getattr(response, 'raw_generated_text', None),
                           'generation_call_id': getattr(response, 'trace_call_id', None)}
            with optional_trace(store, 'legacy.local.parse', parse_input) as parse_call:
                parse_call.record('parse_input', parse_input)
                json_str = self._extract_and_repair_json(raw_output)
                parse_call.record('repaired', {'original_response_content': raw_output,
                                              'repaired_text': json_str})
                data = json.loads(json_str)
                result = self.schema(**data)
                parse_call.record('parsed', {'parsed_result': result.model_dump(mode='json')})
                parse_call.finish({'parse_success': True})
            call.finish({'parse_success': True})
            return result

    def _extract_and_repair_json(self, text: str) -> str:
        """提取并修复JSON字符串（智能修复版）"""
        # 移除markdown代码块标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # **关键修复1：先统一替换中文引号为英文引号**
        # 这样后续的括号匹配和JSON解析才能正常工作
        text = text.replace('"', '"').replace('"', '"')
        debug_logger.debug("预处理步骤1：替换中文引号为英文引号")

        # **关键修复2：修复 "reasoning": ""， 模式（必须在提取JSON之前）**
        # 这种格式会导致括号匹配算法错误判断字符串边界
        pattern_empty_reasoning = r'"reasoning"\s*:\s*""([，。；：、])'
        if re.search(pattern_empty_reasoning, text):
            debug_logger.debug("预处理步骤2：检测到空reasoning后跟中文标点，立即修复")
            text = re.sub(pattern_empty_reasoning, r'"reasoning": "', text)
            debug_logger.debug("预处理步骤2：已修复为 \"reasoning\": \"")

        # 尝试多种修复策略
        strategies = [
            self._try_direct_parse,
            self._try_fix_common_errors,
            self._try_aggressive_repair,
            self._try_json_repair_library
        ]

        for strategy in strategies:
            try:
                result = strategy(text)
                if result:
                    # 验证JSON
                    json.loads(result)
                    debug_logger.info(f"JSON修复成功: 使用策略 {strategy.__name__}")
                    return result
            except Exception as e:
                debug_logger.debug(f"策略 {strategy.__name__} 失败: {e}")
                continue

        raise ValueError("所有JSON修复策略都失败")

    def _try_direct_parse(self, text: str) -> Optional[str]:
        """策略1: 直接提取并解析"""
        json_str = self._extract_json_with_brackets(text)
        if json_str:
            # 直接尝试解析
            json.loads(json_str)  # 验证
            return json_str
        return None

    def _try_fix_common_errors(self, text: str) -> Optional[str]:
        """策略2: 修复常见错误"""
        json_str = self._extract_json_with_brackets(text)
        if not json_str:
            return None

        # 修复常见问题
        # 1. 修复中文标点符号导致的格式错误
        json_str = self._fix_chinese_punctuation(json_str)

        # 2. 修复reasoning字段中的引号问题
        json_str = self._fix_reasoning_field(json_str)

        # 3. 修复null和布尔值
        json_str = re.sub(r':\s*"null"', ': null', json_str)
        json_str = re.sub(r':\s*"true"', ': true', json_str)
        json_str = re.sub(r':\s*"false"', ': false', json_str)

        # 4. 移除末尾多余逗号
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        return json_str

    def _fix_chinese_punctuation(self, json_str: str) -> str:
        """修复中文标点符号导致的JSON格式错误

        注意：中文引号已在_extract_and_repair_json中预处理过
        这里主要处理：
        - "reasoning": ""，但... → "reasoning": "但...
        - reasoning字段中的中文逗号，→ ,
        """
        debug_logger.debug(f"开始中文标点修复，原始长度: {len(json_str)}")

        # 1. 修复 "reasoning": ""，... 模式（空字符串后跟中文逗号）
        # 注意：此时中文引号已经是英文引号了
        pattern_empty = r'"reasoning"\s*:\s*""([，。；：、])'
        if re.search(pattern_empty, json_str):
            debug_logger.debug(f"检测到空reasoning后跟中文标点")
            # 移除空字符串和中文标点
            json_str = re.sub(pattern_empty, r'"reasoning": "', json_str)
            debug_logger.debug("已移除空字符串和中文标点")

        # 2. 将reasoning字段值中的中文逗号替换为英文逗号
        # 但要小心：不要替换JSON结构中的逗号（字段之间的分隔符）
        def fix_reasoning_content(match):
            prefix = match.group(1)  # "reasoning": "
            content = match.group(2)  # 实际内容
            suffix = match.group(3)   # 结尾的 "

            # 只替换内容中的中文标点
            content = content.replace('，', ',')

            return prefix + content + suffix

        # 匹配reasoning字段：非贪婪匹配到下一个引号
        reasoning_full_pattern = r'("reasoning"\s*:\s*")(.*?)(")'
        json_str = re.sub(reasoning_full_pattern, fix_reasoning_content, json_str, flags=re.DOTALL)

        debug_logger.debug(f"中文标点修复完成，新长度: {len(json_str)}")

        return json_str

    def _fix_reasoning_field(self, json_str: str) -> str:
        """智能修复reasoning字段中的引号问题"""
        # 查找reasoning字段
        pattern = r'"reasoning"\s*:\s*"(.*?)"(?=\s*,\s*"hallucination_status"|$)'

        def fix_quotes(match):
            content = match.group(1)
            # 转义内部的双引号和反斜杠
            content = content.replace('\\', '\\\\')  # 先转义反斜杠
            content = content.replace('"', '\\"')     # 再转义引号
            # 移除可能的换行符
            content = content.replace('\n', ' ').replace('\r', ' ')
            return f'"reasoning": "{content}"'

        # 使用非贪婪匹配，但要处理可能的多行
        # 先尝试简单匹配
        try:
            result = re.sub(pattern, fix_quotes, json_str, flags=re.DOTALL)
            return result
        except:
            # 如果失败，使用更激进的方法
            return self._fix_reasoning_aggressive(json_str)

    def _fix_reasoning_aggressive(self, json_str: str) -> str:
        """激进的reasoning字段修复"""
        # 找到reasoning字段的开始
        start_pattern = r'"reasoning"\s*:\s*"'
        match = re.search(start_pattern, json_str)
        if not match:
            return json_str

        start_pos = match.end()

        # 从这里开始找到字段结束位置
        # 查找下一个字段名（hallucination_status）
        end_pattern = r'"\s*,\s*"hallucination_status'
        end_match = re.search(end_pattern, json_str[start_pos:])

        if not end_match:
            return json_str

        # 提取reasoning内容（不含两端引号）
        end_pos = start_pos + end_match.start()
        reasoning_content = json_str[start_pos:end_pos]

        # 转义特殊字符
        reasoning_content = reasoning_content.replace('\\', '\\\\')
        reasoning_content = reasoning_content.replace('"', '\\"')
        reasoning_content = reasoning_content.replace('\n', ' ')
        reasoning_content = reasoning_content.replace('\r', ' ')

        # 重构JSON
        new_json = json_str[:start_pos] + reasoning_content + json_str[end_pos:]
        return new_json

    def _try_aggressive_repair(self, text: str) -> Optional[str]:
        """策略3: 激进修复 - 重新构建JSON"""
        json_str = self._extract_json_with_brackets(text)
        if not json_str:
            return None

        try:
            # 尝试从不完整的JSON中提取关键字段
            reasoning_match = re.search(r'"reasoning"\s*:\s*"(.+?)"(?=\s*,\s*"|$)', json_str, re.DOTALL)
            status_match = re.search(r'"hallucination_status"\s*:\s*(true|false)', json_str)
            type_match = re.search(r'"hallucination_type"\s*:\s*(?:"(\w+)"|null)', json_str)
            grade_match = re.search(r'"evidence_grade"\s*:\s*"([^"]+)"', json_str)

            if reasoning_match and status_match:
                # 重新构建JSON
                reasoning = reasoning_match.group(1).replace('"', '\\"').replace('\n', ' ')
                status = status_match.group(1)
                hal_type = type_match.group(1) if (type_match and type_match.group(1)) else None
                grade = grade_match.group(1) if grade_match else "E2"

                new_json = {
                    "reasoning": reasoning,
                    "hallucination_status": status == "true",
                    "hallucination_type": hal_type,
                    "evidence_grade": grade
                }

                return json.dumps(new_json, ensure_ascii=False)
        except Exception as e:
            debug_logger.debug(f"激进修复失败: {e}")

        return None

    def _try_json_repair_library(self, text: str) -> Optional[str]:
        """策略4: 使用json_repair库（如果可用）"""
        try:
            import json_repair
            json_str = self._extract_json_with_brackets(text)
            if json_str:
                repaired = json_repair.repair_json(json_str)
                return repaired
        except ImportError:
            debug_logger.debug("json_repair库不可用")
        except Exception as e:
            debug_logger.debug(f"json_repair库修复失败: {e}")

        return None

    def _extract_json_with_brackets(self, text: str) -> Optional[str]:
        """使用括号匹配提取完整JSON"""
        # 找到第一个 {
        start = text.find('{')
        if start == -1:
            return None

        # 使用栈匹配括号
        stack = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            char = text[i]

            # 处理转义字符
            if escape:
                escape = False
                continue

            if char == '\\':
                escape = True
                continue

            # 处理字符串
            if char == '"':
                in_string = not in_string
                continue

            # 在字符串内部，忽略括号
            if in_string:
                continue

            # 计数括号
            if char == '{':
                stack += 1
            elif char == '}':
                stack -= 1
                if stack == 0:
                    # 找到匹配的结束括号
                    return text[start:i+1]

        return None


def create_local_llm(config: Dict[str, Any]) -> LocalLLM:
    """
    创建本地LLM实例（JSON模式）

    返回基础LLM对象，支持with_structured_output方法
    这样可以像API模式一样使用：
        llm = create_local_llm(config)
        llm_structured = llm.with_structured_output(schema)
    """
    base_llm = LocalLLM(
        model_path=config['model_path'],
        device_map=config.get('device_map', 'auto'),
        torch_dtype=config.get('torch_dtype', 'bfloat16'),
        trust_remote_code=config.get('trust_remote_code', False),
        temperature=config.get('temperature', 0.3),
        max_tokens=config.get('max_tokens', 2000)
    )

    return base_llm  # 返回基础LLM，支持with_structured_output
