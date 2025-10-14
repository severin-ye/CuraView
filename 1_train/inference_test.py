#!/usr/bin/env python3
"""
微调后模型推理测试脚本
支持LoRA、全参数微调后的模型推理测试
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from swift.llm import (
    PtEngine, RequestConfig, InferRequest, 
    get_model_tokenizer, get_template, BaseArguments
)
from swift.tuners import Swift


class ModelInferenceTester:
    """微调后模型推理测试器"""
    
    def __init__(self, checkpoint_path: str, test_questions: Optional[List[str]] = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.test_questions = test_questions if test_questions is not None else self._get_default_questions()
        self.engine = None
        self.is_lora = False
        
    def _get_default_questions(self) -> List[str]:
        """获取默认测试问题"""
        return [
            "你是谁？",
            "你能做什么？",
            "请写一首关于春天的诗",
            "解释一下什么是机器学习",
            "用Python写一个快速排序算法",
            "今天天气怎么样？",
            "请总结一下人工智能的发展历史",
            "什么是大语言模型？",
        ]
    
    def _load_model_from_lora_checkpoint(self):
        """从LoRA checkpoint加载模型"""
        print(f"📁 加载LoRA checkpoint: {self.checkpoint_path}")
        
        # 加载训练参数
        args = BaseArguments.from_pretrained(str(self.checkpoint_path))
        print(f"🔧 基础模型: {args.model}")
        print(f"🎨 模板类型: {getattr(args, 'template', 'default')}")
        print(f"💭 系统提示: {getattr(args, 'system', None)}")
        
        # 加载模型和分词器
        model_id = getattr(args, 'model', None)
        if not model_id:
            raise ValueError("无法获取基础模型路径")
        model, tokenizer = get_model_tokenizer(model_id)
        
        # 加载LoRA权重
        model = Swift.from_pretrained(model, str(self.checkpoint_path))
        
        # 创建模板
        template_type = getattr(args, 'template', 'default')
        system_prompt = getattr(args, 'system', None)
        template = get_template(template_type, tokenizer, default_system=system_prompt)
        
        # 创建推理引擎
        self.engine = PtEngine.from_model_template(model, template, max_batch_size=1)
        self.is_lora = True
        
    def _load_model_from_full_checkpoint(self):
        """从全参数checkpoint加载模型"""
        print(f"📁 加载全参数checkpoint: {self.checkpoint_path}")
        
        # 检查是否有训练参数文件
        args_file = self.checkpoint_path / "args.json"
        if args_file.exists():
            args = BaseArguments.from_pretrained(str(self.checkpoint_path))
            model_path = str(self.checkpoint_path)
            template_type = getattr(args, 'template', 'default')
            default_system = getattr(args, 'system', None)
        else:
            # 如果没有args.json，假设checkpoint就是模型路径
            model_path = str(self.checkpoint_path)
            template_type = 'default'
            default_system = None
        
        # 加载模型和分词器
        model, tokenizer = get_model_tokenizer(model_path)
        
        # 创建模板
        template = get_template(template_type, tokenizer, default_system=default_system)
        
        # 创建推理引擎
        self.engine = PtEngine.from_model_template(model, template, max_batch_size=1)
        self.is_lora = False
    
    def load_model(self):
        """自动检测并加载模型"""
        print("🚀 开始加载模型...")
        
        # 检查是否为LoRA checkpoint（包含adapter相关文件）
        adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
        has_adapter = any((self.checkpoint_path / f).exists() for f in adapter_files)
        
        if has_adapter:
            self._load_model_from_lora_checkpoint()
        else:
            self._load_model_from_full_checkpoint()
    def _extract_response_content(self, resp) -> str:
        """从响应中提取内容"""
        try:
            if hasattr(resp, 'choices') and resp.choices:
                choice = resp.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    return str(content) if content is not None else ""
        except (AttributeError, IndexError, TypeError):
            pass
        return ""
    
    def test_single_question(self, question: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """测试单个问题"""
        if not self.engine:
            raise ValueError("模型未加载，请先调用load_model()")
        
        request_config = RequestConfig(max_tokens=max_tokens, temperature=temperature)
        infer_request = InferRequest(messages=[{'role': 'user', 'content': question}])
        
        resp_list = self.engine.infer([infer_request], request_config)
        if resp_list and len(resp_list) > 0:
            return self._extract_response_content(resp_list[0])
        return ""
    
    def run_batch_test(self, save_results: bool = True, output_file: str = "") -> Dict[str, Any]:
        """批量测试"""
        print("🧪 开始批量测试...")
        
        results = {
            "checkpoint_path": str(self.checkpoint_path),
            "model_type": "LoRA" if self.is_lora else "Full",
            "test_results": []
        }
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n📝 问题 {i}/{len(self.test_questions)}: {question}")
            print("-" * 50)
            
            try:
                response = self.test_single_question(question)
                print(f"🤖 回答: {response}")
                
                results["test_results"].append({
                    "question": question,
                    "response": response,
                    "status": "success"
                })
                
            except Exception as e:
                error_msg = f"推理失败: {str(e)}"
                print(f"❌ {error_msg}")
                
                results["test_results"].append({
                    "question": question,
                    "response": "",
                    "status": "error",
                    "error": error_msg
                })
        
        # 保存结果
        if save_results:
            if not output_file:
                output_file = f"inference_results_{self.checkpoint_path.name}.json"
            
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 测试结果已保存到: {output_path}")
        
        return results
    
    def interactive_test(self):
        """交互式测试"""
        print("🎮 进入交互式测试模式 (输入 'quit' 退出)")
        
        while True:
            try:
                question = input("\n👤 请输入问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见!")
                    break
                
                if not question:
                    continue
                
                print("🤖 AI回答:")
                response = self.test_single_question(question)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 推理错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="微调后模型推理测试")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="模型checkpoint路径")
    parser.add_argument("--mode", "-m", choices=["batch", "interactive", "single"], 
                       default="batch", help="测试模式")
    parser.add_argument("--question", "-q", type=str, 
                       help="单个问题测试（仅在single模式下）")
    parser.add_argument("--questions-file", type=str,
                       help="自定义问题文件路径（JSON格式）")
    parser.add_argument("--output", "-o", type=str,
                       help="结果输出文件路径")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    parser.add_argument("--gpu", type=str, default="0",
                       help="指定GPU设备")
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 加载自定义问题（如果提供）
    test_questions = None
    # 处理测试问题
    test_questions = None
    if args.questions_file:
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
            test_questions = questions_data if isinstance(questions_data, list) else questions_data.get('questions', [])
    
    # 创建测试器
    tester = ModelInferenceTester(args.checkpoint, test_questions)
    
    # 加载模型
    tester.load_model()
    
    # 根据模式执行测试
    if args.mode == "batch":
        tester.run_batch_test(save_results=True, output_file=args.output)
    elif args.mode == "interactive":
        tester.interactive_test()
    elif args.mode == "single":
        if not args.question:
            print("❌ single模式需要指定--question参数")
            return
        
        print(f"📝 问题: {args.question}")
        response = tester.test_single_question(
            args.question, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature
        )
        print(f"🤖 回答: {response}")


if __name__ == "__main__":
    main()