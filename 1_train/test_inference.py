#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微调模型推理测试脚本
支持测试LoRA、全参数微调后的模型

Usage:
    python test_inference.py --checkpoint_dir ./output/lora_finetune/vx-xxx/checkpoint-xxx
    python test_inference.py --model_dir ./output/full_finetune --full_model
"""

import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

import torch
from swift.llm import PtEngine, RequestConfig, InferRequest, safe_snapshot_download
from swift.llm import get_model_tokenizer, get_template
from swift.tuners import Swift
from swift.utils import get_logger

logger = get_logger()

class ModelTester:
    """模型推理测试器"""
    
    def __init__(self):
        self.engine: Optional[PtEngine] = None
        self.setup_environment()
    
    def setup_environment(self):
        """设置环境"""
        if torch.cuda.is_available():
            logger.info(f"检测到 {torch.cuda.device_count()} 张GPU")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
        else:
            logger.warning("未检测到GPU，使用CPU推理")
    
    def load_lora_model(self, checkpoint_dir: str, model_name: Optional[str] = None):
        """加载LoRA微调后的模型"""
        logger.info(f"加载LoRA模型: {checkpoint_dir}")
        
        # 如果是ModelScope上的模型，先下载
        if not os.path.exists(checkpoint_dir):
            checkpoint_dir = safe_snapshot_download(checkpoint_dir)
        
        # 从checkpoint中读取配置
        from swift.llm import BaseArguments
        args = BaseArguments.from_pretrained(checkpoint_dir)
        
        logger.info(f"基础模型: {getattr(args, 'model', 'unknown')}")
        logger.info(f"模板类型: {getattr(args, 'template', 'default')}")
        
        # 加载基础模型
        model_path = getattr(args, 'model', None)
        if not model_path:
            raise ValueError("无法获取基础模型路径")
        model, tokenizer = get_model_tokenizer(model_path)
        
        # 加载LoRA权重
        model = Swift.from_pretrained(model, checkpoint_dir)
        
        # 设置模板
        template_type = getattr(args, 'template', 'default')
        system_prompt = getattr(args, 'system', None)
        template = get_template(template_type, tokenizer, default_system=system_prompt)
        
        # 创建推理引擎
        self.engine = PtEngine.from_model_template(model, template, max_batch_size=2)
        
        logger.info("LoRA模型加载完成")
        return args
    
    def load_full_model(self, model_dir: str):
        """加载全参数微调后的模型"""
        logger.info(f"加载全参数模型: {model_dir}")
        
        # 加载模型和tokenizer
        model, tokenizer = get_model_tokenizer(model_dir)
        
        # 获取模板 - 使用默认模板
        template_type = "default"
        template = get_template(template_type, tokenizer)
        
        # 创建推理引擎  
        self.engine = PtEngine.from_model_template(model, template, max_batch_size=2)
        
        logger.info("全参数模型加载完成")
    
    def _extract_response_content(self, resp: Any) -> str:
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
    
    def single_inference(self, query: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """单次推理"""
        if self.engine is None:
            raise ValueError("请先加载模型")
        
        request_config = RequestConfig(max_tokens=max_tokens, temperature=temperature)
        infer_request = InferRequest(messages=[{'role': 'user', 'content': query}])
        
        start_time = time.time()
        resp_list = self.engine.infer([infer_request], request_config)
        inference_time = time.time() - start_time
        
        # 从响应中提取内容
        response = ""
        if resp_list and len(resp_list) > 0:
            response = self._extract_response_content(resp_list[0])
        
        logger.info(f"推理耗时: {inference_time:.2f}秒")
        return response
    
    def batch_inference(self, queries: List[str], max_tokens: int = 512, temperature: float = 0.7) -> List[str]:
        """批量推理"""
        if self.engine is None:
            raise ValueError("请先加载模型")
        
        request_config = RequestConfig(max_tokens=max_tokens, temperature=temperature)
        infer_requests = [InferRequest(messages=[{'role': 'user', 'content': query}]) for query in queries]
        
        start_time = time.time()
        resp_list = self.engine.infer(infer_requests, request_config)
        inference_time = time.time() - start_time
        
        # 从响应中提取内容
        responses = []
        for resp in resp_list:
            response = self._extract_response_content(resp)
            responses.append(response)
        
        logger.info(f"批量推理 {len(queries)} 个请求耗时: {inference_time:.2f}秒")
        logger.info(f"平均每个请求: {inference_time/len(queries):.2f}秒")
        
        return responses
    
    def test_conversation(self):
        """测试对话能力"""
        print("\n" + "="*50)
        print("开始对话测试（输入 'quit' 退出）")
        print("="*50)
        
        conversation_history: List[Dict[str, str]] = []
        
        while True:
            user_input = input("\n用户: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
                
            if not user_input:
                continue
            
            try:
                # 构建完整对话历史
                messages = conversation_history + [{'role': 'user', 'content': user_input}]
                # 使用Any类型避免类型检查问题
                from typing import Any
                infer_request = InferRequest(messages=messages)  # type: ignore
                request_config = RequestConfig(max_tokens=512, temperature=0.7)
                
                start_time = time.time()
                if self.engine is not None:
                    resp_list = self.engine.infer([infer_request], request_config)
                    inference_time = time.time() - start_time
                    
                    response = ""
                    if resp_list and len(resp_list) > 0:
                        response = self._extract_response_content(resp_list[0])
                    
                    print(f"\n助手: {response}")
                    print(f"耗时: {inference_time:.2f}秒")
                    
                    # 更新对话历史
                    conversation_history.append({'role': 'user', 'content': user_input})
                    conversation_history.append({'role': 'assistant', 'content': response})
                    
                    # 限制对话历史长度
                    if len(conversation_history) > 10:
                        conversation_history = conversation_history[-10:]
                else:
                    print("错误：模型未加载")
                    
            except Exception as e:
                print(f"推理出错: {e}")
    
    def run_predefined_tests(self):
        """运行预定义测试"""
        test_queries = [
            "你是谁？",
            "请介绍一下人工智能",
            "用Python写一个冒泡排序",
            "解释一下什么是深度学习",
            "今天天气怎么样？"
        ]
        
        print("\n" + "="*50)
        print("开始预定义测试")
        print("="*50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n测试 {i}: {query}")
            print("-" * 40)
            
            try:
                response = self.single_inference(query)
                print(f"回答: {response}")
            except Exception as e:
                print(f"推理出错: {e}")
    
    def benchmark_performance(self, num_requests: int = 10):
        """性能基准测试"""
        print(f"\n开始性能测试 ({num_requests} 个请求)")
        print("="*50)
        
        test_query = "请介绍一下人工智能的发展历史"
        queries = [test_query] * num_requests
        
        # 批量测试
        start_time = time.time()
        responses = self.batch_inference(queries, max_tokens=256)
        total_time = time.time() - start_time
        
        print(f"批量推理总耗时: {total_time:.2f}秒")
        print(f"平均每个请求: {total_time/num_requests:.2f}秒")
        print(f"吞吐量: {num_requests/total_time:.2f} 请求/秒")
        
        # 单个测试对比
        single_times = []
        for i in range(min(3, num_requests)):
            start_time = time.time()
            self.single_inference(test_query, max_tokens=256)
            single_time = time.time() - start_time
            single_times.append(single_time)
        
        avg_single_time = sum(single_times) / len(single_times)
        print(f"单次推理平均耗时: {avg_single_time:.2f}秒")

def main():
    parser = argparse.ArgumentParser(description="微调模型推理测试")
    parser.add_argument("--checkpoint_dir", type=str, help="LoRA检查点目录")
    parser.add_argument("--model_dir", type=str, help="全参数微调模型目录")
    parser.add_argument("--full_model", action="store_true", help="使用全参数模型")
    parser.add_argument("--test_type", choices=["single", "batch", "conversation", "predefined", "benchmark"], 
                        default="predefined", help="测试类型")
    parser.add_argument("--query", type=str, help="单次推理的查询内容")
    parser.add_argument("--max_tokens", type=int, default=512, help="最大生成tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ModelTester()
    
    # 加载模型
    try:
        if args.full_model and args.model_dir:
            tester.load_full_model(args.model_dir)
        elif args.checkpoint_dir:
            tester.load_lora_model(args.checkpoint_dir)
        else:
            print("错误: 请指定 --checkpoint_dir 或 --model_dir")
            return
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 执行测试
    if args.test_type == "single":
        if args.query:
            response = tester.single_inference(args.query, args.max_tokens, args.temperature)
            print(f"查询: {args.query}")
            print(f"回答: {response}")
        else:
            print("单次推理需要提供 --query 参数")
    elif args.test_type == "conversation":
        tester.test_conversation()
    elif args.test_type == "predefined":
        tester.run_predefined_tests()
    elif args.test_type == "benchmark":
        tester.benchmark_performance()

if __name__ == "__main__":
    main()