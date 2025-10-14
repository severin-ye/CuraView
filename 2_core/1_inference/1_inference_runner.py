#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心推理模块 - 基于MS-Swift的模型推理器
支持LoRA、全参数微调后的模型推理
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# 添加utils路径
sys.path.append(str(Path(__file__).parent.parent.parent / "1_utils"))

from config_loader import ConfigLoader
from logger import Logger

try:
    import torch
    from swift.llm import (
        PtEngine, RequestConfig, InferRequest, 
        get_model_tokenizer, get_template, BaseArguments
    )
    from swift.tuners import Swift
except ImportError as e:
    print(f"❌ 导入依赖模块失败: {e}")
    print("请确保已安装: pip install ms-swift torch")
    sys.exit(1)

class InferenceManager:
    """推理管理器 - 统一的模型推理接口"""
    
    def __init__(self, checkpoint_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        初始化推理管理器
        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径
        """
        self.logger = Logger("InferenceManager").get_logger()
        self.config_loader = ConfigLoader()
        
        # 加载配置
        if config_path:
            self.config = self.config_loader.load_config(config_path)
        else:
            self.config = {}
        
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.engine = None
        self.is_lora = False
        self.model_info = {}
    
    def detect_model_type(self, checkpoint_path: Path) -> str:
        """
        检测模型类型
        Args:
            checkpoint_path: 检查点路径
        Returns:
            str: 模型类型 ('lora' 或 'full')
        """
        # 检查是否为LoRA checkpoint（包含adapter相关文件）
        adapter_files = [
            "adapter_config.json", 
            "adapter_model.safetensors", 
            "adapter_model.bin",
            "adapter_model.pt"
        ]
        
        has_adapter = any((checkpoint_path / f).exists() for f in adapter_files)
        return "lora" if has_adapter else "full"
    
    def load_lora_model(self, checkpoint_path: Path):
        """
        加载LoRA模型
        Args:
            checkpoint_path: LoRA检查点路径
        """
        self.logger.info(f"📁 加载LoRA checkpoint: {checkpoint_path}")
        
        try:
            # 加载训练参数
            args = BaseArguments.from_pretrained(str(checkpoint_path))
            model_id = getattr(args, 'model', None)
            
            if not model_id:
                raise ValueError("❌ 无法获取基础模型路径")
            
            self.logger.info(f"🔧 基础模型: {model_id}")
            self.logger.info(f"🎨 模板类型: {getattr(args, 'template', 'default')}")
            
            # 记录模型信息
            self.model_info = {
                'base_model': model_id,
                'template_type': getattr(args, 'template', 'default'),
                'system_prompt': getattr(args, 'system', None),
                'checkpoint_path': str(checkpoint_path),
                'model_type': 'lora'
            }
            
            # 加载模型和分词器
            model, tokenizer = get_model_tokenizer(model_id)
            
            # 加载LoRA权重
            model = Swift.from_pretrained(model, str(checkpoint_path))
            
            # 创建模板
            template_type = getattr(args, 'template', 'default')
            system_prompt = getattr(args, 'system', None)
            template = get_template(template_type, tokenizer, default_system=system_prompt)
            
            # 创建推理引擎
            self.engine = PtEngine.from_model_template(model, template, max_batch_size=1)
            self.is_lora = True
            
            self.logger.info("✅ LoRA模型加载成功")
            
        except Exception as e:
            self.logger.error(f"❌ LoRA模型加载失败: {str(e)}")
            raise e
    
    def load_full_model(self, checkpoint_path: Path):
        """
        加载全参数模型
        Args:
            checkpoint_path: 全参数模型路径
        """
        self.logger.info(f"📁 加载全参数checkpoint: {checkpoint_path}")
        
        try:
            # 检查是否有训练参数文件
            args_file = checkpoint_path / "args.json"
            if args_file.exists():
                args = BaseArguments.from_pretrained(str(checkpoint_path))
                model_path = str(checkpoint_path)
                template_type = getattr(args, 'template', 'default')
                default_system = getattr(args, 'system', None)
            else:
                # 如果没有args.json，假设checkpoint就是模型路径
                model_path = str(checkpoint_path)
                template_type = 'default'
                default_system = None
            
            # 记录模型信息
            self.model_info = {
                'model_path': model_path,
                'template_type': template_type,
                'system_prompt': default_system,
                'checkpoint_path': str(checkpoint_path),
                'model_type': 'full'
            }
            
            # 加载模型和分词器
            model, tokenizer = get_model_tokenizer(model_path)
            
            # 创建模板
            template = get_template(template_type, tokenizer, default_system=default_system)
            
            # 创建推理引擎
            self.engine = PtEngine.from_model_template(model, template, max_batch_size=1)
            self.is_lora = False
            
            self.logger.info("✅ 全参数模型加载成功")
            
        except Exception as e:
            self.logger.error(f"❌ 全参数模型加载失败: {str(e)}")
            raise e
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        自动检测并加载模型
        Args:
            checkpoint_path: 模型检查点路径
        """
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            raise ValueError("❌ 模型检查点路径无效")
        
        self.logger.info("🚀 开始加载模型...")
        
        # 检测模型类型
        model_type = self.detect_model_type(self.checkpoint_path)
        self.logger.info(f"🔍 检测到模型类型: {model_type.upper()}")
        
        # 根据类型加载模型
        if model_type == "lora":
            self.load_lora_model(self.checkpoint_path)
        else:
            self.load_full_model(self.checkpoint_path)
    
    def extract_response_content(self, response) -> str:
        """
        从响应中提取内容
        Args:
            response: 推理响应对象
        Returns:
            str: 提取的文本内容
        """
        try:
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    return str(content) if content is not None else ""
        except (AttributeError, IndexError, TypeError) as e:
            self.logger.warning(f"⚠️  提取响应内容失败: {e}")
        return ""
    
    def infer_single(self, 
                    question: str, 
                    max_tokens: int = 512, 
                    temperature: float = 0.7,
                    top_p: float = 0.9) -> str:
        """
        单条推理
        Args:
            question: 输入问题
            max_tokens: 最大生成token数
            temperature: 生成温度
            top_p: 核采样概率
        Returns:
            str: 生成的回答
        """
        if not self.engine:
            raise ValueError("❌ 模型未加载，请先调用load_model()")
        
        try:
            # 创建请求配置
            request_config = RequestConfig(
                max_tokens=max_tokens, 
                temperature=temperature,
                top_p=top_p
            )
            
            # 创建推理请求
            infer_request = InferRequest(
                messages=[{'role': 'user', 'content': question}]
            )
            
            # 执行推理
            resp_list = self.engine.infer([infer_request], request_config)
            
            if resp_list and len(resp_list) > 0:
                return self.extract_response_content(resp_list[0])
            
            return ""
            
        except Exception as e:
            self.logger.error(f"❌ 推理失败: {str(e)}")
            raise e
    
    def infer_batch(self, 
                   questions: List[str], 
                   max_tokens: int = 512, 
                   temperature: float = 0.7) -> List[str]:
        """
        批量推理
        Args:
            questions: 问题列表
            max_tokens: 最大生成token数
            temperature: 生成温度
        Returns:
            List[str]: 回答列表
        """
        if not self.engine:
            raise ValueError("❌ 模型未加载，请先调用load_model()")
        
        self.logger.info(f"🔄 开始批量推理，共{len(questions)}个问题")
        
        results = []
        for i, question in enumerate(questions, 1):
            try:
                self.logger.info(f"📝 处理问题 {i}/{len(questions)}")
                response = self.infer_single(question, max_tokens, temperature)
                results.append(response)
            except Exception as e:
                self.logger.error(f"❌ 问题 {i} 推理失败: {str(e)}")
                results.append("")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        return self.model_info.copy()
    
    def interactive_chat(self):
        """交互式对话"""
        if not self.engine:
            raise ValueError("❌ 模型未加载，请先调用load_model()")
        
        self.logger.info("🎮 进入交互式对话模式 (输入 'quit' 退出)")
        print("\n" + "="*50)
        print("🤖 AI助手已就绪，开始对话吧！")
        print("💡 输入 'quit', 'exit' 或 '退出' 结束对话")
        print("="*50)
        
        conversation_history = []
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您: ").strip()
                
                # 检查退出命令
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                # 记录对话历史
                conversation_history.append(f"用户: {user_input}")
                
                # 执行推理
                print("🤖 AI: ", end="", flush=True)
                response = self.infer_single(user_input)
                print(response)
                
                # 记录AI回复
                conversation_history.append(f"AI: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 对话已中断，再见！")
                break
            except Exception as e:
                self.logger.error(f"❌ 对话过程中出错: {str(e)}")
                print(f"😓 抱歉，出现了一些问题: {str(e)}")

class InferencePresets:
    """推理预设配置"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认推理配置"""
        return {
            'max_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
        }
    
    @staticmethod
    def get_creative_config() -> Dict[str, Any]:
        """获取创意生成配置"""
        return {
            'max_tokens': 1024,
            'temperature': 0.9,
            'top_p': 0.95,
        }
    
    @staticmethod
    def get_precise_config() -> Dict[str, Any]:
        """获取精确回答配置"""
        return {
            'max_tokens': 256,
            'temperature': 0.1,
            'top_p': 0.8,
        }

def create_inference_manager(checkpoint_path: Optional[str] = None, 
                           config_path: Optional[str] = None) -> InferenceManager:
    """
    创建推理管理器实例
    Args:
        checkpoint_path: 模型检查点路径
        config_path: 配置文件路径
    Returns:
        InferenceManager: 推理管理器实例
    """
    return InferenceManager(checkpoint_path, config_path)

if __name__ == "__main__":
    # 示例用法
    print("🧪 推理器测试...")
    
    # 创建推理管理器
    inference_manager = create_inference_manager()
    
    # 显示预设配置
    configs = {
        "默认配置": InferencePresets.get_default_config(),
        "创意配置": InferencePresets.get_creative_config(),
        "精确配置": InferencePresets.get_precise_config(),
    }
    
    print("🔧 可用的推理配置:")
    for name, config in configs.items():
        print(f"  {name}: {config}")
    
    print("✅ 推理器模块加载成功")