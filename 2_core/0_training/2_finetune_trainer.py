#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心训练模块 - 基于MS-Swift的微调训练器
支持LoRA、QLoRA、全参数微调、多模态微调
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
    from swift.llm import sft_main
    from swift.utils import get_logger as swift_get_logger
except ImportError as e:
    print(f"❌ 导入Swift模块失败: {e}")
    print("请确保已安装ms-swift: pip install ms-swift")
    sys.exit(1)

import torch

class TrainingManager:
    """训练管理器 - 统一的微调训练接口"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化训练管理器
        Args:
            config_path: 配置文件路径
        """
        self.logger = Logger("TrainingManager").get_logger()
        self.config_loader = ConfigLoader()
        
        # 加载配置
        if config_path:
            self.config = self.config_loader.load_config(config_path)
        else:
            self.config = {}
            
        self.swift_logger = swift_get_logger()
        self.setup_environment()
    
    def setup_environment(self):
        """设置训练环境"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self.logger.info(f"🔧 检测到 {device_count} 张GPU")
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"  GPU {i}: {gpu_name} ({memory:.1f}GB)")
        else:
            self.logger.warning("⚠️  未检测到可用GPU，将使用CPU训练（不推荐）")
    
    def prepare_training_args(self, train_config: Dict[str, Any]) -> List[str]:
        """
        准备训练参数
        Args:
            train_config: 训练配置字典
        Returns:
            List[str]: 命令行参数列表
        """
        model = train_config.get('model', 'Qwen/Qwen2.5-7B-Instruct')
        dataset = train_config.get('dataset', ['AI-ModelScope/alpaca-gpt4-data-zh#1000'])
        output_dir = train_config.get('output_dir', './output')
        train_type = train_config.get('train_type', 'lora')
        
        # 基础参数
        args = [
            '--model', model,
            '--dataset', ' '.join(dataset if isinstance(dataset, list) else [dataset]),
            '--train_type', train_type,
            '--output_dir', output_dir,
            '--num_train_epochs', str(train_config.get('num_train_epochs', 3)),
            '--per_device_train_batch_size', str(train_config.get('per_device_train_batch_size', 2)),
            '--gradient_accumulation_steps', str(train_config.get('gradient_accumulation_steps', 8)),
            '--learning_rate', str(train_config.get('learning_rate', 1e-4)),
            '--max_length', str(train_config.get('max_length', 2048)),
            '--save_steps', str(train_config.get('save_steps', 100)),
            '--eval_steps', str(train_config.get('eval_steps', 100)),
            '--logging_steps', str(train_config.get('logging_steps', 10)),
            '--warmup_ratio', str(train_config.get('warmup_ratio', 0.03)),
            '--weight_decay', str(train_config.get('weight_decay', 0.1)),
        ]
        
        # LoRA相关参数
        if train_type in ['lora', 'qlora']:
            args.extend([
                '--lora_rank', str(train_config.get('lora_rank', 8)),
                '--lora_alpha', str(train_config.get('lora_alpha', 32)),
                '--lora_dropout', str(train_config.get('lora_dropout', 0.05)),
            ])
        
        # 梯度检查点
        if train_config.get('gradient_checkpointing', True):
            args.append('--gradient_checkpointing')
        
        # DeepSpeed配置（全参数微调）
        if train_type == 'full':
            deepspeed = train_config.get('deepspeed', 'zero2')
            if deepspeed:
                args.extend(['--deepspeed', deepspeed])
        
        # 多模态特殊配置
        if train_type == 'multimodal':
            args.extend([
                '--lora_rank', str(train_config.get('lora_rank', 16)),
                '--lora_alpha', str(train_config.get('lora_alpha', 32)),
            ])
        
        return args
    
    def execute_training(self, train_config: Dict[str, Any]) -> str:
        """
        执行训练
        Args:
            train_config: 训练配置
        Returns:
            str: 输出目录路径
        """
        train_type = train_config.get('train_type', 'lora')
        output_dir = train_config.get('output_dir', './output')
        
        self.logger.info(f"🚀 开始{train_type.upper()}微调训练...")
        self.logger.info(f"📁 模型: {train_config.get('model')}")
        self.logger.info(f"📊 数据集: {train_config.get('dataset')}")
        self.logger.info(f"💾 输出目录: {output_dir}")
        
        # 准备训练参数
        train_args = self.prepare_training_args(train_config)
        
        # 备份原始sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # 设置命令行参数
            sys.argv = ['sft_main'] + train_args
            
            # 执行训练
            self.logger.info("⏳ 开始执行训练...")
            result = sft_main()
            
            self.logger.info(f"✅ {train_type.upper()}微调完成")
            return str(output_dir)
            
        except Exception as e:
            self.logger.error(f"❌ 训练失败: {str(e)}")
            raise e
        finally:
            # 恢复原始参数
            sys.argv = original_argv
    
    def train_lora(self, **kwargs) -> str:
        """LoRA微调"""
        config = {**self.config, **kwargs, 'train_type': 'lora'}
        return self.execute_training(config)
    
    def train_qlora(self, **kwargs) -> str:
        """QLoRA微调"""
        config = {**self.config, **kwargs, 'train_type': 'qlora'}
        # QLoRA的默认配置
        config.setdefault('per_device_train_batch_size', 1)
        config.setdefault('gradient_accumulation_steps', 16)
        return self.execute_training(config)
    
    def train_full_params(self, **kwargs) -> str:
        """全参数微调"""
        config = {**self.config, **kwargs, 'train_type': 'full'}
        # 全参数微调的默认配置
        config.setdefault('per_device_train_batch_size', 1)
        config.setdefault('gradient_accumulation_steps', 32)
        config.setdefault('learning_rate', 5e-5)
        return self.execute_training(config)
    
    def train_multimodal(self, **kwargs) -> str:
        """多模态微调"""
        config = {**self.config, **kwargs, 'train_type': 'multimodal'}
        # 多模态微调的默认配置
        config.setdefault('model', 'Qwen/Qwen2.5-VL-7B-Instruct')
        config.setdefault('dataset', ['AI-ModelScope/coco-en-2-zh#20000'])
        config.setdefault('lora_rank', 16)
        return self.execute_training(config)
    
    def run_training_from_config(self) -> str:
        """从配置文件运行训练"""
        if not self.config:
            raise ValueError("❌ 未提供配置信息")
        
        train_type = self.config.get('train_type', 'lora')
        self.logger.info(f"📋 从配置运行{train_type.upper()}训练")
        
        return self.execute_training(self.config)

class TrainingPresets:
    """训练预设配置"""
    
    @staticmethod
    def get_lora_preset(model: str = "Qwen/Qwen2.5-7B-Instruct") -> Dict[str, Any]:
        """获取LoRA训练预设"""
        return {
            'model': model,
            'train_type': 'lora',
            'dataset': ['AI-ModelScope/alpaca-gpt4-data-zh#1000'],
            'num_train_epochs': 3,
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 8,
            'learning_rate': 1e-4,
            'lora_rank': 8,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'max_length': 2048,
            'save_steps': 100,
            'logging_steps': 10,
            'warmup_ratio': 0.03,
            'weight_decay': 0.1,
            'gradient_checkpointing': True,
        }
    
    @staticmethod
    def get_qlora_preset(model: str = "Qwen/Qwen2.5-7B-Instruct") -> Dict[str, Any]:
        """获取QLoRA训练预设"""
        preset = TrainingPresets.get_lora_preset(model)
        preset.update({
            'train_type': 'qlora',
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 16,
        })
        return preset
    
    @staticmethod
    def get_full_preset(model: str = "Qwen/Qwen2.5-7B-Instruct") -> Dict[str, Any]:
        """获取全参数训练预设"""
        return {
            'model': model,
            'train_type': 'full',
            'dataset': ['AI-ModelScope/alpaca-gpt4-data-zh#1000'],
            'num_train_epochs': 3,
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 32,
            'learning_rate': 5e-5,
            'max_length': 2048,
            'save_steps': 100,
            'logging_steps': 10,
            'warmup_ratio': 0.03,
            'weight_decay': 0.1,
            'gradient_checkpointing': True,
            'deepspeed': 'zero2',
        }
    
    @staticmethod
    def get_multimodal_preset(model: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> Dict[str, Any]:
        """获取多模态训练预设"""
        return {
            'model': model,
            'train_type': 'multimodal',
            'dataset': ['AI-ModelScope/coco-en-2-zh#20000'],
            'num_train_epochs': 3,
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,
            'learning_rate': 1e-4,
            'lora_rank': 16,
            'lora_alpha': 32,
            'max_length': 2048,
            'save_steps': 100,
            'logging_steps': 10,
        }

def create_trainer(config_path: Optional[str] = None) -> TrainingManager:
    """
    创建训练器实例
    Args:
        config_path: 配置文件路径
    Returns:
        TrainingManager: 训练管理器实例
    """
    return TrainingManager(config_path)

if __name__ == "__main__":
    # 示例用法
    print("🧪 训练器测试...")
    
    # 创建训练器
    trainer = create_trainer()
    
    # 使用预设配置进行LoRA训练
    lora_config = TrainingPresets.get_lora_preset()
    lora_config['output_dir'] = './test_output'
    lora_config['dataset'] = ['AI-ModelScope/alpaca-gpt4-data-zh#100']  # 小数据集用于测试
    
    print("🔧 LoRA训练配置:")
    for key, value in lora_config.items():
        print(f"  {key}: {value}")
    
    # 注意：这里不实际运行训练，只是展示配置
    print("✅ 训练器模块加载成功")