#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MS-Swift微调训练脚本（修复版）
支持LoRA、QLoRA、全参数微调等多种训练方式
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

try:
    from swift.llm import sft_main
    from swift.utils import get_logger
except ImportError as e:
    print(f"导入Swift模块失败: {e}")
    print("请确保已安装ms-swift: pip install ms-swift")
    sys.exit(1)

import torch

logger = get_logger()

class FineTuner:
    """微调训练器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = {}
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
            
        self.setup_environment()
    
    def load_config(self, config_file: str):
        """加载配置文件"""
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def setup_environment(self):
        """设置训练环境"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"检测到 {device_count} 张GPU")
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  GPU {i}: {gpu_name} ({memory:.1f}GB)")
        else:
            logger.warning("未检测到可用GPU，将使用CPU训练（不推荐）")
    
    def train_lora(self, 
                   model: str = "Qwen/Qwen2.5-7B-Instruct",
                   dataset: Optional[List[str]] = None,
                   output_dir: str = "./output",
                   **kwargs) -> str:
        """LoRA微调"""
        
        if dataset is None:
            dataset = ["AI-ModelScope/alpaca-gpt4-data-zh#1000"]
            
        logger.info("开始LoRA微调训练...")
        logger.info(f"模型: {model}")
        logger.info(f"数据集: {dataset}")
        logger.info(f"输出目录: {output_dir}")
        
        # 构建训练参数
        import sys
        
        # 构建命令行参数
        train_args = [
            '--model', model,
            '--dataset', ' '.join(dataset),
            '--train_type', 'lora',
            '--output_dir', output_dir,
            '--lora_rank', str(kwargs.get('lora_rank', 8)),
            '--lora_alpha', str(kwargs.get('lora_alpha', 32)),
            '--lora_dropout', str(kwargs.get('lora_dropout', 0.05)),
            '--num_train_epochs', str(kwargs.get('num_train_epochs', 3)),
            '--per_device_train_batch_size', str(kwargs.get('per_device_train_batch_size', 2)),
            '--gradient_accumulation_steps', str(kwargs.get('gradient_accumulation_steps', 8)),
            '--learning_rate', str(kwargs.get('learning_rate', 1e-4)),
            '--max_length', str(kwargs.get('max_length', 2048)),
            '--save_steps', str(kwargs.get('save_steps', 100)),
            '--eval_steps', str(kwargs.get('eval_steps', 100)),
            '--logging_steps', str(kwargs.get('logging_steps', 10)),
            '--warmup_ratio', str(kwargs.get('warmup_ratio', 0.03)),
            '--weight_decay', str(kwargs.get('weight_decay', 0.1)),
        ]
        
        if kwargs.get('gradient_checkpointing', True):
            train_args.append('--gradient_checkpointing')
        
        # 备份原始sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # 设置命令行参数
            sys.argv = ['sft_main'] + train_args
            
            # 执行训练
            result = sft_main()
        finally:
            # 恢复原始参数
            sys.argv = original_argv
        
        logger.info("LoRA微调完成")
        return str(output_dir)
    
    def train_qlora(self, 
                    model: str = "Qwen/Qwen2.5-7B-Instruct",
                    dataset: Optional[List[str]] = None,
                    output_dir: str = "./output",
                    **kwargs) -> str:
        """QLoRA微调"""
        
        if dataset is None:
            dataset = ["AI-ModelScope/alpaca-gpt4-data-zh#1000"]
            
        logger.info("开始QLoRA微调训练...")
        
        import sys
        
        # 构建命令行参数
        train_args = [
            '--model', model,
            '--dataset', ' '.join(dataset),
            '--train_type', 'qlora',
            '--output_dir', output_dir,
            '--lora_rank', str(kwargs.get('lora_rank', 8)),
            '--lora_alpha', str(kwargs.get('lora_alpha', 32)),
            '--num_train_epochs', str(kwargs.get('num_train_epochs', 3)),
            '--per_device_train_batch_size', str(kwargs.get('per_device_train_batch_size', 1)),
            '--gradient_accumulation_steps', str(kwargs.get('gradient_accumulation_steps', 16)),
            '--learning_rate', str(kwargs.get('learning_rate', 1e-4)),
            '--max_length', str(kwargs.get('max_length', 2048)),
            '--save_steps', str(kwargs.get('save_steps', 100)),
            '--gradient_checkpointing',
        ]
        
        # 备份原始sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # 设置命令行参数
            sys.argv = ['sft_main'] + train_args
            
            # 执行训练
            result = sft_main()
        finally:
            # 恢复原始参数
            sys.argv = original_argv
        
        logger.info("QLoRA微调完成")
        return str(output_dir)
    
    def train_full_params(self, 
                         model: str = "Qwen/Qwen2.5-7B-Instruct",
                         dataset: Optional[List[str]] = None,
                         output_dir: str = "./output",
                         **kwargs) -> str:
        """全参数微调"""
        
        if dataset is None:
            dataset = ["AI-ModelScope/alpaca-gpt4-data-zh#1000"]
            
        logger.info("开始全参数微调训练...")
        
        import sys
        
        # 构建命令行参数
        train_args = [
            '--model', model,
            '--dataset', ' '.join(dataset),
            '--train_type', 'full',
            '--output_dir', output_dir,
            '--num_train_epochs', str(kwargs.get('num_train_epochs', 3)),
            '--per_device_train_batch_size', str(kwargs.get('per_device_train_batch_size', 1)),
            '--gradient_accumulation_steps', str(kwargs.get('gradient_accumulation_steps', 32)),
            '--learning_rate', str(kwargs.get('learning_rate', 5e-5)),
            '--max_length', str(kwargs.get('max_length', 2048)),
            '--save_steps', str(kwargs.get('save_steps', 100)),
            '--gradient_checkpointing',
        ]
        
        deepspeed = kwargs.get('deepspeed', 'zero2')
        if deepspeed:
            train_args.extend(['--deepspeed', deepspeed])
        
        # 备份原始sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # 设置命令行参数
            sys.argv = ['sft_main'] + train_args
            
            # 执行训练
            result = sft_main()
        finally:
            # 恢复原始参数
            sys.argv = original_argv
        
        logger.info("全参数微调完成")
        return str(output_dir)
    
    def train_multimodal(self, 
                        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                        dataset: Optional[List[str]] = None,
                        output_dir: str = "./output",
                        **kwargs) -> str:
        """多模态微调"""
        
        if dataset is None:
            dataset = ["AI-ModelScope/coco-en-2-zh#20000"]
            
        logger.info("开始多模态微调训练...")
        
        import sys
        
        # 构建命令行参数
        train_args = [
            '--model', model,
            '--dataset', ' '.join(dataset),
            '--train_type', 'lora',
            '--output_dir', output_dir,
            '--lora_rank', str(kwargs.get('lora_rank', 16)),
            '--lora_alpha', str(kwargs.get('lora_alpha', 32)),
            '--num_train_epochs', str(kwargs.get('num_train_epochs', 3)),
            '--per_device_train_batch_size', str(kwargs.get('per_device_train_batch_size', 1)),
            '--gradient_accumulation_steps', str(kwargs.get('gradient_accumulation_steps', 8)),
            '--learning_rate', str(kwargs.get('learning_rate', 1e-4)),
            '--max_length', str(kwargs.get('max_length', 2048)),
            '--save_steps', str(kwargs.get('save_steps', 100)),
        ]
        
        # 备份原始sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # 设置命令行参数
            sys.argv = ['sft_main'] + train_args
            
            # 执行训练
            result = sft_main()
        finally:
            # 恢复原始参数
            sys.argv = original_argv
        
        logger.info("多模态微调完成")
        return str(output_dir)
    
    def run_training(self) -> str:
        """根据配置运行训练"""
        if not self.config:
            raise ValueError("未提供配置信息")
        
        train_type = self.config.get('train_type', 'lora')
        
        # 移除train_type，剩余参数传递给训练函数
        train_config = self.config.copy()
        train_config.pop('train_type', None)
        
        if train_type == 'lora':
            return self.train_lora(**train_config)
        elif train_type == 'qlora':
            return self.train_qlora(**train_config)
        elif train_type == 'full':
            return self.train_full_params(**train_config)
        elif train_type == 'multimodal':
            return self.train_multimodal(**train_config)
        else:
            raise ValueError(f"不支持的训练类型: {train_type}")

def create_training_args(**kwargs) -> Dict[str, Any]:
    """创建训练参数字典"""
    return {
        'model': kwargs.get('model', 'Qwen/Qwen2.5-7B-Instruct'),
        'dataset': kwargs.get('dataset'),
        'train_type': kwargs.get('train_type', 'lora'),
        'output_dir': kwargs.get('output_dir', './output'),
        'num_train_epochs': kwargs.get('num_train_epochs', 3),
        'per_device_train_batch_size': kwargs.get('per_device_train_batch_size', 2),
        'gradient_accumulation_steps': kwargs.get('gradient_accumulation_steps', 8),
        'learning_rate': kwargs.get('learning_rate', 1e-4),
        'max_length': kwargs.get('max_length', 2048),
        'lora_rank': kwargs.get('lora_rank', 8),
        'lora_alpha': kwargs.get('lora_alpha', 32),
        'save_steps': kwargs.get('save_steps', 100),
        'logging_steps': kwargs.get('logging_steps', 10),
    }

def main():
    parser = argparse.ArgumentParser(description="MS-Swift微调训练脚本")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="基础模型")
    parser.add_argument("--dataset", type=str, nargs='+', help="训练数据集")
    parser.add_argument("--train_type", choices=['lora', 'qlora', 'full', 'multimodal'], 
                        default='lora', help="训练类型")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每设备批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    
    args = parser.parse_args()
    
    # 创建训练器
    if args.config:
        trainer = FineTuner(args.config)
        result = trainer.run_training()
    else:
        trainer = FineTuner()
        
        # 从命令行参数创建配置
        train_args = create_training_args(
            model=args.model,
            dataset=args.dataset,
            train_type=args.train_type,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            max_length=args.max_length,
            save_steps=args.save_steps,
        )
        
        if args.train_type == 'lora':
            result = trainer.train_lora(**train_args)
        elif args.train_type == 'qlora':
            result = trainer.train_qlora(**train_args)
        elif args.train_type == 'full':
            result = trainer.train_full_params(**train_args)
        elif args.train_type == 'multimodal':
            result = trainer.train_multimodal(**train_args)
    
    print(f"✅ 训练完成，结果保存在: {result}")

if __name__ == "__main__":
    main()