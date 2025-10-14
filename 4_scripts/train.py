#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练脚本 - 基于新架构的模型训练入口
支持LoRA、QLoRA、全参数微调、多模态微调
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "2_core"))

from logger import Logger
from config_loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description="统一训练脚本", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # 基础参数
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="基础模型路径")
    parser.add_argument("--dataset", type=str, nargs='+', help="训练数据集")
    parser.add_argument("--output_dir", type=str, default="./6_output", help="输出目录")
    
    # 训练类型
    parser.add_argument("--train_type", choices=['lora', 'qlora', 'full', 'multimodal'], 
                        default='lora', help="训练类型")
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="每设备批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 其他参数
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    
    # DeepSpeed参数（全参数微调）
    parser.add_argument("--deepspeed", type=str, default="zero2", help="DeepSpeed配置")
    
    # 预设配置
    parser.add_argument("--preset", choices=['lora', 'qlora', 'full', 'multimodal'], 
                        help="使用预设配置")
    
    # 验证模式
    parser.add_argument("--dry_run", action="store_true", help="只验证配置，不执行训练")
    
    args = parser.parse_args()
    
    # 初始化日志
    logger = Logger("TrainScript").get_logger()
    logger.info("🚀 启动统一训练脚本")
    
    try:
        # 导入核心API
        from training.trainer import TrainingManager, TrainingPresets
        
        # 创建训练器
        trainer = TrainingManager(args.config)
        
        # 构建训练配置
        if args.preset:
            # 使用预设配置
            logger.info(f"📋 使用预设配置: {args.preset}")
            presets = {
                'lora': TrainingPresets.get_lora_preset,
                'qlora': TrainingPresets.get_qlora_preset,
                'full': TrainingPresets.get_full_preset,
                'multimodal': TrainingPresets.get_multimodal_preset
            }
            train_config = presets[args.preset](args.model)
            
            # 应用命令行覆盖
            if args.dataset:
                train_config['dataset'] = args.dataset
            if args.output_dir != "./6_output":
                train_config['output_dir'] = args.output_dir
        else:
            # 从命令行参数构建配置
            train_config = {
                'model': args.model,
                'dataset': args.dataset or ['AI-ModelScope/alpaca-gpt4-data-zh#1000'],
                'train_type': args.train_type,
                'output_dir': args.output_dir,
                'num_train_epochs': args.num_train_epochs,
                'per_device_train_batch_size': args.per_device_train_batch_size,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'learning_rate': args.learning_rate,
                'max_length': args.max_length,
                'lora_rank': args.lora_rank,
                'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout,
                'save_steps': args.save_steps,
                'logging_steps': args.logging_steps,
                'warmup_ratio': args.warmup_ratio,
                'weight_decay': args.weight_decay,
                'gradient_checkpointing': args.gradient_checkpointing,
                'deepspeed': args.deepspeed if args.train_type == 'full' else None
            }
        
        # 显示配置信息
        logger.info("📊 训练配置:")
        for key, value in train_config.items():
            logger.info(f"  {key}: {value}")
        
        if args.dry_run:
            logger.info("🔍 验证模式，配置检查完成")
            return
        
        # 执行训练
        logger.info("⏳ 开始训练...")
        result = trainer.execute_training(train_config)
        
        logger.info(f"✅ 训练完成！结果保存在: {result}")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()