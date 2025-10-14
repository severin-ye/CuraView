"""
数据加载与预处理模块

负责数据集的加载、预处理和批处理功能。
"""

import json
import os
from typing import List, Dict, Any, Optional, Iterator, Union
from datasets import Dataset, load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """数据集处理器，负责数据的加载和预处理"""
    
    def __init__(self, tokenizer_path: str, max_length: int = 2048):
        """
        初始化数据集处理器
        
        Args:
            tokenizer_path: tokenizer路径
            max_length: 最大序列长度
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        
        # 确保有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_json_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        加载JSON格式的数据集
        
        Args:
            dataset_path: 数据集文件路径
            
        Returns:
            数据集列表
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载数据集: {dataset_path}, 样本数: {len(data)}")
            return data
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def validate_dataset(self, data: List[Dict[str, Any]]) -> bool:
        """
        验证数据集格式
        
        Args:
            data: 数据集
            
        Returns:
            是否验证通过
        """
        required_keys = ["input", "output"]
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.error(f"样本 {i} 不是字典格式")
                return False
            
            for key in required_keys:
                if key not in item:
                    logger.error(f"样本 {i} 缺少必需字段: {key}")
                    return False
                    
            if not isinstance(item["input"], str) or not isinstance(item["output"], str):
                logger.error(f"样本 {i} 的input或output不是字符串")
                return False
        
        logger.info(f"数据集验证通过，共 {len(data)} 个样本")
        return True
    
    def format_conversation(self, item: Dict[str, Any]) -> str:
        """
        格式化对话数据
        
        Args:
            item: 单个数据样本
            
        Returns:
            格式化后的文本
        """
        system = item.get("system", "")
        input_text = item["input"]
        output_text = item["output"]
        
        if system:
            formatted = f"<|im_start|>system\n{system}<|im_end|>\n"
        else:
            formatted = ""
            
        formatted += f"<|im_start|>user\n{input_text}<|im_end|>\n"
        formatted += f"<|im_start|>assistant\n{output_text}<|im_end|>"
        
        return formatted
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        tokenize函数
        
        Args:
            examples: 批量样本
            
        Returns:
            tokenize后的结果
        """
        # 格式化文本
        formatted_texts = []
        for i in range(len(examples["input"])):
            item = {
                "system": examples.get("system", [""] * len(examples["input"]))[i],
                "input": examples["input"][i],
                "output": examples["output"][i]
            }
            formatted_texts.append(self.format_conversation(item))
        
        # tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # 设置labels（用于计算loss）
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, dataset_path: str, batch_size: int = 8) -> DataLoader:
        """
        准备数据集用于训练
        
        Args:
            dataset_path: 数据集路径
            batch_size: 批大小
            
        Returns:
            DataLoader对象
        """
        # 加载原始数据
        raw_data = self.load_json_dataset(dataset_path)
        
        # 验证数据格式
        if not self.validate_dataset(raw_data):
            raise ValueError("数据集格式验证失败")
        
        # 转换为HuggingFace Dataset
        dataset = Dataset.from_list(raw_data)
        
        # 应用tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 设置格式
        tokenized_dataset.set_format(type="torch")
        
        # 创建DataLoader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"数据集准备完成，批大小: {batch_size}")
        return dataloader
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        批处理函数
        
        Args:
            batch: 批数据
            
        Returns:
            处理后的批数据
        """
        # 获取所有keys
        keys = batch[0].keys()
        
        # 对每个key进行padding
        result = {}
        for key in keys:
            sequences = [item[key] for item in batch]
            
            if key in ["input_ids", "attention_mask", "labels"]:
                # padding到批内最大长度
                max_len = max(len(seq) for seq in sequences)
                padded = []
                
                for seq in sequences:
                    if len(seq) < max_len:
                        if key == "input_ids" or key == "labels":
                            pad_value = self.tokenizer.pad_token_id
                        else:  # attention_mask
                            pad_value = 0
                        
                        padded_seq = seq.tolist() + [pad_value] * (max_len - len(seq))
                        padded.append(torch.tensor(padded_seq))
                    else:
                        padded.append(seq)
                
                result[key] = torch.stack(padded)
            else:
                result[key] = torch.stack(sequences)
        
        return result


class MultiDatasetLoader:
    """多数据集加载器"""
    
    def __init__(self, tokenizer_path: str, max_length: int = 2048):
        """初始化多数据集加载器"""
        self.processor = DatasetProcessor(tokenizer_path, max_length)
        self.datasets = {}
    
    def add_dataset(self, name: str, path: str, weight: float = 1.0):
        """
        添加数据集
        
        Args:
            name: 数据集名称
            path: 数据集路径
            weight: 数据集权重
        """
        data = self.processor.load_json_dataset(path)
        self.datasets[name] = {
            "data": data,
            "weight": weight,
            "path": path
        }
        logger.info(f"添加数据集: {name}, 样本数: {len(data)}, 权重: {weight}")
    
    def create_mixed_dataset(self, total_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        创建混合数据集
        
        Args:
            total_samples: 总样本数，None表示使用所有样本
            
        Returns:
            混合后的数据集
        """
        if not self.datasets:
            raise ValueError("没有添加任何数据集")
        
        # 计算每个数据集的采样数量
        total_weight = sum(info["weight"] for info in self.datasets.values())
        mixed_data = []
        
        for name, info in self.datasets.items():
            data = info["data"]
            weight = info["weight"]
            
            if total_samples:
                sample_count = int(total_samples * weight / total_weight)
                sample_count = min(sample_count, len(data))
            else:
                sample_count = len(data)
            
            # 采样数据
            if sample_count < len(data):
                import random
                sampled_data = random.sample(data, sample_count)
            else:
                sampled_data = data.copy()
            
            mixed_data.extend(sampled_data)
            logger.info(f"从数据集 {name} 采样 {len(sampled_data)} 个样本")
        
        # 打乱混合数据
        import random
        random.shuffle(mixed_data)
        
        logger.info(f"混合数据集创建完成，总样本数: {len(mixed_data)}")
        return mixed_data
    
    def create_dataloader(self, batch_size: int = 8, total_samples: Optional[int] = None) -> DataLoader:
        """
        创建混合数据集的DataLoader
        
        Args:
            batch_size: 批大小
            total_samples: 总样本数
            
        Returns:
            DataLoader对象
        """
        mixed_data = self.create_mixed_dataset(total_samples)
        
        # 验证数据格式
        if not self.processor.validate_dataset(mixed_data):
            raise ValueError("混合数据集格式验证失败")
        
        # 转换为HuggingFace Dataset
        dataset = Dataset.from_list(mixed_data)
        
        # 应用tokenize
        tokenized_dataset = dataset.map(
            self.processor.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 设置格式
        tokenized_dataset.set_format(type="torch")
        
        # 创建DataLoader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.processor._collate_fn
        )
        
        return dataloader


# 便捷函数
def create_dataloader(
    dataset_path: str,
    tokenizer_path: str,
    batch_size: int = 8,
    max_length: int = 2048
) -> DataLoader:
    """
    创建数据加载器的便捷函数
    
    Args:
        dataset_path: 数据集路径
        tokenizer_path: tokenizer路径
        batch_size: 批大小
        max_length: 最大序列长度
        
    Returns:
        DataLoader对象
    """
    processor = DatasetProcessor(tokenizer_path, max_length)
    return processor.prepare_dataset(dataset_path, batch_size)


def validate_dataset_file(dataset_path: str) -> bool:
    """
    验证数据集文件的便捷函数
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        是否验证通过
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processor = DatasetProcessor("", 2048)  # 临时处理器
        return processor.validate_dataset(data)
    except Exception as e:
        logger.error(f"验证数据集文件失败: {e}")
        return False