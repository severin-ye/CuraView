#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器模块 - 统一的配置文件管理
支持JSON配置文件的加载、合并和路径解析
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, base_path: str = "./0_configs"):
        """
        初始化配置加载器
        Args:
            base_path: 配置文件基础路径
        """
        self.base_path = Path(base_path)
        self._config_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_config(self, config_file: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        加载单个配置文件
        Args:
            config_file: 配置文件名或路径
            use_cache: 是否使用缓存
        Returns:
            配置字典
        """
        if use_cache and config_file in self._config_cache:
            return self._config_cache[config_file].copy()
        
        # 处理路径
        if not config_file.endswith('.json'):
            config_file += '.json'
        
        config_path = self.base_path / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 解析路径
            config = self._resolve_paths(config)
            
            if use_cache:
                self._config_cache[config_file] = config.copy()
            
            logger.info(f"已加载配置文件: {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {config_path}, 错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {config_path}, 错误: {e}")
    
    def load_train_config(self) -> Dict[str, Any]:
        """加载训练配置"""
        return self.load_config("0_train_config")
    
    def load_model_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        return self.load_config("1_model_config")
    
    def load_deploy_config(self) -> Dict[str, Any]:
        """加载部署配置"""
        return self.load_config("2_deploy_config")
    
    def load_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        加载Agent配置
        Args:
            agent_name: Agent名称 (rag, preference, medical等)
        """
        agent_configs = {
            'rag': 'agents/0_rag_agent',
            'preference': 'agents/1_preference_agent', 
            'medical': 'agents/2_medical_agent'
        }
        
        config_file = agent_configs.get(agent_name)
        if not config_file:
            raise ValueError(f"未知的Agent类型: {agent_name}")
        
        return self.load_config(config_file)
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并多个配置字典
        Args:
            *configs: 要合并的配置字典列表
        Returns:
            合并后的配置
        """
        merged = {}
        for config in configs:
            merged = self._deep_merge(merged, config)
        return merged
    
    def save_config(self, config: Dict[str, Any], config_file: str):
        """
        保存配置到文件
        Args:
            config: 配置字典
            config_file: 配置文件名
        """
        if not config_file.endswith('.json'):
            config_file += '.json'
        
        config_path = self.base_path / config_file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"配置已保存到: {config_path}")
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析配置中的相对路径
        Args:
            config: 配置字典
        Returns:
            解析后的配置
        """
        def resolve_value(value):
            if isinstance(value, str) and value.startswith('./'):
                # 相对路径解析
                return str(Path(value).resolve())
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value
        
        return resolve_value(config)
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典
        Args:
            dict1: 基础字典
            dict2: 要合并的字典
        Returns:
            合并后的字典
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_available_configs(self) -> Dict[str, List[str]]:
        """
        获取所有可用的配置文件
        Returns:
            配置文件列表字典
        """
        configs = {
            'main': [],
            'agents': []
        }
        
        # 主配置文件
        for config_file in self.base_path.glob('*.json'):
            configs['main'].append(config_file.stem)
        
        # Agent配置文件
        agents_path = self.base_path / 'agents'
        if agents_path.exists():
            for config_file in agents_path.glob('*.json'):
                configs['agents'].append(config_file.stem)
        
        return configs
    
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        验证配置文件格式
        Args:
            config: 要验证的配置
            schema: 配置模式
        Returns:
            是否有效
        """
        # 简单的配置验证逻辑
        try:
            for key, expected_type in schema.items():
                if key in config:
                    if not isinstance(config[key], expected_type):
                        logger.error(f"配置项 {key} 类型错误，期望 {expected_type}")
                        return False
                else:
                    logger.warning(f"缺少配置项: {key}")
            
            return True
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

# 全局配置加载器实例
config_loader = ConfigLoader()

def load_config(config_name: str) -> Dict[str, Any]:
    """便捷的配置加载函数"""
    return config_loader.load_config(config_name)

def get_train_config() -> Dict[str, Any]:
    """获取训练配置"""
    return config_loader.load_train_config()

def get_model_config() -> Dict[str, Any]:
    """获取模型配置"""
    return config_loader.load_model_config()

def get_deploy_config() -> Dict[str, Any]:
    """获取部署配置"""
    return config_loader.load_deploy_config()

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """获取Agent配置"""
    return config_loader.load_agent_config(agent_name)

if __name__ == "__main__":
    # 测试代码
    loader = ConfigLoader()
    
    try:
        train_config = loader.load_train_config()
        print("训练配置加载成功:", train_config.get('model_name', 'Unknown'))
        
        model_config = loader.load_model_config()
        print("模型配置加载成功:", model_config.get('model_id', 'Unknown'))
        
        available = loader.get_available_configs()
        print("可用配置:", available)
        
    except Exception as e:
        print(f"测试失败: {e}")