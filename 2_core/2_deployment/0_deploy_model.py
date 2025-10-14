#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心部署模块 - 基于MS-Swift的模型部署器
支持LoRA、全参数模型的本地和云端部署
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# 添加utils路径
sys.path.append(str(Path(__file__).parent.parent.parent / "1_utils"))

from config_loader import ConfigLoader
from logger import Logger

class DeploymentManager:
    """部署管理器 - 统一的模型部署接口"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化部署管理器
        Args:
            config_path: 配置文件路径
        """
        self.logger = Logger("DeploymentManager").get_logger()
        self.config_loader = ConfigLoader()
        
        # 加载配置
        if config_path:
            self.config = self.config_loader.load_config(config_path)
        else:
            self.config = {}
        
        self.process = None
        self.deployment_info = {}
    
    def detect_model_type(self, checkpoint_path: str) -> str:
        """
        检测模型类型
        Args:
            checkpoint_path: 检查点路径
        Returns:
            str: 模型类型 ('lora' 或 'full')
        """
        path = Path(checkpoint_path)
        
        # 检查是否为LoRA checkpoint
        adapter_files = [
            "adapter_config.json", 
            "adapter_model.safetensors", 
            "adapter_model.bin",
            "adapter_model.pt"
        ]
        
        has_adapter = any((path / f).exists() for f in adapter_files)
        return "lora" if has_adapter else "full"
    
    def build_deployment_command(self, 
                                checkpoint_path: str,
                                model_type: str,
                                deploy_config: Dict[str, Any]) -> List[str]:
        """
        构建部署命令
        Args:
            checkpoint_path: 检查点路径
            model_type: 模型类型
            deploy_config: 部署配置
        Returns:
            List[str]: 部署命令列表
        """
        # 基础命令
        cmd = ["swift", "deploy"]
        
        # 根据模型类型设置参数
        if model_type == "lora":
            cmd.extend(["--adapters", checkpoint_path])
        else:
            cmd.extend(["--model", checkpoint_path])
        
        # 网络配置
        port = deploy_config.get('port', 8000)
        host = deploy_config.get('host', '0.0.0.0')
        cmd.extend(["--port", str(port)])
        cmd.extend(["--host", host])
        
        # 推理后端
        infer_backend = deploy_config.get('infer_backend', 'pt')
        cmd.extend(["--infer_backend", infer_backend])
        
        # 生成参数
        if 'temperature' in deploy_config:
            cmd.extend(["--temperature", str(deploy_config['temperature'])])
        
        if 'max_new_tokens' in deploy_config:
            cmd.extend(["--max_new_tokens", str(deploy_config['max_new_tokens'])])
        
        if 'top_p' in deploy_config:
            cmd.extend(["--top_p", str(deploy_config['top_p'])])
        
        # 服务模型名称
        if 'served_model_name' in deploy_config:
            cmd.extend(["--served_model_name", deploy_config['served_model_name']])
        
        # GPU设置
        if 'gpu_ids' in deploy_config:
            gpu_ids = deploy_config['gpu_ids']
            if isinstance(gpu_ids, list):
                gpu_str = ",".join(map(str, gpu_ids))
            else:
                gpu_str = str(gpu_ids)
            
            # 设置环境变量
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        
        # vLLM特定配置
        if infer_backend == 'vllm':
            if 'tensor_parallel_size' in deploy_config:
                cmd.extend(["--tensor_parallel_size", str(deploy_config['tensor_parallel_size'])])
            if 'max_model_len' in deploy_config:
                cmd.extend(["--max_model_len", str(deploy_config['max_model_len'])])
        
        # DeepSpeed配置
        if 'deepspeed' in deploy_config:
            cmd.extend(["--deepspeed", deploy_config['deepspeed']])
        
        return cmd
    
    def deploy_single_model(self, 
                          checkpoint_path: str, 
                          deploy_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        部署单个模型
        Args:
            checkpoint_path: 检查点路径
            deploy_config: 部署配置
        Returns:
            Dict[str, Any]: 部署信息
        """
        if deploy_config is None:
            deploy_config = self.config.get('deployment', {})
        
        # 确保deploy_config不为None
        if not deploy_config:
            deploy_config = {}
        
        # 检测模型类型
        model_type = self.detect_model_type(checkpoint_path)
        self.logger.info(f"🔍 检测到模型类型: {model_type.upper()}")
        
        # 构建部署命令
        cmd = self.build_deployment_command(checkpoint_path, model_type, deploy_config)
        
        # 记录部署信息
        self.deployment_info = {
            'checkpoint_path': checkpoint_path,
            'model_type': model_type,
            'deploy_config': deploy_config,
            'command': ' '.join(cmd),
            'port': deploy_config.get('port', 8000),
            'host': deploy_config.get('host', '0.0.0.0'),
            'status': 'deploying'
        }
        
        self.logger.info(f"🚀 开始部署{model_type.upper()}模型")
        self.logger.info(f"📁 检查点路径: {checkpoint_path}")
        self.logger.info(f"🌐 服务地址: http://{deploy_config.get('host', '0.0.0.0')}:{deploy_config.get('port', 8000)}")
        self.logger.info(f"⚙️  推理后端: {deploy_config.get('infer_backend', 'pt')}")
        
        try:
            # 启动部署进程
            self.logger.info(f"💻 执行命令: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.deployment_info['status'] = 'running'
            self.deployment_info['pid'] = self.process.pid
            
            self.logger.info(f"✅ 部署启动成功，进程ID: {self.process.pid}")
            
            return self.deployment_info
            
        except Exception as e:
            self.deployment_info['status'] = 'failed'
            self.deployment_info['error'] = str(e)
            self.logger.error(f"❌ 部署失败: {str(e)}")
            raise e
    
    def deploy_multi_lora(self, 
                         lora_configs: Dict[str, str], 
                         deploy_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        部署多LoRA模型
        Args:
            lora_configs: LoRA配置字典 {name: path}
            deploy_config: 部署配置
        Returns:
            Dict[str, Any]: 部署信息
        """
        if deploy_config is None:
            deploy_config = self.config.get('deployment', {})
        
        # 确保deploy_config不为None
        if not deploy_config:
            deploy_config = {}
        
        # 构建adapters字符串
        adapters_str = " ".join([f"{name}={path}" for name, path in lora_configs.items()])
        
        # 基础命令
        cmd = [
            "swift", "deploy",
            "--adapters", adapters_str,
            "--port", str(deploy_config.get('port', 8000)),
            "--host", deploy_config.get('host', '0.0.0.0')
        ]
        
        # 添加其他配置
        if 'infer_backend' in deploy_config:
            cmd.extend(["--infer_backend", deploy_config['infer_backend']])
        if 'temperature' in deploy_config:
            cmd.extend(["--temperature", str(deploy_config['temperature'])])
        if 'max_new_tokens' in deploy_config:
            cmd.extend(["--max_new_tokens", str(deploy_config['max_new_tokens'])])
        
        self.logger.info(f"🔄 开始部署多LoRA模型: {list(lora_configs.keys())}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            deployment_info = {
                'model_type': 'multi_lora',
                'lora_configs': lora_configs,
                'deploy_config': deploy_config,
                'command': ' '.join(cmd),
                'port': deploy_config.get('port', 8000),
                'host': deploy_config.get('host', '0.0.0.0'),
                'status': 'running',
                'pid': self.process.pid
            }
            
            self.logger.info("✅ 多LoRA模型部署启动成功")
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"❌ 多LoRA部署失败: {str(e)}")
            raise e
    
    def stop_deployment(self):
        """停止部署"""
        if self.process and self.process.poll() is None:
            self.logger.info("🛑 正在停止部署...")
            self.process.terminate()
            
            try:
                self.process.wait(timeout=10)
                self.logger.info("✅ 部署已停止")
            except subprocess.TimeoutExpired:
                self.logger.warning("⚠️  强制终止部署进程")
                self.process.kill()
                self.process.wait()
            
            if 'status' in self.deployment_info:
                self.deployment_info['status'] = 'stopped'
        else:
            self.logger.info("ℹ️  没有运行中的部署")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        if not self.deployment_info:
            return {'status': 'not_deployed'}
        
        # 检查进程状态
        if self.process:
            if self.process.poll() is None:
                self.deployment_info['status'] = 'running'
            else:
                self.deployment_info['status'] = 'stopped'
                self.deployment_info['return_code'] = self.process.returncode
        
        return self.deployment_info.copy()
    
    def get_deployment_logs(self, lines: int = 50) -> Dict[str, List[str]]:
        """
        获取部署日志
        Args:
            lines: 获取的日志行数
        Returns:
            Dict[str, List[str]]: 包含stdout和stderr的日志
        """
        if not self.process:
            return {'stdout': [], 'stderr': []}
        
        try:
            # 获取stdout
            stdout_lines = []
            if self.process.stdout:
                # 注意：这里只是示例，实际实现可能需要更复杂的日志管理
                pass
            
            # 获取stderr
            stderr_lines = []
            if self.process.stderr:
                # 注意：这里只是示例，实际实现可能需要更复杂的日志管理
                pass
            
            return {
                'stdout': stdout_lines[-lines:] if stdout_lines else [],
                'stderr': stderr_lines[-lines:] if stderr_lines else []
            }
        except Exception as e:
            self.logger.error(f"❌ 获取日志失败: {str(e)}")
            return {'stdout': [], 'stderr': []}

class DeploymentPresets:
    """部署预设配置"""
    
    @staticmethod
    def get_local_preset() -> Dict[str, Any]:
        """获取本地部署预设"""
        return {
            'host': '127.0.0.1',
            'port': 8000,
            'infer_backend': 'pt',
            'temperature': 0.7,
            'max_new_tokens': 1024,
            'top_p': 0.9
        }
    
    @staticmethod
    def get_server_preset() -> Dict[str, Any]:
        """获取服务器部署预设"""
        return {
            'host': '0.0.0.0',
            'port': 8000,
            'infer_backend': 'vllm',
            'temperature': 0.7,
            'max_new_tokens': 2048,
            'top_p': 0.9,
            'tensor_parallel_size': 1,
            'max_model_len': 4096
        }
    
    @staticmethod
    def get_production_preset() -> Dict[str, Any]:
        """获取生产环境预设"""
        return {
            'host': '0.0.0.0',
            'port': 8000,
            'infer_backend': 'vllm',
            'temperature': 0.7,
            'max_new_tokens': 1024,
            'top_p': 0.9,
            'tensor_parallel_size': 2,
            'max_model_len': 8192,
            'served_model_name': 'custom-model'
        }

def create_deployment_manager(config_path: Optional[str] = None) -> DeploymentManager:
    """
    创建部署管理器实例
    Args:
        config_path: 配置文件路径
    Returns:
        DeploymentManager: 部署管理器实例
    """
    return DeploymentManager(config_path)

if __name__ == "__main__":
    # 示例用法
    print("🧪 部署器测试...")
    
    # 创建部署管理器
    deploy_manager = create_deployment_manager()
    
    # 显示预设配置
    presets = {
        "本地部署": DeploymentPresets.get_local_preset(),
        "服务器部署": DeploymentPresets.get_server_preset(),
        "生产环境": DeploymentPresets.get_production_preset(),
    }
    
    print("🔧 可用的部署预设:")
    for name, preset in presets.items():
        print(f"  {name}: {preset}")
    
    print("✅ 部署器模块加载成功")