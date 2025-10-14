#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU管理模块 - 显存分配与监控
支持自动分配最空闲设备、显存监控等功能
"""

import os
import torch
import subprocess
import time
import threading
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import psutil
import logging

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """GPU信息数据类"""
    device_id: int
    name: str
    total_memory: float  # GB
    used_memory: float   # GB
    free_memory: float   # GB
    utilization: float   # %
    temperature: Optional[int] = None  # °C
    power_usage: Optional[float] = None  # W

class GPUManager:
    """GPU管理器"""
    
    def __init__(self, monitor_interval: float = 5.0):
        """
        初始化GPU管理器
        Args:
            monitor_interval: 监控间隔时间(秒)
        """
        self.monitor_interval = monitor_interval
        self._monitoring = False
        self._monitor_thread = None
        self._gpu_history: Dict[int, List[GPUInfo]] = {}
        
        # 检查CUDA可用性
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA不可用，GPU功能将被禁用")
            return
        
        self.device_count = torch.cuda.device_count()
        logger.info(f"检测到 {self.device_count} 张GPU")
        
        # 初始化GPU信息
        self._update_gpu_info()
    
    def get_gpu_info(self, device_id: Optional[int] = None) -> List[GPUInfo]:
        """
        获取GPU信息
        Args:
            device_id: 指定GPU ID，None表示获取所有GPU信息
        Returns:
            GPU信息列表
        """
        if not self.cuda_available:
            return []
        
        gpu_infos = []
        
        if device_id is not None:
            if 0 <= device_id < self.device_count:
                gpu_infos.append(self._get_single_gpu_info(device_id))
            else:
                raise ValueError(f"无效的GPU ID: {device_id}")
        else:
            for i in range(self.device_count):
                gpu_infos.append(self._get_single_gpu_info(i))
        
        return gpu_infos
    
    def _get_single_gpu_info(self, device_id: int) -> GPUInfo:
        """获取单个GPU信息"""
        # PyTorch GPU信息
        props = torch.cuda.get_device_properties(device_id)
        name = props.name
        total_memory = props.total_memory / (1024**3)  # 转换为GB
        
        # 当前使用的显存
        if device_id == torch.cuda.current_device():
            used_memory = torch.cuda.memory_allocated(device_id) / (1024**3)
        else:
            # 切换设备获取信息
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(device_id)
            used_memory = torch.cuda.memory_allocated(device_id) / (1024**3)
            torch.cuda.set_device(current_device)
        
        free_memory = total_memory - used_memory
        utilization = (used_memory / total_memory) * 100
        
        # 尝试获取nvidia-smi信息
        temperature, power_usage = self._get_nvidia_smi_info(device_id)
        
        return GPUInfo(
            device_id=device_id,
            name=name,
            total_memory=total_memory,
            used_memory=used_memory,
            free_memory=free_memory,
            utilization=utilization,
            temperature=temperature,
            power_usage=power_usage
        )
    
    def _get_nvidia_smi_info(self, device_id: int) -> Tuple[Optional[int], Optional[float]]:
        """通过nvidia-smi获取额外GPU信息"""
        try:
            cmd = [
                "nvidia-smi", 
                "--query-gpu=temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                f"--id={device_id}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0] != '[Not Supported]':
                    parts = lines[0].split(', ')
                    if len(parts) >= 2:
                        try:
                            temp = int(parts[0]) if parts[0] != '[Not Supported]' else None
                            power = float(parts[1]) if parts[1] != '[Not Supported]' else None
                            return temp, power
                        except (ValueError, IndexError):
                            pass
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return None, None
    
    def get_best_device(self, 
                       memory_required: float = 0.0,
                       exclude_devices: List[int] = None) -> Optional[int]:
        """
        获取最适合的GPU设备
        Args:
            memory_required: 所需显存(GB)
            exclude_devices: 排除的设备列表
        Returns:
            最佳设备ID，如果没有可用设备返回None
        """
        if not self.cuda_available:
            return None
        
        exclude_devices = exclude_devices or []
        gpu_infos = self.get_gpu_info()
        
        # 过滤可用设备
        available_gpus = []
        for gpu_info in gpu_infos:
            if (gpu_info.device_id not in exclude_devices and 
                gpu_info.free_memory >= memory_required):
                available_gpus.append(gpu_info)
        
        if not available_gpus:
            logger.warning("没有满足要求的可用GPU")
            return None
        
        # 按可用显存排序，选择显存最多的
        best_gpu = max(available_gpus, key=lambda x: x.free_memory)
        
        logger.info(f"选择GPU {best_gpu.device_id}: {best_gpu.name}, "
                   f"可用显存: {best_gpu.free_memory:.1f}GB")
        
        return best_gpu.device_id
    
    def set_device(self, device_id: int):
        """设置当前使用的GPU设备"""
        if not self.cuda_available:
            logger.warning("CUDA不可用，无法设置GPU设备")
            return
        
        if 0 <= device_id < self.device_count:
            torch.cuda.set_device(device_id)
            logger.info(f"已设置当前GPU设备为: {device_id}")
        else:
            raise ValueError(f"无效的GPU设备ID: {device_id}")
    
    def clear_memory(self, device_id: Optional[int] = None):
        """
        清理GPU显存
        Args:
            device_id: 指定GPU ID，None表示清理当前设备
        """
        if not self.cuda_available:
            return
        
        if device_id is not None:
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            torch.cuda.set_device(current_device)
            logger.info(f"已清理GPU {device_id} 显存缓存")
        else:
            torch.cuda.empty_cache()
            logger.info("已清理当前GPU显存缓存")
    
    def get_memory_usage(self, device_id: Optional[int] = None) -> Dict[str, float]:
        """
        获取显存使用情况
        Args:
            device_id: GPU设备ID
        Returns:
            显存使用信息字典
        """
        if not self.cuda_available:
            return {}
        
        if device_id is None:
            device_id = torch.cuda.current_device()
        
        # 获取显存信息
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        cached_memory = torch.cuda.memory_reserved(device_id)
        free_memory = total_memory - cached_memory
        
        return {
            'total_gb': total_memory / (1024**3),
            'allocated_gb': allocated_memory / (1024**3),
            'cached_gb': cached_memory / (1024**3),
            'free_gb': free_memory / (1024**3),
            'utilization_percent': (allocated_memory / total_memory) * 100
        }
    
    def start_monitoring(self):
        """开始GPU监控"""
        if not self.cuda_available or self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("GPU监控已启动")
    
    def stop_monitoring(self):
        """停止GPU监控"""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=self.monitor_interval + 1)
            logger.info("GPU监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                gpu_infos = self.get_gpu_info()
                
                for gpu_info in gpu_infos:
                    device_id = gpu_info.device_id
                    
                    # 保存历史记录
                    if device_id not in self._gpu_history:
                        self._gpu_history[device_id] = []
                    
                    self._gpu_history[device_id].append(gpu_info)
                    
                    # 保持历史记录数量限制
                    if len(self._gpu_history[device_id]) > 100:
                        self._gpu_history[device_id] = self._gpu_history[device_id][-100:]
                    
                    # 检查异常情况
                    self._check_gpu_alerts(gpu_info)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"GPU监控出错: {e}")
                time.sleep(self.monitor_interval)
    
    def _check_gpu_alerts(self, gpu_info: GPUInfo):
        """检查GPU告警"""
        # 显存使用率过高
        if gpu_info.utilization > 95:
            logger.warning(f"GPU {gpu_info.device_id} 显存使用率过高: {gpu_info.utilization:.1f}%")
        
        # 温度过高
        if gpu_info.temperature and gpu_info.temperature > 80:
            logger.warning(f"GPU {gpu_info.device_id} 温度过高: {gpu_info.temperature}°C")
    
    def _update_gpu_info(self):
        """更新GPU信息"""
        if self.cuda_available:
            for i in range(self.device_count):
                gpu_info = self._get_single_gpu_info(i)
                logger.info(f"GPU {i}: {gpu_info.name} "
                          f"({gpu_info.total_memory:.1f}GB 总显存)")
    
    def get_gpu_summary(self) -> Dict[str, Any]:
        """获取GPU使用摘要"""
        if not self.cuda_available:
            return {"cuda_available": False}
        
        gpu_infos = self.get_gpu_info()
        summary = {
            "cuda_available": True,
            "device_count": self.device_count,
            "current_device": torch.cuda.current_device(),
            "devices": []
        }
        
        for gpu_info in gpu_infos:
            device_summary = {
                "id": gpu_info.device_id,
                "name": gpu_info.name,
                "memory": {
                    "total": gpu_info.total_memory,
                    "used": gpu_info.used_memory,
                    "free": gpu_info.free_memory,
                    "utilization": gpu_info.utilization
                }
            }
            
            if gpu_info.temperature is not None:
                device_summary["temperature"] = gpu_info.temperature
            if gpu_info.power_usage is not None:
                device_summary["power_usage"] = gpu_info.power_usage
            
            summary["devices"].append(device_summary)
        
        return summary

# 全局GPU管理器实例
gpu_manager = GPUManager()

def get_best_device(memory_required: float = 0.0) -> Optional[int]:
    """获取最佳GPU设备的便捷函数"""
    return gpu_manager.get_best_device(memory_required)

def get_gpu_info(device_id: Optional[int] = None) -> List[GPUInfo]:
    """获取GPU信息的便捷函数"""
    return gpu_manager.get_gpu_info(device_id)

def clear_gpu_memory(device_id: Optional[int] = None):
    """清理GPU显存的便捷函数"""
    gpu_manager.clear_memory(device_id)

def setup_gpu_environment(memory_required: float = 0.0) -> Optional[int]:
    """
    设置GPU环境的便捷函数
    Args:
        memory_required: 所需显存(GB)
    Returns:
        选择的GPU设备ID
    """
    device_id = get_best_device(memory_required)
    if device_id is not None:
        gpu_manager.set_device(device_id)
        # 清理显存
        clear_gpu_memory(device_id)
    return device_id

if __name__ == "__main__":
    # 测试代码
    manager = GPUManager()
    
    print("=== GPU信息 ===")
    infos = manager.get_gpu_info()
    for info in infos:
        print(f"GPU {info.device_id}: {info.name}")
        print(f"  显存: {info.used_memory:.1f}GB / {info.total_memory:.1f}GB ({info.utilization:.1f}%)")
        if info.temperature:
            print(f"  温度: {info.temperature}°C")
        if info.power_usage:
            print(f"  功耗: {info.power_usage}W")
    
    print("\n=== 最佳设备选择 ===")
    best_device = manager.get_best_device(memory_required=4.0)
    print(f"最佳设备: GPU {best_device}")
    
    print("\n=== GPU摘要 ===")
    summary = manager.get_gpu_summary()
    print(summary)