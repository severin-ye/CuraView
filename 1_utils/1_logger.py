#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志记录模块 - 标准化日志输出与管理
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 添加颜色
        if sys.stdout.isatty():  # 只在终端中使用颜色
            log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, 
                 name: str = "hd_system",
                 log_dir: str = "./6_output/logs",
                 log_level: str = "INFO",
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 structured_format: bool = False):
        """
        初始化日志记录器
        Args:
            name: 日志记录器名称
            log_dir: 日志文件目录
            log_level: 日志级别
            max_file_size: 单个日志文件最大大小
            backup_count: 日志文件备份数量
            enable_console: 是否启用控制台输出
            enable_file: 是否启用文件输出
            structured_format: 是否使用结构化格式(JSON)
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.structured_format = structured_format
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 设置格式
        if structured_format:
            formatter = logging.Formatter('%(message)s')
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
        else:
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(process)d | %(funcName)s:%(lineno)d | %(message)s'
            )
        
        # 控制台处理器
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if enable_file:
            # 常规日志文件
            log_file = self.log_dir / f"{name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, 
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            
            if structured_format:
                file_handler.setFormatter(formatter)
            else:
                file_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(file_handler)
            
            # 错误日志文件
            error_file = self.log_dir / f"{name}_error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            
            if structured_format:
                error_handler.setFormatter(formatter)
            else:
                error_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(error_handler)
    
    def _format_structured_message(self, level: str, message: str, **kwargs) -> str:
        """格式化结构化日志消息"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            **kwargs
        }
        return json.dumps(log_entry, ensure_ascii=False)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        if self.structured_format:
            message = self._format_structured_message('DEBUG', message, **kwargs)
        self.logger.debug(message)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        if self.structured_format:
            message = self._format_structured_message('INFO', message, **kwargs)
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        if self.structured_format:
            message = self._format_structured_message('WARNING', message, **kwargs)
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        if self.structured_format:
            message = self._format_structured_message('ERROR', message, **kwargs)
        self.logger.error(message)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        if self.structured_format:
            message = self._format_structured_message('CRITICAL', message, **kwargs)
        self.logger.critical(message)
    
    def exception(self, message: str, **kwargs):
        """异常日志"""
        if self.structured_format:
            message = self._format_structured_message('ERROR', message, **kwargs)
        self.logger.exception(message)
    
    def log_function_call(self, func_name: str, args: Optional[Dict[str, Any]] = None, **kwargs):
        """记录函数调用"""
        self.info(f"调用函数: {func_name}", function=func_name, args=args, **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """记录性能指标"""
        self.info(f"性能指标: {operation} 耗时 {duration:.3f}s", 
                 operation=operation, duration=duration, **kwargs)
    
    def log_gpu_usage(self, gpu_id: int, memory_used: float, memory_total: float, **kwargs):
        """记录GPU使用情况"""
        usage_percent = (memory_used / memory_total) * 100
        self.info(f"GPU {gpu_id} 显存使用: {memory_used:.1f}GB / {memory_total:.1f}GB ({usage_percent:.1f}%)",
                 gpu_id=gpu_id, memory_used=memory_used, memory_total=memory_total, 
                 usage_percent=usage_percent, **kwargs)

class LoggerManager:
    """日志管理器 - 管理多个日志记录器"""
    
    _loggers: Dict[str, StructuredLogger] = {}
    
    @classmethod
    def get_logger(cls, 
                   name: str,
                   log_dir: str = "./6_output/logs",
                   log_level: str = "INFO",
                   **kwargs) -> StructuredLogger:
        """
        获取或创建日志记录器
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            log_level: 日志级别
            **kwargs: 其他参数
        Returns:
            日志记录器实例
        """
        if name not in cls._loggers:
            cls._loggers[name] = StructuredLogger(
                name=name,
                log_dir=log_dir,
                log_level=log_level,
                **kwargs
            )
        return cls._loggers[name]
    
    @classmethod
    def get_training_logger(cls) -> StructuredLogger:
        """获取训练日志记录器"""
        return cls.get_logger("training", log_level="INFO")
    
    @classmethod
    def get_inference_logger(cls) -> StructuredLogger:
        """获取推理日志记录器"""
        return cls.get_logger("inference", log_level="INFO")
    
    @classmethod
    def get_deployment_logger(cls) -> StructuredLogger:
        """获取部署日志记录器"""
        return cls.get_logger("deployment", log_level="INFO")
    
    @classmethod
    def get_agent_logger(cls, agent_name: str) -> StructuredLogger:
        """获取Agent日志记录器"""
        return cls.get_logger(f"agent_{agent_name}", log_level="INFO")
    
    @classmethod
    def setup_default_logging(cls, log_level: str = "INFO"):
        """设置默认日志配置"""
        # 设置根日志记录器
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # 禁用一些第三方库的详细日志
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)

# 便捷函数
def get_logger(name: str = "hd_system", **kwargs) -> StructuredLogger:
    """获取日志记录器的便捷函数"""
    return LoggerManager.get_logger(name, **kwargs)

def setup_logging(log_level: str = "INFO"):
    """设置默认日志配置的便捷函数"""
    LoggerManager.setup_default_logging(log_level)

if __name__ == "__main__":
    # 测试代码
    logger = get_logger("test")
    
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    
    logger.log_function_call("test_function", {"param1": "value1"})
    logger.log_performance("model_inference", 1.234)
    logger.log_gpu_usage(0, 8.5, 40.0)
    
    print("日志测试完成")