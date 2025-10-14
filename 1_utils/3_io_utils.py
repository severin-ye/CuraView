#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件IO工具模块 - 文件操作、缓存和序列化
"""

import os
import json
import yaml
import pickle
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import threading
import time

class FileCache:
    """文件缓存管理器"""
    
    def __init__(self, cache_dir: str = "./6_output/cache", max_size: int = 1000, ttl: int = 3600):
        """
        初始化文件缓存
        Args:
            cache_dir: 缓存目录
            max_size: 最大缓存条目数
            ttl: 缓存生存时间(秒)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.ttl = ttl
        self._cache_index = {}
        self._lock = threading.RLock()
        
        # 加载缓存索引
        self._load_index()
        
        # 启动清理线程
        self._start_cleanup_thread()
    
    def _get_cache_key(self, key: str) -> str:
        """生成缓存键的哈希值"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _load_index(self):
        """加载缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    self._cache_index = json.load(f)
            except Exception:
                self._cache_index = {}
    
    def _save_index(self):
        """保存缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(self._cache_index, f, indent=2)
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        Args:
            key: 缓存键
        Returns:
            缓存值，如果不存在或过期返回None
        """
        with self._lock:
            cache_key = self._get_cache_key(key)
            
            if cache_key not in self._cache_index:
                return None
            
            cache_info = self._cache_index[cache_key]
            
            # 检查是否过期
            if time.time() - cache_info['timestamp'] > self.ttl:
                self._remove_cache(cache_key)
                return None
            
            # 读取缓存文件
            cache_path = self._get_cache_path(cache_key)
            if not cache_path.exists():
                del self._cache_index[cache_key]
                return None
            
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                self._remove_cache(cache_key)
                return None
    
    def set(self, key: str, value: Any):
        """
        设置缓存值
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            # 检查缓存大小限制
            if len(self._cache_index) >= self.max_size:
                self._evict_oldest()
            
            cache_key = self._get_cache_key(key)
            cache_path = self._get_cache_path(cache_key)
            
            # 保存缓存值
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # 更新索引
                self._cache_index[cache_key] = {
                    'key': key,
                    'timestamp': time.time(),
                    'size': cache_path.stat().st_size
                }
                
                self._save_index()
                
            except Exception as e:
                if cache_path.exists():
                    cache_path.unlink()
                raise e
    
    def _remove_cache(self, cache_key: str):
        """删除缓存项"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()
        
        if cache_key in self._cache_index:
            del self._cache_index[cache_key]
    
    def _evict_oldest(self):
        """删除最旧的缓存项"""
        if not self._cache_index:
            return
        
        oldest_key = min(
            self._cache_index.keys(),
            key=lambda k: self._cache_index[k]['timestamp']
        )
        
        self._remove_cache(oldest_key)
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        def cleanup_worker():
            while True:
                time.sleep(300)  # 5分钟清理一次
                self._cleanup_expired()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, info in self._cache_index.items()
                if current_time - info['timestamp'] > self.ttl
            ]
            
            for key in expired_keys:
                self._remove_cache(key)
            
            if expired_keys:
                self._save_index()
    
    def clear(self):
        """清空所有缓存"""
        with self._lock:
            for cache_key in list(self._cache_index.keys()):
                self._remove_cache(cache_key)
            
            self._cache_index = {}
            self._save_index()

class IOUtils:
    """文件IO工具类"""
    
    @staticmethod
    def read_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        读取JSON文件
        Args:
            file_path: 文件路径
            encoding: 文件编码
        Returns:
            解析后的字典
        """
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    
    @staticmethod
    def write_json(data: Dict[str, Any], file_path: Union[str, Path], 
                   encoding: str = 'utf-8', indent: int = 2):
        """
        写入JSON文件
        Args:
            data: 要写入的数据
            file_path: 文件路径
            encoding: 文件编码
            indent: 缩进空格数
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    
    @staticmethod
    def read_yaml(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        读取YAML文件
        Args:
            file_path: 文件路径
            encoding: 文件编码
        Returns:
            解析后的字典
        """
        with open(file_path, 'r', encoding=encoding) as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def write_yaml(data: Dict[str, Any], file_path: Union[str, Path], 
                   encoding: str = 'utf-8'):
        """
        写入YAML文件
        Args:
            data: 要写入的数据
            file_path: 文件路径
            encoding: 文件编码
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def read_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        读取文本文件
        Args:
            file_path: 文件路径
            encoding: 文件编码
        Returns:
            文件内容
        """
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def write_text(content: str, file_path: Union[str, Path], 
                   encoding: str = 'utf-8'):
        """
        写入文本文件
        Args:
            content: 文件内容
            file_path: 文件路径
            encoding: 文件编码
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def read_lines(file_path: Union[str, Path], encoding: str = 'utf-8', 
                   strip: bool = True) -> List[str]:
        """
        按行读取文件
        Args:
            file_path: 文件路径
            encoding: 文件编码
            strip: 是否去除行尾空白字符
        Returns:
            行列表
        """
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
            if strip:
                lines = [line.strip() for line in lines]
            return lines
    
    @staticmethod
    def write_lines(lines: List[str], file_path: Union[str, Path], 
                    encoding: str = 'utf-8'):
        """
        按行写入文件
        Args:
            lines: 行列表
            file_path: 文件路径
            encoding: 文件编码
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            for line in lines:
                f.write(line + '\n')
    
    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path], 
                  create_dirs: bool = True):
        """
        复制文件
        Args:
            src: 源文件路径
            dst: 目标文件路径
            create_dirs: 是否创建目标目录
        """
        src = Path(src)
        dst = Path(dst)
        
        if create_dirs:
            dst.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src, dst)
    
    @staticmethod
    def copy_directory(src: Union[str, Path], dst: Union[str, Path]):
        """
        复制目录
        Args:
            src: 源目录路径
            dst: 目标目录路径
        """
        shutil.copytree(src, dst, dirs_exist_ok=True)
    
    @staticmethod
    def move_file(src: Union[str, Path], dst: Union[str, Path]):
        """
        移动文件
        Args:
            src: 源文件路径
            dst: 目标文件路径
        """
        shutil.move(src, dst)
    
    @staticmethod
    def delete_file(file_path: Union[str, Path]):
        """
        删除文件
        Args:
            file_path: 文件路径
        """
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
    
    @staticmethod
    def delete_directory(dir_path: Union[str, Path]):
        """
        删除目录
        Args:
            dir_path: 目录路径
        """
        dir_path = Path(dir_path)
        if dir_path.exists():
            shutil.rmtree(dir_path)
    
    @staticmethod
    def ensure_directory(dir_path: Union[str, Path]):
        """
        确保目录存在
        Args:
            dir_path: 目录路径
        """
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        获取文件大小
        Args:
            file_path: 文件路径
        Returns:
            文件大小(字节)
        """
        return Path(file_path).stat().st_size
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
        """
        获取文件哈希值
        Args:
            file_path: 文件路径
            algorithm: 哈希算法(md5, sha1, sha256等)
        Returns:
            文件哈希值
        """
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def find_files(directory: Union[str, Path], pattern: str = "*", 
                   recursive: bool = True) -> List[Path]:
        """
        查找文件
        Args:
            directory: 搜索目录
            pattern: 文件模式
            recursive: 是否递归搜索
        Returns:
            文件路径列表
        """
        directory = Path(directory)
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))

def with_file_cache(cache_key_func: Optional[Callable] = None, 
                   ttl: int = 3600, cache_dir: str = "./6_output/cache"):
    """
    文件缓存装饰器
    Args:
        cache_key_func: 缓存键生成函数
        ttl: 缓存生存时间
        cache_dir: 缓存目录
    """
    cache = FileCache(cache_dir=cache_dir, ttl=ttl)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # 尝试获取缓存
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator

# 全局实例
file_cache = FileCache()
io_utils = IOUtils()

if __name__ == "__main__":
    # 测试代码
    
    # 测试JSON操作
    test_data = {"test": "data", "timestamp": str(datetime.now())}
    test_file = "./6_output/test/test.json"
    
    print("🧪 测试文件IO工具...")
    
    # 写入和读取JSON
    IOUtils.write_json(test_data, test_file)
    loaded_data = IOUtils.read_json(test_file)
    print(f"✅ JSON读写测试通过: {loaded_data}")
    
    # 测试缓存
    cache = FileCache()
    cache.set("test_key", {"cached": "value"})
    cached_value = cache.get("test_key")
    print(f"✅ 缓存测试通过: {cached_value}")
    
    # 测试装饰器
    @with_file_cache()
    def expensive_function(x):
        time.sleep(0.1)  # 模拟耗时操作
        return x * 2
    
    start_time = time.time()
    result1 = expensive_function(42)
    first_time = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_function(42)
    second_time = time.time() - start_time
    
    print(f"✅ 缓存装饰器测试: 第一次 {first_time:.3f}s, 第二次 {second_time:.3f}s")
    
    print("🎉 所有测试通过！")