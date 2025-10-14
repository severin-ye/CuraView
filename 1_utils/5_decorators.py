#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装饰器库 - 常用的函数装饰器
包括性能计时、重试、异常捕获、缓存等
"""

import time
import functools
import traceback
import threading
from typing import Any, Callable, Dict, Optional, Union, List
from pathlib import Path
import json
import hashlib

def timer(func: Callable) -> Callable:
    """
    性能计时装饰器
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"⏱️  {func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    
    return wrapper

def retry(max_attempts: int = 3, 
          delay: float = 1.0, 
          backoff: float = 2.0,
          exceptions: tuple = (Exception,),
          on_failure: Optional[Callable] = None):
    """
    重试装饰器
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间(秒)
        backoff: 延迟倍数
        exceptions: 需要重试的异常类型
        on_failure: 失败时的回调函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # 最后一次尝试失败
                        if on_failure:
                            on_failure(func.__name__, attempt + 1, e)
                        raise e
                    
                    print(f"🔄 {func.__name__} 第{attempt + 1}次尝试失败: {e}")
                    print(f"⏳ {current_delay:.1f}秒后重试...")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None  # 理论上不会到达这里
        
        return wrapper
    return decorator

def exception_handler(log_errors: bool = True, 
                     return_on_error: Any = None,
                     raise_on_error: bool = False):
    """
    异常处理装饰器
    Args:
        log_errors: 是否记录错误日志
        return_on_error: 出错时返回的值
        raise_on_error: 是否重新抛出异常
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    print(f"❌ {func.__name__} 执行出错: {e}")
                    print(f"📍 错误位置: {traceback.format_exc()}")
                
                if raise_on_error:
                    raise e
                
                return return_on_error
        
        return wrapper
    return decorator

def cache_result(max_size: int = 128, ttl: Optional[float] = None):
    """
    结果缓存装饰器
    Args:
        max_size: 缓存最大条目数
        ttl: 缓存生存时间(秒)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        lock = threading.RLock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            with lock:
                current_time = time.time()
                
                # 检查缓存是否存在且未过期
                if cache_key in cache:
                    if ttl is None or (current_time - cache_times[cache_key]) < ttl:
                        return cache[cache_key]
                    else:
                        # 缓存过期，删除
                        del cache[cache_key]
                        del cache_times[cache_key]
                
                # 检查缓存大小限制
                if len(cache) >= max_size:
                    # 删除最旧的缓存项
                    oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                    del cache[oldest_key]
                    del cache_times[oldest_key]
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                cache[cache_key] = result
                cache_times[cache_key] = current_time
                
                return result
        
        # 添加缓存管理方法（使用 setattr 避免类型检查问题）
        def cache_clear():
            cache.clear()
            cache_times.clear()
        
        def cache_info():
            return {
                'size': len(cache),
                'max_size': max_size,
                'ttl': ttl
            }
        
        setattr(wrapper, 'cache_clear', cache_clear)
        setattr(wrapper, 'cache_info', cache_info)
        
        return wrapper
    return decorator

def rate_limit(calls_per_second: float = 1.0):
    """
    限流装饰器
    Args:
        calls_per_second: 每秒允许的调用次数
    """
    min_interval = 1.0 / calls_per_second
    
    def decorator(func: Callable) -> Callable:
        last_called = [0.0]
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                current_time = time.time()
                elapsed = current_time - last_called[0]
                
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    time.sleep(sleep_time)
                
                last_called[0] = time.time()
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def validate_types(**type_checks):
    """
    类型验证装饰器
    Args:
        **type_checks: 参数名到类型的映射
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 验证类型
            for param_name, expected_type in type_checks.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"参数 '{param_name}' 应为 {expected_type.__name__} 类型，"
                            f"实际为 {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def log_calls(logger=None, level: str = "INFO", include_result: bool = False):
    """
    函数调用日志装饰器
    Args:
        logger: 日志记录器，如果为None则使用print
        level: 日志级别
        include_result: 是否记录返回结果
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 记录调用
            args_str = ", ".join([repr(arg) for arg in args])
            kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            params_str = ", ".join(filter(None, [args_str, kwargs_str]))
            
            call_info = f"📞 调用 {func.__name__}({params_str})"
            
            if logger:
                getattr(logger, level.lower())(call_info)
            else:
                print(call_info)
            
            # 执行函数
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # 记录结果
            result_info = f"✅ {func.__name__} 完成，耗时 {end_time - start_time:.4f}秒"
            if include_result:
                result_info += f"，返回值: {repr(result)}"
            
            if logger:
                getattr(logger, level.lower())(result_info)
            else:
                print(result_info)
            
            return result
        
        return wrapper
    return decorator

def async_timeout(seconds: float):
    """
    异步超时装饰器（需要asyncio）
    Args:
        seconds: 超时时间(秒)
    """
    def decorator(func: Callable) -> Callable:
        if asyncio is None:
            raise ImportError("asyncio未安装，无法使用异步超时装饰器")
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"函数 {func.__name__} 执行超时 ({seconds}秒)")
        
        return wrapper
    return decorator

def singleton(cls):
    """
    单例模式装饰器
    """
    instances = {}
    lock = threading.Lock()
    
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return wrapper

def deprecated(reason: str = ""):
    """
    弃用警告装饰器
    Args:
        reason: 弃用原因
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import warnings
            warning_msg = f"函数 {func.__name__} 已弃用"
            if reason:
                warning_msg += f": {reason}"
            
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def profile_memory(func: Callable) -> Callable:
    """
    内存使用分析装饰器
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_delta = mem_after - mem_before
            
            print(f"🧠 {func.__name__} 内存使用: {mem_delta:+.2f}MB (执行前: {mem_before:.2f}MB, 执行后: {mem_after:.2f}MB)")
            
            return result
            
        except ImportError:
            print("⚠️  psutil未安装，无法分析内存使用")
            return func(*args, **kwargs)
    
    return wrapper

def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """生成缓存键"""
    # 创建一个包含函数名、参数的字符串
    key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
    
    # 生成哈希值作为缓存键
    return hashlib.md5(key_data.encode()).hexdigest()

# 尝试导入asyncio
try:
    import asyncio
except ImportError:
    asyncio = None

# 组合装饰器示例
def robust_api_call(max_attempts: int = 3, 
                   delay: float = 1.0,
                   rate_limit_rps: float = 1.0,
                   cache_ttl: Optional[float] = 300):
    """
    组合装饰器：适用于API调用的鲁棒性装饰器
    包含重试、限流、缓存等功能
    """
    def decorator(func: Callable) -> Callable:
        # 按顺序应用装饰器
        func = exception_handler(log_errors=True)(func)
        func = retry(max_attempts=max_attempts, delay=delay)(func)
        func = rate_limit(calls_per_second=rate_limit_rps)(func)
        func = cache_result(ttl=cache_ttl)(func)
        func = timer(func)
        
        return func
    
    return decorator

if __name__ == "__main__":
    # 测试装饰器
    print("🧪 测试装饰器功能...")
    
    @timer
    @cache_result(max_size=5, ttl=10)
    @retry(max_attempts=2)
    def test_function(x: int) -> int:
        """测试函数"""
        if x < 0:
            raise ValueError("负数不支持")
        time.sleep(0.1)  # 模拟耗时操作
        return x * 2
    
    # 测试正常调用
    print("第一次调用:")
    result1 = test_function(5)
    print(f"结果: {result1}")
    
    print("\n第二次调用 (应该使用缓存):")
    result2 = test_function(5)
    print(f"结果: {result2}")
    
    # 测试类型验证
    @validate_types(name=str, age=int)
    def greet(name: str, age: int) -> str:
        return f"Hello, {name}! You are {age} years old."
    
    try:
        print(f"\n类型验证测试: {greet('Alice', 25)}")
        greet('Bob', '30')  # 这会抛出类型错误
    except TypeError as e:
        print(f"✅ 类型验证捕获错误: {e}")
    
    # 测试日志记录
    @log_calls(include_result=True)
    def calculate(a: int, b: int) -> int:
        return a + b
    
    print(f"\n日志记录测试:")
    result = calculate(3, 4)
    
    print("✅ 装饰器测试完成！")