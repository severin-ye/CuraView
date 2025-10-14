"""
智能推理器（显存优化）

提供智能化的GPU显存管理和优化推理功能。
"""

import torch
import gc
import psutil
import GPUtil
from typing import Optional, Dict, Any, List, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryManager:
    """GPU内存管理器"""
    
    def __init__(self, target_memory_ratio: float = 0.9):
        """
        初始化内存管理器
        
        Args:
            target_memory_ratio: 目标内存使用比例
        """
        self.target_memory_ratio = target_memory_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """
        获取GPU内存信息
        
        Returns:
            GPU内存信息字典
        """
        if not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0, "ratio": 0}
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)
        free_memory = total_memory - cached_memory
        
        return {
            "total": total_memory / 1024**3,  # GB
            "allocated": allocated_memory / 1024**3,
            "cached": cached_memory / 1024**3,
            "free": free_memory / 1024**3,
            "ratio": cached_memory / total_memory
        }
    
    def clear_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU缓存已清理")
    
    def optimize_memory_usage(self):
        """优化内存使用"""
        self.clear_cache()
        
        # 设置内存分配策略
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.target_memory_ratio)
    
    @contextmanager
    def memory_guard(self):
        """内存保护上下文管理器"""
        initial_memory = self.get_gpu_memory_info()
        try:
            yield
        finally:
            self.clear_cache()
            final_memory = self.get_gpu_memory_info()
            logger.debug(f"内存使用: {initial_memory['ratio']:.2%} -> {final_memory['ratio']:.2%}")
    
    def check_memory_available(self, required_memory_gb: float) -> bool:
        """
        检查是否有足够的内存
        
        Args:
            required_memory_gb: 需要的内存(GB)
            
        Returns:
            是否有足够内存
        """
        memory_info = self.get_gpu_memory_info()
        return memory_info["free"] >= required_memory_gb


class SmartGPUInference:
    """智能GPU推理器"""
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 max_memory_gb: Optional[float] = None,
                 optimization_level: int = 1):
        """
        初始化智能推理器
        
        Args:
            model_path: 模型路径
            device: 设备类型
            max_memory_gb: 最大内存限制(GB)
            optimization_level: 优化级别(0-3)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_memory_gb = max_memory_gb
        self.optimization_level = optimization_level
        
        self.memory_manager = MemoryManager()
        self.model = None
        self.tokenizer = None
        
        # 优化配置
        self.optimization_configs = {
            0: {"torch_dtype": torch.float32, "device_map": None},
            1: {"torch_dtype": torch.float16, "device_map": "auto"},
            2: {"torch_dtype": torch.float16, "device_map": "auto", "load_in_8bit": True},
            3: {"torch_dtype": torch.float16, "device_map": "auto", "load_in_4bit": True}
        }
    
    def estimate_model_memory(self) -> float:
        """
        估算模型内存需求
        
        Returns:
            预估内存需求(GB)
        """
        try:
            # 简单估算：根据模型文件大小
            import os
            model_size = 0
            for root, dirs, files in os.walk(self.model_path):
                for file in files:
                    if file.endswith(('.bin', '.safetensors')):
                        model_size += os.path.getsize(os.path.join(root, file))
            
            # 模型权重 + 激活值缓存 + 其他开销
            estimated_memory = model_size / 1024**3 * 1.5  # GB
            logger.info(f"预估模型内存需求: {estimated_memory:.2f} GB")
            return estimated_memory
        except Exception as e:
            logger.warning(f"无法估算模型内存需求: {e}")
            return 8.0  # 默认估算值
    
    def auto_select_optimization(self) -> Dict[str, Any]:
        """
        自动选择优化配置
        
        Returns:
            优化配置字典
        """
        memory_info = self.memory_manager.get_gpu_memory_info()
        estimated_memory = self.estimate_model_memory()
        
        available_memory = memory_info["free"]
        
        # 根据可用内存自动选择优化级别
        if available_memory >= estimated_memory * 1.2:
            selected_level = min(self.optimization_level, 1)
        elif available_memory >= estimated_memory:
            selected_level = min(self.optimization_level, 2)
        else:
            selected_level = 3
        
        config = self.optimization_configs[selected_level].copy()
        logger.info(f"自动选择优化级别: {selected_level}, 配置: {config}")
        return config
    
    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        加载模型和tokenizer
        
        Returns:
            模型和tokenizer实例
        """
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer
        
        with self.memory_manager.memory_guard():
            # 加载tokenizer
            logger.info("加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 自动选择优化配置
            optimization_config = self.auto_select_optimization()
            
            # 加载模型
            logger.info(f"加载模型 {self.model_path}...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    **optimization_config
                )
                
                # 如果没有使用device_map，手动移动到设备
                if optimization_config.get("device_map") is None:
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                logger.info("模型加载成功")
                
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                # 尝试更激进的优化
                logger.info("尝试更激进的内存优化...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True
                )
                self.model.eval()
                logger.info("模型加载成功（8bit量化）")
        
        return self.model, self.tokenizer
    
    def smart_batch_inference(self, 
                            texts: List[str], 
                            max_new_tokens: int = 512,
                            batch_size: Optional[int] = None,
                            **generate_kwargs) -> List[str]:
        """
        智能批量推理
        
        Args:
            texts: 输入文本列表
            max_new_tokens: 最大生成token数
            batch_size: 批大小，None为自动确定
            **generate_kwargs: 生成参数
            
        Returns:
            生成结果列表
        """
        model, tokenizer = self.load_model()
        
        if batch_size is None:
            batch_size = self.auto_determine_batch_size(texts[0], max_new_tokens)
        
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"处理批次 {batch_num}/{total_batches}, 大小: {len(batch_texts)}")
            
            with self.memory_manager.memory_guard():
                try:
                    batch_results = self._process_batch(
                        batch_texts, max_new_tokens, **generate_kwargs
                    )
                    results.extend(batch_results)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"批次 {batch_num} 内存不足，使用单条处理")
                        # 降级为单条处理
                        for text in batch_texts:
                            single_result = self._process_batch(
                                [text], max_new_tokens, **generate_kwargs
                            )
                            results.extend(single_result)
                    else:
                        raise
        
        return results
    
    def auto_determine_batch_size(self, sample_text: str, max_new_tokens: int) -> int:
        """
        自动确定批大小
        
        Args:
            sample_text: 样本文本
            max_new_tokens: 最大生成token数
            
        Returns:
            推荐的批大小
        """
        model, tokenizer = self.load_model()
        
        # 估算单条样本的内存需求
        sample_tokens = tokenizer.encode(sample_text, return_tensors="pt")
        estimated_tokens = len(sample_tokens[0]) + max_new_tokens
        
        # 简单的内存估算
        memory_info = self.memory_manager.get_gpu_memory_info()
        available_memory = memory_info["free"]
        
        # 保守估算：每个token大约需要4字节的激活值内存
        estimated_memory_per_sample = estimated_tokens * 4 / 1024**3  # GB
        
        if estimated_memory_per_sample > 0:
            max_batch_size = int(available_memory * 0.8 / estimated_memory_per_sample)
            batch_size = max(1, min(max_batch_size, 8))  # 限制在1-8之间
        else:
            batch_size = 4  # 默认值
        
        logger.info(f"自动确定批大小: {batch_size}")
        return batch_size
    
    def _process_batch(self, 
                      texts: List[str], 
                      max_new_tokens: int, 
                      **generate_kwargs) -> List[str]:
        """
        处理单个批次
        
        Args:
            texts: 批次文本
            max_new_tokens: 最大生成token数
            **generate_kwargs: 生成参数
            
        Returns:
            生成结果列表
        """
        model, tokenizer = self.load_model()
        
        # 编码输入
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # 移动到设备
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            generate_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                **generate_kwargs
            }
            
            outputs = model.generate(**inputs, **generate_config)
        
        # 解码结果
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        
        results = []
        for tokens in generated_tokens:
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(text.strip())
        
        return results
    
    def get_memory_usage_report(self) -> Dict[str, Any]:
        """
        获取内存使用报告
        
        Returns:
            内存使用报告
        """
        gpu_info = self.memory_manager.get_gpu_memory_info()
        
        # CPU内存信息
        cpu_memory = psutil.virtual_memory()
        
        return {
            "gpu": gpu_info,
            "cpu": {
                "total": cpu_memory.total / 1024**3,
                "used": cpu_memory.used / 1024**3,
                "available": cpu_memory.available / 1024**3,
                "ratio": cpu_memory.percent / 100
            },
            "optimization_level": self.optimization_level,
            "model_loaded": self.model is not None
        }
    
    def unload_model(self):
        """卸载模型释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.memory_manager.clear_cache()
        logger.info("模型已卸载，内存已释放")


class AdaptiveInference:
    """自适应推理器"""
    
    def __init__(self, model_path: str):
        """
        初始化自适应推理器
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.inference_engines = {}
        self.current_engine = None
        
    def create_engine(self, optimization_level: int) -> SmartGPUInference:
        """
        创建推理引擎
        
        Args:
            optimization_level: 优化级别
            
        Returns:
            推理引擎实例
        """
        if optimization_level not in self.inference_engines:
            self.inference_engines[optimization_level] = SmartGPUInference(
                self.model_path,
                optimization_level=optimization_level
            )
        return self.inference_engines[optimization_level]
    
    def adaptive_inference(self, 
                          texts: Union[str, List[str]], 
                          max_new_tokens: int = 512,
                          **generate_kwargs) -> Union[str, List[str]]:
        """
        自适应推理
        
        Args:
            texts: 输入文本
            max_new_tokens: 最大生成token数
            **generate_kwargs: 生成参数
            
        Returns:
            生成结果
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # 尝试不同的优化级别
        for level in [1, 2, 3]:
            try:
                engine = self.create_engine(level)
                results = engine.smart_batch_inference(
                    texts, max_new_tokens, **generate_kwargs
                )
                self.current_engine = engine
                logger.info(f"使用优化级别 {level} 推理成功")
                return results[0] if is_single else results
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"优化级别 {level} 内存不足，尝试更高级别")
                    continue
                else:
                    raise
        
        raise RuntimeError("所有优化级别都无法完成推理")
    
    def get_best_engine(self) -> Optional[SmartGPUInference]:
        """获取当前最佳引擎"""
        return self.current_engine


# 便捷函数
def create_smart_inference(model_path: str, optimization_level: int = 1) -> SmartGPUInference:
    """
    创建智能推理器的便捷函数
    
    Args:
        model_path: 模型路径
        optimization_level: 优化级别
        
    Returns:
        智能推理器实例
    """
    return SmartGPUInference(model_path, optimization_level=optimization_level)


def auto_inference(model_path: str, 
                  texts: Union[str, List[str]], 
                  max_new_tokens: int = 512,
                  **generate_kwargs) -> Union[str, List[str]]:
    """
    自动推理的便捷函数
    
    Args:
        model_path: 模型路径
        texts: 输入文本
        max_new_tokens: 最大生成token数
        **generate_kwargs: 生成参数
        
    Returns:
        生成结果
    """
    adaptive_engine = AdaptiveInference(model_path)
    return adaptive_engine.adaptive_inference(texts, max_new_tokens, **generate_kwargs)