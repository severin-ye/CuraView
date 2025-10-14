#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心层统一接口 - 整合训练、推理、部署、评估功能
提供简洁的API接口供上层调用
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入各个模块
from training.trainer import TrainingManager, TrainingPresets
from inference.inference import InferenceManager, InferencePresets
from deployment.deploy import DeploymentManager, DeploymentPresets
from evaluation.evaluator import EvaluationManager, EvaluationPresets

# 添加utils路径
sys.path.append(str(current_dir.parent / "1_utils"))
from logger import Logger

class CoreAPI:
    """核心API - 统一的模型训练、推理、部署、评估接口"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化核心API
        Args:
            config_path: 配置文件路径
        """
        self.logger = Logger("CoreAPI").get_logger()
        self.config_path = config_path
        
        # 初始化各个管理器
        self.trainer = None
        self.inference_manager = None
        self.deployment_manager = None
        self.evaluation_manager = None
        
        self.logger.info("🚀 核心API初始化完成")
    
    def get_trainer(self) -> TrainingManager:
        """获取训练管理器"""
        if self.trainer is None:
            self.trainer = TrainingManager(self.config_path)
        return self.trainer
    
    def get_inference_manager(self, checkpoint_path: Optional[str] = None) -> InferenceManager:
        """获取推理管理器"""
        if self.inference_manager is None:
            self.inference_manager = InferenceManager(checkpoint_path, self.config_path)
        return self.inference_manager
    
    def get_deployment_manager(self) -> DeploymentManager:
        """获取部署管理器"""
        if self.deployment_manager is None:
            self.deployment_manager = DeploymentManager(self.config_path)
        return self.deployment_manager
    
    def get_evaluation_manager(self) -> EvaluationManager:
        """获取评估管理器"""
        if self.evaluation_manager is None:
            self.evaluation_manager = EvaluationManager(self.config_path)
        return self.evaluation_manager
    
    # 训练相关接口
    def train_model(self, 
                   train_type: str = "lora",
                   model: str = "Qwen/Qwen2.5-7B-Instruct",
                   dataset: Optional[List[str]] = None,
                   output_dir: str = "./output",
                   **kwargs) -> str:
        """
        训练模型
        Args:
            train_type: 训练类型 (lora/qlora/full/multimodal)
            model: 基础模型
            dataset: 训练数据集
            output_dir: 输出目录
            **kwargs: 其他训练参数
        Returns:
            str: 训练输出目录
        """
        self.logger.info(f"🎯 开始{train_type.upper()}训练")
        
        trainer = self.get_trainer()
        
        if train_type == "lora":
            return trainer.train_lora(
                model=model, dataset=dataset, output_dir=output_dir, **kwargs
            )
        elif train_type == "qlora":
            return trainer.train_qlora(
                model=model, dataset=dataset, output_dir=output_dir, **kwargs
            )
        elif train_type == "full":
            return trainer.train_full_params(
                model=model, dataset=dataset, output_dir=output_dir, **kwargs
            )
        elif train_type == "multimodal":
            return trainer.train_multimodal(
                model=model, dataset=dataset, output_dir=output_dir, **kwargs
            )
        else:
            raise ValueError(f"❌ 不支持的训练类型: {train_type}")
    
    def train_with_preset(self, preset_name: str, **overrides) -> str:
        """
        使用预设配置训练
        Args:
            preset_name: 预设名称 (lora/qlora/full/multimodal)
            **overrides: 覆盖参数
        Returns:
            str: 训练输出目录
        """
        presets = {
            'lora': TrainingPresets.get_lora_preset,
            'qlora': TrainingPresets.get_qlora_preset,
            'full': TrainingPresets.get_full_preset,
            'multimodal': TrainingPresets.get_multimodal_preset
        }
        
        if preset_name not in presets:
            raise ValueError(f"❌ 未知预设: {preset_name}")
        
        config = presets[preset_name]()
        config.update(overrides)
        
        trainer = self.get_trainer()
        trainer.config = config
        return trainer.run_training_from_config()
    
    # 推理相关接口
    def load_model_for_inference(self, checkpoint_path: str):
        """
        加载模型用于推理
        Args:
            checkpoint_path: 模型检查点路径
        """
        self.logger.info(f"📂 加载推理模型: {checkpoint_path}")
        inference_manager = self.get_inference_manager(checkpoint_path)
        inference_manager.load_model()
        self.logger.info("✅ 模型加载完成")
    
    def infer_single(self, 
                    question: str, 
                    checkpoint_path: Optional[str] = None,
                    **infer_config) -> str:
        """
        单条推理
        Args:
            question: 输入问题
            checkpoint_path: 检查点路径（如果未预加载）
            **infer_config: 推理配置
        Returns:
            str: 推理结果
        """
        inference_manager = self.get_inference_manager(checkpoint_path)
        
        # 如果模型未加载，先加载
        if not inference_manager.engine:
            if checkpoint_path:
                inference_manager.load_model(checkpoint_path)
            else:
                raise ValueError("❌ 模型未加载且未提供检查点路径")
        
        return inference_manager.infer_single(question, **infer_config)
    
    def infer_batch(self, 
                   questions: List[str], 
                   checkpoint_path: Optional[str] = None,
                   **infer_config) -> List[str]:
        """
        批量推理
        Args:
            questions: 问题列表
            checkpoint_path: 检查点路径
            **infer_config: 推理配置
        Returns:
            List[str]: 推理结果列表
        """
        inference_manager = self.get_inference_manager(checkpoint_path)
        
        if not inference_manager.engine and checkpoint_path:
            inference_manager.load_model(checkpoint_path)
        
        return inference_manager.infer_batch(questions, **infer_config)
    
    def start_interactive_chat(self, checkpoint_path: Optional[str] = None):
        """
        启动交互式对话
        Args:
            checkpoint_path: 检查点路径
        """
        inference_manager = self.get_inference_manager(checkpoint_path)
        
        if not inference_manager.engine and checkpoint_path:
            inference_manager.load_model(checkpoint_path)
        
        inference_manager.interactive_chat()
    
    # 部署相关接口
    def deploy_model(self, 
                    checkpoint_path: str, 
                    **deploy_config) -> Dict[str, Any]:
        """
        部署模型
        Args:
            checkpoint_path: 检查点路径
            **deploy_config: 部署配置
        Returns:
            Dict[str, Any]: 部署信息
        """
        self.logger.info(f"🚀 部署模型: {checkpoint_path}")
        
        deployment_manager = self.get_deployment_manager()
        return deployment_manager.deploy_single_model(checkpoint_path, deploy_config)
    
    def deploy_with_preset(self, 
                          checkpoint_path: str, 
                          preset_name: str = "local", 
                          **overrides) -> Dict[str, Any]:
        """
        使用预设配置部署
        Args:
            checkpoint_path: 检查点路径
            preset_name: 预设名称 (local/server/production)
            **overrides: 覆盖参数
        Returns:
            Dict[str, Any]: 部署信息
        """
        presets = {
            'local': DeploymentPresets.get_local_preset,
            'server': DeploymentPresets.get_server_preset,
            'production': DeploymentPresets.get_production_preset
        }
        
        if preset_name not in presets:
            raise ValueError(f"❌ 未知预设: {preset_name}")
        
        config = presets[preset_name]()
        config.update(overrides)
        
        return self.deploy_model(checkpoint_path, **config)
    
    def stop_deployment(self):
        """停止部署"""
        if self.deployment_manager:
            self.deployment_manager.stop_deployment()
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        if self.deployment_manager:
            return self.deployment_manager.get_deployment_status()
        return {'status': 'not_deployed'}
    
    # 评估相关接口
    def evaluate_model(self, 
                      checkpoint_path: str,
                      test_data: List[Dict[str, str]],
                      task_type: str = "qa") -> Dict[str, Any]:
        """
        评估模型性能
        Args:
            checkpoint_path: 检查点路径
            test_data: 测试数据
            task_type: 任务类型
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"📊 评估模型: {checkpoint_path}")
        
        # 创建推理函数
        inference_manager = self.get_inference_manager(checkpoint_path)
        inference_manager.load_model()
        
        def inference_func(question: str) -> str:
            return inference_manager.infer_single(question)
        
        # 执行评估
        evaluation_manager = self.get_evaluation_manager()
        return evaluation_manager.evaluate_model_performance(
            inference_func, test_data, task_type
        )
    
    def benchmark_model(self, 
                       checkpoint_path: str,
                       dataset_name: str = "qa_test",
                       sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        基准测试
        Args:
            checkpoint_path: 检查点路径
            dataset_name: 数据集名称
            sample_size: 采样大小
        Returns:
            Dict[str, Any]: 基准测试结果
        """
        self.logger.info(f"🎯 基准测试: {dataset_name}")
        
        # 创建推理函数
        inference_manager = self.get_inference_manager(checkpoint_path)
        inference_manager.load_model()
        
        def inference_func(question: str) -> str:
            return inference_manager.infer_single(question)
        
        # 执行基准测试
        evaluation_manager = self.get_evaluation_manager()
        return evaluation_manager.benchmark_on_dataset(
            inference_func, dataset_name, sample_size=sample_size
        )
    
    # 工作流接口
    def full_pipeline(self, 
                     train_config: Dict[str, Any],
                     eval_config: Optional[Dict[str, Any]] = None,
                     deploy_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        完整的训练-评估-部署流水线
        Args:
            train_config: 训练配置
            eval_config: 评估配置
            deploy_config: 部署配置
        Returns:
            Dict[str, Any]: 流水线结果
        """
        results = {
            'pipeline_start': self.logger.info("🔄 开始完整流水线"),
            'training': None,
            'evaluation': None,
            'deployment': None
        }
        
        try:
            # 1. 训练
            self.logger.info("📚 第1步: 开始训练")
            output_dir = self.train_model(**train_config)
            results['training'] = {'output_dir': output_dir, 'status': 'success'}
            
            # 2. 评估（如果配置了）
            if eval_config:
                self.logger.info("📊 第2步: 开始评估")
                eval_results = self.evaluate_model(output_dir, **eval_config)
                results['evaluation'] = eval_results
            
            # 3. 部署（如果配置了）
            if deploy_config:
                self.logger.info("🚀 第3步: 开始部署")
                deploy_results = self.deploy_model(output_dir, **deploy_config)
                results['deployment'] = deploy_results
            
            self.logger.info("✅ 完整流水线执行成功")
            results['status'] = 'success'
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 流水线执行失败: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            return results

# 全局API实例（单例模式）
_core_api_instance = None

def get_core_api(config_path: Optional[str] = None) -> CoreAPI:
    """
    获取核心API实例（单例）
    Args:
        config_path: 配置文件路径
    Returns:
        CoreAPI: 核心API实例
    """
    global _core_api_instance
    if _core_api_instance is None:
        _core_api_instance = CoreAPI(config_path)
    return _core_api_instance

# 便利函数
def train_model(**kwargs) -> str:
    """便利函数：训练模型"""
    api = get_core_api()
    return api.train_model(**kwargs)

def infer_single(question: str, checkpoint_path: str, **kwargs) -> str:
    """便利函数：单条推理"""
    api = get_core_api()
    return api.infer_single(question, checkpoint_path, **kwargs)

def deploy_model(checkpoint_path: str, **kwargs) -> Dict[str, Any]:
    """便利函数：部署模型"""
    api = get_core_api()
    return api.deploy_model(checkpoint_path, **kwargs)

def evaluate_model(checkpoint_path: str, test_data: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """便利函数：评估模型"""
    api = get_core_api()
    return api.evaluate_model(checkpoint_path, test_data, **kwargs)

if __name__ == "__main__":
    # 示例用法
    print("🧪 核心API测试...")
    
    # 创建API实例
    api = get_core_api()
    
    print("🔧 可用的功能:")
    print("  📚 训练: train_model()")
    print("  🤖 推理: infer_single(), infer_batch()")
    print("  🚀 部署: deploy_model()")
    print("  📊 评估: evaluate_model(), benchmark_model()")
    print("  🔄 流水线: full_pipeline()")
    
    print("✅ 核心API加载成功")