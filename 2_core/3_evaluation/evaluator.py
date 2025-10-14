#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心评估模块 - 模型性能评估和基准测试
支持多种评估指标和基准数据集
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable

# 添加utils路径
sys.path.append(str(Path(__file__).parent.parent.parent / "1_utils"))

from config_loader import ConfigLoader
from logger import Logger
from metrics import MetricsCalculator

class EvaluationManager:
    """评估管理器 - 统一的模型评估接口"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化评估管理器
        Args:
            config_path: 配置文件路径
        """
        self.logger = Logger("EvaluationManager").get_logger()
        self.config_loader = ConfigLoader()
        self.metrics_calculator = MetricsCalculator()
        
        # 加载配置
        if config_path:
            self.config = self.config_loader.load_config(config_path)
        else:
            self.config = {}
        
        self.evaluation_results = {}
    
    def evaluate_generation_quality(self, 
                                   references: List[str], 
                                   candidates: List[str],
                                   task_name: str = "generation") -> Dict[str, Any]:
        """
        评估文本生成质量
        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            task_name: 任务名称
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"📊 开始评估{task_name}任务，共{len(references)}个样本")
        
        start_time = time.time()
        
        # 计算各种指标
        results = {
            'task_name': task_name,
            'sample_count': len(references),
            'timestamp': time.time(),
            'metrics': {}
        }
        
        try:
            # BLEU分数
            bleu_scores = []
            for ref, cand in zip(references, candidates):
                bleu = self.metrics_calculator.calculate_bleu([ref], cand)
                bleu_scores.append(bleu)
            
            avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
            results['metrics']['bleu'] = {
                'average': avg_bleu,
                'scores': bleu_scores
            }
            
            # ROUGE分数
            rouge_scores = []
            for ref, cand in zip(references, candidates):
                rouge = self.metrics_calculator.calculate_rouge(ref, cand)
                rouge_scores.append(rouge)
            
            # 计算平均ROUGE分数
            avg_rouge = {
                'rouge-1': sum(r['rouge-1']['f'] for r in rouge_scores) / len(rouge_scores),
                'rouge-2': sum(r['rouge-2']['f'] for r in rouge_scores) / len(rouge_scores),
                'rouge-l': sum(r['rouge-l']['f'] for r in rouge_scores) / len(rouge_scores)
            }
            results['metrics']['rouge'] = {
                'average': avg_rouge,
                'scores': rouge_scores
            }
            
            # 精确匹配
            exact_matches = [
                self.metrics_calculator.calculate_exact_match(ref, cand)
                for ref, cand in zip(references, candidates)
            ]
            exact_match_rate = sum(exact_matches) / len(exact_matches)
            results['metrics']['exact_match'] = {
                'rate': exact_match_rate,
                'matches': exact_matches
            }
            
            # F1分数
            f1_scores = [
                self.metrics_calculator.calculate_f1_score(ref, cand)
                for ref, cand in zip(references, candidates)
            ]
            avg_f1 = sum(f1_scores) / len(f1_scores)
            results['metrics']['f1'] = {
                'average': avg_f1,
                'scores': f1_scores
            }
            
            # 语义相似度（如果可用）
            try:
                semantic_scores = [
                    self.metrics_calculator.calculate_semantic_similarity(ref, cand)
                    for ref, cand in zip(references, candidates)
                ]
                avg_semantic = sum(semantic_scores) / len(semantic_scores)
                results['metrics']['semantic_similarity'] = {
                    'average': avg_semantic,
                    'scores': semantic_scores
                }
            except Exception as e:
                self.logger.warning(f"⚠️  语义相似度计算失败: {e}")
                results['metrics']['semantic_similarity'] = None
            
            # 计算总用时
            end_time = time.time()
            results['evaluation_time'] = end_time - start_time
            
            self.logger.info(f"✅ 评估完成，用时{results['evaluation_time']:.2f}秒")
            self.logger.info(f"📈 BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge['rouge-l']:.4f}, F1: {avg_f1:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 评估过程中出错: {str(e)}")
            raise e
    
    def evaluate_model_performance(self,
                                 model_inference_func: Callable[[str], str],
                                 test_dataset: List[Dict[str, str]],
                                 task_type: str = "qa") -> Dict[str, Any]:
        """
        评估模型整体性能
        Args:
            model_inference_func: 模型推理函数
            test_dataset: 测试数据集 [{"input": "...", "output": "..."}]
            task_type: 任务类型
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"🎯 开始评估模型性能，任务类型: {task_type}")
        self.logger.info(f"📋 测试样本数: {len(test_dataset)}")
        
        start_time = time.time()
        
        # 执行推理
        predictions = []
        references = []
        inference_times = []
        
        for i, sample in enumerate(test_dataset):
            try:
                # 记录推理时间
                infer_start = time.time()
                prediction = model_inference_func(sample['input'])
                infer_end = time.time()
                
                predictions.append(prediction)
                references.append(sample['output'])
                inference_times.append(infer_end - infer_start)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"📊 已处理 {i + 1}/{len(test_dataset)} 个样本")
                
            except Exception as e:
                self.logger.error(f"❌ 样本 {i+1} 推理失败: {str(e)}")
                predictions.append("")
                references.append(sample['output'])
                inference_times.append(0)
        
        # 计算评估指标
        evaluation_results = self.evaluate_generation_quality(
            references, predictions, f"{task_type}_evaluation"
        )
        
        # 添加性能指标
        evaluation_results['performance'] = {
            'total_inference_time': sum(inference_times),
            'average_inference_time': sum(inference_times) / len(inference_times),
            'samples_per_second': len(test_dataset) / sum(inference_times) if sum(inference_times) > 0 else 0,
            'failed_samples': sum(1 for p in predictions if not p.strip())
        }
        
        total_time = time.time() - start_time
        evaluation_results['total_evaluation_time'] = total_time
        
        self.logger.info(f"🏁 模型评估完成，总用时{total_time:.2f}秒")
        self.logger.info(f"⚡ 平均推理时间: {evaluation_results['performance']['average_inference_time']:.3f}秒/样本")
        
        return evaluation_results
    
    def benchmark_on_dataset(self,
                           model_inference_func: Callable[[str], str],
                           dataset_name: str,
                           dataset_path: Optional[str] = None,
                           sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        在基准数据集上评估
        Args:
            model_inference_func: 模型推理函数
            dataset_name: 数据集名称
            dataset_path: 数据集路径
            sample_size: 采样大小
        Returns:
            Dict[str, Any]: 基准测试结果
        """
        self.logger.info(f"🎯 开始基准测试: {dataset_name}")
        
        # 加载数据集
        test_data = self.load_benchmark_dataset(dataset_name, dataset_path, sample_size)
        
        if not test_data:
            raise ValueError(f"❌ 无法加载数据集: {dataset_name}")
        
        # 执行评估
        results = self.evaluate_model_performance(
            model_inference_func, test_data, dataset_name
        )
        
        results['dataset_name'] = dataset_name
        results['dataset_size'] = len(test_data)
        
        # 保存结果
        self.save_evaluation_results(results, f"benchmark_{dataset_name}")
        
        return results
    
    def load_benchmark_dataset(self,
                             dataset_name: str,
                             dataset_path: Optional[str] = None,
                             sample_size: Optional[int] = None) -> List[Dict[str, str]]:
        """
        加载基准数据集
        Args:
            dataset_name: 数据集名称
            dataset_path: 数据集路径
            sample_size: 采样大小
        Returns:
            List[Dict[str, str]]: 数据集
        """
        try:
            if dataset_path and Path(dataset_path).exists():
                # 从文件加载
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # 使用内置测试数据
                data = self.get_builtin_test_data(dataset_name)
            
            # 采样
            if sample_size and len(data) > sample_size:
                import random
                data = random.sample(data, sample_size)
                self.logger.info(f"📊 随机采样 {sample_size} 个样本")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 加载数据集失败: {str(e)}")
            return []
    
    def get_builtin_test_data(self, dataset_name: str) -> List[Dict[str, str]]:
        """获取内置测试数据"""
        if dataset_name == "qa_test":
            return [
                {"input": "什么是人工智能？", "output": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"},
                {"input": "Python是什么？", "output": "Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。"},
                {"input": "什么是机器学习？", "output": "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式。"},
                {"input": "如何学习编程？", "output": "学习编程需要选择一门语言，理解基础概念，多实践项目，并持续学习新技术。"},
                {"input": "什么是深度学习？", "output": "深度学习是机器学习的一个分支，使用多层神经网络来处理复杂的数据模式。"}
            ]
        elif dataset_name == "math_test":
            return [
                {"input": "2+3等于多少？", "output": "5"},
                {"input": "10的平方根是多少？", "output": "3.16"},
                {"input": "9×7等于多少？", "output": "63"},
                {"input": "100除以4等于多少？", "output": "25"},
                {"input": "圆周率的近似值是多少？", "output": "3.14159"}
            ]
        else:
            return [
                {"input": "你好", "output": "你好！我是AI助手，很高兴为您服务。"},
                {"input": "你能做什么？", "output": "我可以回答问题、提供信息、帮助解决问题等。"},
                {"input": "今天天气怎么样？", "output": "抱歉，我无法获取实时天气信息。"},
            ]
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str):
        """
        保存评估结果
        Args:
            results: 评估结果
            filename: 文件名
        """
        try:
            # 创建输出目录
            output_dir = Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)
            
            # 添加时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"{filename}_{timestamp}.json"
            
            # 保存结果
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 评估结果已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存评估结果失败: {str(e)}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        比较多个模型的评估结果
        Args:
            model_results: 模型结果字典 {model_name: results}
        Returns:
            Dict[str, Any]: 比较结果
        """
        self.logger.info(f"📊 开始比较{len(model_results)}个模型")
        
        comparison = {
            'models': list(model_results.keys()),
            'comparison_time': time.time(),
            'metrics_comparison': {}
        }
        
        # 提取关键指标进行比较
        metrics_to_compare = ['bleu', 'rouge', 'f1', 'exact_match']
        
        for metric in metrics_to_compare:
            comparison['metrics_comparison'][metric] = {}
            
            for model_name, results in model_results.items():
                if 'metrics' in results and metric in results['metrics']:
                    if isinstance(results['metrics'][metric], dict):
                        if 'average' in results['metrics'][metric]:
                            comparison['metrics_comparison'][metric][model_name] = results['metrics'][metric]['average']
                        elif 'rate' in results['metrics'][metric]:
                            comparison['metrics_comparison'][metric][model_name] = results['metrics'][metric]['rate']
        
        # 找出最佳模型
        best_models = {}
        for metric, scores in comparison['metrics_comparison'].items():
            if scores:
                if metric == 'rouge':
                    # ROUGE是字典，取rouge-l
                    rouge_l_scores = {model: score['rouge-l'] if isinstance(score, dict) else score for model, score in scores.items()}
                    best_model = max(rouge_l_scores.items(), key=lambda x: x[1])
                else:
                    best_model = max(scores.items(), key=lambda x: x[1])
                best_models[metric] = best_model
        
        comparison['best_models'] = best_models
        
        self.logger.info("🏆 模型比较完成")
        for metric, (model, score) in best_models.items():
            self.logger.info(f"  {metric}: {model} ({score:.4f})")
        
        return comparison

class EvaluationPresets:
    """评估预设配置"""
    
    @staticmethod
    def get_quick_eval_config() -> Dict[str, Any]:
        """快速评估配置"""
        return {
            'sample_size': 50,
            'metrics': ['bleu', 'rouge', 'f1'],
            'save_results': True
        }
    
    @staticmethod
    def get_comprehensive_eval_config() -> Dict[str, Any]:
        """全面评估配置"""
        return {
            'sample_size': None,  # 使用全部数据
            'metrics': ['bleu', 'rouge', 'f1', 'exact_match', 'semantic_similarity'],
            'save_results': True,
            'generate_report': True
        }
    
    @staticmethod
    def get_benchmark_config() -> Dict[str, Any]:
        """基准测试配置"""
        return {
            'datasets': ['qa_test', 'math_test'],
            'sample_size': 100,
            'compare_models': True,
            'save_results': True
        }

def create_evaluation_manager(config_path: Optional[str] = None) -> EvaluationManager:
    """
    创建评估管理器实例
    Args:
        config_path: 配置文件路径
    Returns:
        EvaluationManager: 评估管理器实例
    """
    return EvaluationManager(config_path)

if __name__ == "__main__":
    # 示例用法
    print("🧪 评估器测试...")
    
    # 创建评估管理器
    eval_manager = create_evaluation_manager()
    
    # 测试数据
    references = ["这是一个测试句子。", "人工智能很有趣。"]
    candidates = ["这是测试句子。", "AI很有趣。"]
    
    # 执行评估
    results = eval_manager.evaluate_generation_quality(references, candidates, "test")
    
    print("📊 评估结果:")
    print(f"  BLEU: {results['metrics']['bleu']['average']:.4f}")
    print(f"  F1: {results['metrics']['f1']['average']:.4f}")
    
    print("✅ 评估器模块加载成功")