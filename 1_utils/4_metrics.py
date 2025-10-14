#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标模块 - 常见的模型评估指标实现
包括BLEU、ROUGE、METEOR、Loss等指标
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter, defaultdict
import json

try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sacrebleu import BLEU, CHRF, TER
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self):
        """初始化评估指标计算器"""
        self.metrics_history = []
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """确保NLTK数据可用"""
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("下载NLTK数据...")
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('wordnet', quiet=True)
                except:
                    pass
    
    def tokenize(self, text: str) -> List[str]:
        """
        文本分词
        Args:
            text: 输入文本
        Returns:
            分词结果
        """
        # 简单的中英文分词
        if re.search(r'[\u4e00-\u9fff]', text):
            # 中文：按字符分词
            return list(re.sub(r'[^\u4e00-\u9fff\w]', ' ', text))
        else:
            # 英文：按单词分词
            return re.findall(r'\b\w+\b', text.lower())
    
    def calculate_bleu(self, 
                      predictions: List[str], 
                      references: List[List[str]], 
                      max_order: int = 4,
                      smooth: bool = True) -> Dict[str, float]:
        """
        计算BLEU分数
        Args:
            predictions: 预测结果列表
            references: 参考答案列表（每个预测可有多个参考）
            max_order: 最大n-gram阶数
            smooth: 是否使用平滑
        Returns:
            BLEU分数字典
        """
        if len(predictions) != len(references):
            raise ValueError("预测和参考数量不匹配")
        
        # 使用sacrebleu（如果可用）
        if SACREBLEU_AVAILABLE:
            try:
                # 转换格式：每个参考列表取第一个
                refs = [refs[0] if refs else "" for refs in references]
                
                bleu = BLEU(max_ngram_order=max_order, smooth_method='exp' if smooth else 'none')
                score = bleu.corpus_score(predictions, [refs])
                
                return {
                    'bleu': score.score / 100.0,
                    'bleu_1': score.precisions[0] / 100.0 if len(score.precisions) > 0 else 0.0,
                    'bleu_2': score.precisions[1] / 100.0 if len(score.precisions) > 1 else 0.0,
                    'bleu_3': score.precisions[2] / 100.0 if len(score.precisions) > 2 else 0.0,
                    'bleu_4': score.precisions[3] / 100.0 if len(score.precisions) > 3 else 0.0,
                }
            except Exception:
                pass
        
        # 回退到NLTK或自实现
        if NLTK_AVAILABLE:
            try:
                # 分词
                pred_tokens = [self.tokenize(pred) for pred in predictions]
                ref_tokens = [[self.tokenize(ref) for ref in refs] for refs in references]
                
                smoothing = SmoothingFunction().method1 if smooth else None
                
                # 计算不同阶数的BLEU
                bleu_scores = {}
                for n in range(1, max_order + 1):
                    weights = [1.0/n] * n + [0.0] * (4-n)
                    scores = []
                    for pred, refs in zip(pred_tokens, ref_tokens):
                        if refs:  # 确保有参考答案
                            score = sentence_bleu(refs, pred, weights=weights, smoothing_function=smoothing)
                            scores.append(score)
                    
                    if scores:
                        bleu_scores[f'bleu_{n}'] = np.mean(scores)
                
                # 计算总体BLEU-4
                if 'bleu_4' in bleu_scores:
                    bleu_scores['bleu'] = bleu_scores['bleu_4']
                
                return bleu_scores
                
            except Exception as e:
                print(f"NLTK BLEU计算失败: {e}")
        
        # 简单的自实现BLEU
        return self._simple_bleu(predictions, references, max_order)
    
    def _simple_bleu(self, predictions: List[str], references: List[List[str]], max_order: int) -> Dict[str, float]:
        """简单的BLEU实现"""
        total_precisions = [0.0] * max_order
        total_counts = [0] * max_order
        
        for pred, refs in zip(predictions, references):
            if not refs:
                continue
            
            pred_tokens = self.tokenize(pred)
            ref_tokens_list = [self.tokenize(ref) for ref in refs]
            
            # 选择最佳参考
            best_ref = max(ref_tokens_list, key=len) if ref_tokens_list else []
            
            for n in range(1, max_order + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams = self._get_ngrams(best_ref, n)
                
                overlap = sum((Counter(pred_ngrams) & Counter(ref_ngrams)).values())
                total_count = len(pred_ngrams)
                
                if total_count > 0:
                    total_precisions[n-1] += overlap / total_count
                    total_counts[n-1] += 1
        
        # 计算平均精度
        bleu_scores = {}
        for n in range(max_order):
            if total_counts[n] > 0:
                bleu_scores[f'bleu_{n+1}'] = total_precisions[n] / total_counts[n]
            else:
                bleu_scores[f'bleu_{n+1}'] = 0.0
        
        # 几何平均作为总体BLEU
        if max_order >= 4:
            bleu_scores['bleu'] = (bleu_scores['bleu_1'] * bleu_scores['bleu_2'] * 
                                  bleu_scores['bleu_3'] * bleu_scores['bleu_4']) ** 0.25
        
        return bleu_scores
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """获取n-gram"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        计算ROUGE分数
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
        Returns:
            ROUGE分数字典
        """
        if len(predictions) != len(references):
            raise ValueError("预测和参考数量不匹配")
        
        if ROUGE_AVAILABLE:
            try:
                rouge = Rouge()
                scores = rouge.get_scores(predictions, references, avg=True)
                
                return {
                    'rouge_1_f': scores['rouge-1']['f'],
                    'rouge_1_p': scores['rouge-1']['p'],
                    'rouge_1_r': scores['rouge-1']['r'],
                    'rouge_2_f': scores['rouge-2']['f'],
                    'rouge_2_p': scores['rouge-2']['p'],
                    'rouge_2_r': scores['rouge-2']['r'],
                    'rouge_l_f': scores['rouge-l']['f'],
                    'rouge_l_p': scores['rouge-l']['p'],
                    'rouge_l_r': scores['rouge-l']['r'],
                }
            except Exception as e:
                print(f"ROUGE计算失败: {e}")
        
        # 简单的自实现ROUGE
        return self._simple_rouge(predictions, references)
    
    def _simple_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """简单的ROUGE实现"""
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.tokenize(pred)
            ref_tokens = self.tokenize(ref)
            
            # ROUGE-1
            pred_unigrams = set(pred_tokens)
            ref_unigrams = set(ref_tokens)
            overlap_1 = len(pred_unigrams & ref_unigrams)
            
            r1_precision = overlap_1 / len(pred_unigrams) if pred_unigrams else 0
            r1_recall = overlap_1 / len(ref_unigrams) if ref_unigrams else 0
            r1_f1 = 2 * r1_precision * r1_recall / (r1_precision + r1_recall) if (r1_precision + r1_recall) > 0 else 0
            
            rouge_1_scores.append({'f': r1_f1, 'p': r1_precision, 'r': r1_recall})
            
            # ROUGE-2
            pred_bigrams = set(self._get_ngrams(pred_tokens, 2))
            ref_bigrams = set(self._get_ngrams(ref_tokens, 2))
            overlap_2 = len(pred_bigrams & ref_bigrams)
            
            r2_precision = overlap_2 / len(pred_bigrams) if pred_bigrams else 0
            r2_recall = overlap_2 / len(ref_bigrams) if ref_bigrams else 0
            r2_f1 = 2 * r2_precision * r2_recall / (r2_precision + r2_recall) if (r2_precision + r2_recall) > 0 else 0
            
            rouge_2_scores.append({'f': r2_f1, 'p': r2_precision, 'r': r2_recall})
            
            # ROUGE-L (最长公共子序列)
            lcs_length = self._lcs_length(pred_tokens, ref_tokens)
            rl_precision = lcs_length / len(pred_tokens) if pred_tokens else 0
            rl_recall = lcs_length / len(ref_tokens) if ref_tokens else 0
            rl_f1 = 2 * rl_precision * rl_recall / (rl_precision + rl_recall) if (rl_precision + rl_recall) > 0 else 0
            
            rouge_l_scores.append({'f': rl_f1, 'p': rl_precision, 'r': rl_recall})
        
        # 计算平均分数
        return {
            'rouge_1_f': np.mean([s['f'] for s in rouge_1_scores]),
            'rouge_1_p': np.mean([s['p'] for s in rouge_1_scores]),
            'rouge_1_r': np.mean([s['r'] for s in rouge_1_scores]),
            'rouge_2_f': np.mean([s['f'] for s in rouge_2_scores]),
            'rouge_2_p': np.mean([s['p'] for s in rouge_2_scores]),
            'rouge_2_r': np.mean([s['r'] for s in rouge_2_scores]),
            'rouge_l_f': np.mean([s['f'] for s in rouge_l_scores]),
            'rouge_l_p': np.mean([s['p'] for s in rouge_l_scores]),
            'rouge_l_r': np.mean([s['r'] for s in rouge_l_scores]),
        }
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """
        计算精确匹配率
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
        Returns:
            精确匹配率
        """
        if len(predictions) != len(references):
            raise ValueError("预测和参考数量不匹配")
        
        matches = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip() == ref.strip())
        
        return matches / len(predictions)
    
    def calculate_f1_score(self, predictions: List[str], references: List[str]) -> float:
        """
        计算F1分数（基于token重叠）
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
        Returns:
            F1分数
        """
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self.tokenize(pred))
            ref_tokens = set(self.tokenize(ref))
            
            if not pred_tokens and not ref_tokens:
                f1_scores.append(1.0)
                continue
            
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue
            
            overlap = len(pred_tokens & ref_tokens)
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_score = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1_score)
        
        return np.mean(f1_scores)
    
    def calculate_perplexity(self, losses: List[float]) -> float:
        """
        计算困惑度
        Args:
            losses: 损失值列表
        Returns:
            困惑度
        """
        if not losses:
            return float('inf')
        
        avg_loss = np.mean(losses)
        return math.exp(avg_loss)
    
    def calculate_semantic_similarity(self, predictions: List[str], references: List[str],
                                    model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
        """
        计算语义相似度（需要sentence-transformers）
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            model_name: 使用的模型名称
        Returns:
            平均语义相似度
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            model = SentenceTransformer(model_name)
            
            pred_embeddings = model.encode(predictions)
            ref_embeddings = model.encode(references)
            
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            return np.mean(similarities)
            
        except ImportError:
            print("sentence-transformers未安装，无法计算语义相似度")
            return 0.0
        except Exception as e:
            print(f"语义相似度计算失败: {e}")
            return 0.0
    
    def evaluate_generation(self, predictions: List[str], references: List[Union[str, List[str]]],
                          metrics: List[str] = None) -> Dict[str, float]:
        """
        综合评估文本生成结果
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            metrics: 要计算的指标列表
        Returns:
            评估结果字典
        """
        if metrics is None:
            metrics = ['bleu', 'rouge', 'exact_match', 'f1']
        
        results = {}
        
        # 标准化references格式
        normalized_refs = []
        single_refs = []
        for ref in references:
            if isinstance(ref, list):
                normalized_refs.append(ref)
                single_refs.append(ref[0] if ref else "")
            else:
                normalized_refs.append([ref])
                single_refs.append(ref)
        
        # 计算各种指标
        if 'bleu' in metrics:
            try:
                bleu_scores = self.calculate_bleu(predictions, normalized_refs)
                results.update(bleu_scores)
            except Exception as e:
                print(f"BLEU计算失败: {e}")
        
        if 'rouge' in metrics:
            try:
                rouge_scores = self.calculate_rouge(predictions, single_refs)
                results.update(rouge_scores)
            except Exception as e:
                print(f"ROUGE计算失败: {e}")
        
        if 'exact_match' in metrics:
            try:
                em_score = self.calculate_exact_match(predictions, single_refs)
                results['exact_match'] = em_score
            except Exception as e:
                print(f"精确匹配计算失败: {e}")
        
        if 'f1' in metrics:
            try:
                f1_score = self.calculate_f1_score(predictions, single_refs)
                results['f1'] = f1_score
            except Exception as e:
                print(f"F1计算失败: {e}")
        
        if 'semantic_similarity' in metrics:
            try:
                sem_sim = self.calculate_semantic_similarity(predictions, single_refs)
                results['semantic_similarity'] = sem_sim
            except Exception as e:
                print(f"语义相似度计算失败: {e}")
        
        # 保存历史记录
        self.metrics_history.append({
            'timestamp': str(np.datetime64('now')),
            'metrics': results,
            'sample_count': len(predictions)
        })
        
        return results
    
    def save_results(self, results: Dict[str, float], file_path: str):
        """
        保存评估结果
        Args:
            results: 评估结果
            file_path: 保存路径
        """
        from .3_io_utils import IOUtils
        
        output_data = {
            'timestamp': str(np.datetime64('now')),
            'metrics': results,
            'history': self.metrics_history
        }
        
        IOUtils.write_json(output_data, file_path)

# 全局实例
metrics_calculator = MetricsCalculator()

def calculate_metrics(predictions: List[str], references: List[Union[str, List[str]]],
                     metrics: List[str] = None) -> Dict[str, float]:
    """便捷的指标计算函数"""
    return metrics_calculator.evaluate_generation(predictions, references, metrics)

if __name__ == "__main__":
    # 测试代码
    print("🧪 测试评估指标...")
    
    # 测试数据
    predictions = [
        "这是一个测试句子。",
        "机器学习是人工智能的一个分支。",
        "今天天气很好。"
    ]
    
    references = [
        "这是测试句子。",
        "机器学习是AI的重要分支。", 
        "今天的天气非常好。"
    ]
    
    # 计算指标
    calc = MetricsCalculator()
    results = calc.evaluate_generation(predictions, references)
    
    print("📊 评估结果:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")
    
    print("✅ 评估指标测试完成！")