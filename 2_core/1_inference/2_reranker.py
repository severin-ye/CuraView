"""
输出重排序模块

提供多种重排序算法，用于优化生成结果的质量和相关性。
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from transformers import AutoModel, AutoTokenizer
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """重排序器基类"""
    
    @abstractmethod
    def rerank(self, 
               query: str, 
               candidates: List[str], 
               scores: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """
        重排序候选结果
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            scores: 原始分数（可选）
            
        Returns:
            重排序后的结果和分数
        """
        pass


class SemanticReranker(BaseReranker):
    """基于语义相似度的重排序器"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化语义重排序器
        
        Args:
            model_name: 语义模型名称
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model(self):
        """加载语义模型"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                self.model.to(self.device)
                logger.info(f"语义模型加载成功: {self.model_name}")
            except ImportError:
                # 备用方案：使用标准transformers
                self.model = AutoModel.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model.to(self.device)
                logger.info(f"使用标准transformers加载模型: {self.model_name}")
    
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本向量
        """
        self._load_model()
        
        if hasattr(self.model, 'encode'):
            # sentence-transformers
            embeddings = self.model.encode(texts, convert_to_tensor=True)
        else:
            # 标准transformers
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return F.normalize(embeddings, p=2, dim=1)
    
    def rerank(self, 
               query: str, 
               candidates: List[str], 
               scores: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """
        基于语义相似度重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            scores: 原始分数
            
        Returns:
            重排序后的结果
        """
        if not candidates:
            return []
        
        # 编码查询和候选文本
        query_embedding = self._encode_texts([query])
        candidate_embeddings = self._encode_texts(candidates)
        
        # 计算相似度
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(1), 
            candidate_embeddings.unsqueeze(0),
            dim=2
        ).squeeze(0)
        
        # 转换为numpy以便处理
        similarities = similarities.cpu().numpy()
        
        # 如果有原始分数，进行加权组合
        if scores is not None:
            scores = np.array(scores)
            # 归一化分数
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            normalized_similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
            
            # 加权组合（相似度权重更高）
            final_scores = 0.7 * normalized_similarities + 0.3 * normalized_scores
        else:
            final_scores = similarities
        
        # 排序
        sorted_indices = np.argsort(final_scores)[::-1]
        
        return [(candidates[i], float(final_scores[i])) for i in sorted_indices]


class DiversityReranker(BaseReranker):
    """基于多样性的重排序器"""
    
    def __init__(self, diversity_weight: float = 0.5):
        """
        初始化多样性重排序器
        
        Args:
            diversity_weight: 多样性权重
        """
        self.diversity_weight = diversity_weight
        self.semantic_reranker = SemanticReranker()
    
    def _calculate_diversity_score(self, candidate: str, selected: List[str]) -> float:
        """
        计算多样性分数
        
        Args:
            candidate: 候选文本
            selected: 已选择的文本列表
            
        Returns:
            多样性分数
        """
        if not selected:
            return 1.0
        
        # 计算与已选择文本的最大相似度
        all_texts = selected + [candidate]
        embeddings = self.semantic_reranker._encode_texts(all_texts)
        
        candidate_embedding = embeddings[-1:].unsqueeze(0)
        selected_embeddings = embeddings[:-1].unsqueeze(1)
        
        similarities = torch.cosine_similarity(
            candidate_embedding, 
            selected_embeddings, 
            dim=2
        ).squeeze()
        
        if len(selected) == 1:
            max_similarity = similarities.item()
        else:
            max_similarity = similarities.max().item()
        
        # 多样性分数 = 1 - 最大相似度
        return 1.0 - max_similarity
    
    def rerank(self, 
               query: str, 
               candidates: List[str], 
               scores: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """
        基于多样性重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            scores: 原始分数
            
        Returns:
            重排序后的结果
        """
        if not candidates:
            return []
        
        # 首先进行语义排序
        semantic_results = self.semantic_reranker.rerank(query, candidates, scores)
        
        # 贪婪选择保证多样性
        selected_results = []
        remaining_candidates = semantic_results.copy()
        
        while remaining_candidates:
            best_score = -1
            best_idx = 0
            
            for i, (candidate, semantic_score) in enumerate(remaining_candidates):
                # 计算多样性分数
                diversity_score = self._calculate_diversity_score(
                    candidate, 
                    [result[0] for result in selected_results]
                )
                
                # 组合分数
                combined_score = (
                    (1 - self.diversity_weight) * semantic_score + 
                    self.diversity_weight * diversity_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = i
            
            # 选择最佳候选
            selected_candidate = remaining_candidates.pop(best_idx)
            selected_results.append((selected_candidate[0], best_score))
        
        return selected_results


class LengthAwareReranker(BaseReranker):
    """长度感知重排序器"""
    
    def __init__(self, 
                 target_length: Optional[int] = None, 
                 length_weight: float = 0.2):
        """
        初始化长度感知重排序器
        
        Args:
            target_length: 目标长度
            length_weight: 长度权重
        """
        self.target_length = target_length
        self.length_weight = length_weight
        self.semantic_reranker = SemanticReranker()
    
    def _calculate_length_score(self, text: str) -> float:
        """
        计算长度分数
        
        Args:
            text: 文本
            
        Returns:
            长度分数
        """
        if self.target_length is None:
            return 1.0
        
        text_length = len(text.split())
        
        # 使用高斯函数计算长度偏好
        length_diff = abs(text_length - self.target_length)
        length_score = np.exp(-(length_diff ** 2) / (2 * (self.target_length * 0.3) ** 2))
        
        return length_score
    
    def rerank(self, 
               query: str, 
               candidates: List[str], 
               scores: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """
        基于长度感知重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            scores: 原始分数
            
        Returns:
            重排序后的结果
        """
        if not candidates:
            return []
        
        # 如果没有目标长度，自动估算
        if self.target_length is None:
            lengths = [len(candidate.split()) for candidate in candidates]
            self.target_length = int(np.median(lengths))
        
        # 语义排序
        semantic_results = self.semantic_reranker.rerank(query, candidates, scores)
        
        # 添加长度分数
        reranked_results = []
        for candidate, semantic_score in semantic_results:
            length_score = self._calculate_length_score(candidate)
            
            # 组合分数
            combined_score = (
                (1 - self.length_weight) * semantic_score + 
                self.length_weight * length_score
            )
            
            reranked_results.append((candidate, combined_score))
        
        # 重新排序
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results


class QualityReranker(BaseReranker):
    """质量评估重排序器"""
    
    def __init__(self):
        """初始化质量重排序器"""
        self.semantic_reranker = SemanticReranker()
    
    def _calculate_quality_score(self, text: str) -> float:
        """
        计算质量分数
        
        Args:
            text: 文本
            
        Returns:
            质量分数
        """
        quality_score = 1.0
        
        # 长度检查
        if len(text.strip()) < 10:
            quality_score *= 0.5
        
        # 重复检查
        sentences = text.split('。')
        if len(sentences) > 1:
            unique_sentences = set(sentences)
            repetition_ratio = len(unique_sentences) / len(sentences)
            quality_score *= repetition_ratio
        
        # 结构检查
        if text.count('？') > 3 or text.count('！') > 3:
            quality_score *= 0.8
        
        # 完整性检查
        if not text.strip().endswith(('。', '！', '？', '"', '"')):
            quality_score *= 0.9
        
        return quality_score
    
    def rerank(self, 
               query: str, 
               candidates: List[str], 
               scores: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """
        基于质量评估重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            scores: 原始分数
            
        Returns:
            重排序后的结果
        """
        if not candidates:
            return []
        
        # 语义排序
        semantic_results = self.semantic_reranker.rerank(query, candidates, scores)
        
        # 添加质量分数
        quality_results = []
        for candidate, semantic_score in semantic_results:
            quality_score = self._calculate_quality_score(candidate)
            
            # 组合分数
            combined_score = 0.7 * semantic_score + 0.3 * quality_score
            quality_results.append((candidate, combined_score))
        
        # 重新排序
        quality_results.sort(key=lambda x: x[1], reverse=True)
        
        return quality_results


class HybridReranker(BaseReranker):
    """混合重排序器"""
    
    def __init__(self, 
                 use_semantic: bool = True,
                 use_diversity: bool = True,
                 use_length_aware: bool = True,
                 use_quality: bool = True,
                 target_length: Optional[int] = None):
        """
        初始化混合重排序器
        
        Args:
            use_semantic: 是否使用语义重排序
            use_diversity: 是否使用多样性重排序
            use_length_aware: 是否使用长度感知重排序
            use_quality: 是否使用质量重排序
            target_length: 目标长度
        """
        self.rerankers = {}
        
        if use_semantic:
            self.rerankers['semantic'] = SemanticReranker()
        
        if use_diversity:
            self.rerankers['diversity'] = DiversityReranker()
        
        if use_length_aware:
            self.rerankers['length'] = LengthAwareReranker(target_length)
        
        if use_quality:
            self.rerankers['quality'] = QualityReranker()
    
    def rerank(self, 
               query: str, 
               candidates: List[str], 
               scores: Optional[List[float]] = None,
               reranker_weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        混合重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            scores: 原始分数
            reranker_weights: 重排序器权重
            
        Returns:
            重排序后的结果
        """
        if not candidates:
            return []
        
        if reranker_weights is None:
            reranker_weights = {name: 1.0 for name in self.rerankers.keys()}
        
        # 收集所有重排序结果
        all_results = {}
        for name, reranker in self.rerankers.items():
            results = reranker.rerank(query, candidates, scores)
            all_results[name] = {result[0]: result[1] for result in results}
        
        # 组合分数
        final_scores = {}
        total_weight = sum(reranker_weights.values())
        
        for candidate in candidates:
            combined_score = 0.0
            for name, weight in reranker_weights.items():
                if name in all_results and candidate in all_results[name]:
                    combined_score += weight * all_results[name][candidate]
            
            final_scores[candidate] = combined_score / total_weight
        
        # 排序
        sorted_results = sorted(
            final_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_results


# 重排序器工厂
class RerankerFactory:
    """重排序器工厂类"""
    
    @staticmethod
    def create_reranker(reranker_type: str, **kwargs) -> BaseReranker:
        """
        创建重排序器
        
        Args:
            reranker_type: 重排序器类型
            **kwargs: 参数
            
        Returns:
            重排序器实例
        """
        rerankers = {
            'semantic': SemanticReranker,
            'diversity': DiversityReranker,
            'length_aware': LengthAwareReranker,
            'quality': QualityReranker,
            'hybrid': HybridReranker
        }
        
        if reranker_type not in rerankers:
            raise ValueError(f"不支持的重排序器类型: {reranker_type}")
        
        return rerankers[reranker_type](**kwargs)


# 便捷函数
def rerank_candidates(query: str, 
                     candidates: List[str], 
                     method: str = 'semantic',
                     scores: Optional[List[float]] = None,
                     **kwargs) -> List[Tuple[str, float]]:
    """
    重排序候选结果的便捷函数
    
    Args:
        query: 查询文本
        candidates: 候选结果列表
        method: 重排序方法
        scores: 原始分数
        **kwargs: 其他参数
        
    Returns:
        重排序后的结果
    """
    reranker = RerankerFactory.create_reranker(method, **kwargs)
    return reranker.rerank(query, candidates, scores)