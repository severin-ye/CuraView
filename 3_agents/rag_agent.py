#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG智能体 - 检索增强生成智能体
结合向量数据库检索和大语言模型生成能力
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# 添加父级路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "1_utils"))

from base_agent import BaseAgent
from logger import Logger
from config_loader import ConfigLoader

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    print(f"⚠️  RAG依赖未安装: {e}")
    print("请安装: pip install sentence-transformers faiss-cpu numpy")

class RAGAgent(BaseAgent):
    """RAG智能体 - 支持知识库检索的对话智能体"""
    
    def __init__(self, 
                 agent_name: str = "RAG助手",
                 config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 knowledge_base_path: Optional[str] = None):
        """
        初始化RAG智能体
        Args:
            agent_name: 智能体名称
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
            knowledge_base_path: 知识库路径
        """
        super().__init__(agent_name, config_path, checkpoint_path)
        
        # RAG特定配置
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = None
        self.vector_index = None
        self.documents = []
        self.document_embeddings = None
        
        # RAG参数
        self.top_k = self.config.get('top_k', 5)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.5)
        self.max_context_length = self.config.get('max_context_length', 2000)
        
        # 加载嵌入模型
        self.load_embedding_model()
        
        # 加载知识库
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
        
        self.logger.info("🔍 RAG智能体初始化完成")
    
    def get_system_prompt(self) -> str:
        """获取RAG智能体的系统提示词"""
        return """你是一个专业的RAG（检索增强生成）智能体。你具有以下能力：

1. 🔍 知识检索：从知识库中检索相关信息
2. 📚 上下文理解：基于检索到的信息回答问题
3. 🎯 准确回答：结合知识库内容提供准确的答案
4. 🔗 来源引用：在回答中标注信息来源

回答规则：
- 优先使用检索到的知识库信息
- 如果知识库中没有相关信息，请明确说明
- 对于检索到的信息，请在回答末尾注明来源
- 保持回答的准确性和相关性
- 避免编造不存在的信息

请根据用户的问题，检索相关知识并给出准确的回答。"""
    
    def load_embedding_model(self):
        """加载嵌入模型"""
        try:
            model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"✅ 嵌入模型加载成功: {model_name}")
        except Exception as e:
            self.logger.error(f"❌ 嵌入模型加载失败: {str(e)}")
            self.embedding_model = None
    
    def load_knowledge_base(self, kb_path: str):
        """
        加载知识库
        Args:
            kb_path: 知识库路径
        """
        try:
            kb_path_obj = Path(kb_path)
            
            if not kb_path_obj.exists():
                self.logger.warning(f"⚠️  知识库路径不存在: {kb_path_obj}")
                return
            
            # 根据文件类型加载知识库
            if kb_path_obj.suffix == '.json':
                self.load_json_knowledge_base(kb_path_obj)
            elif kb_path_obj.suffix == '.txt':
                self.load_text_knowledge_base(kb_path_obj)
            elif kb_path_obj.is_dir():
                self.load_directory_knowledge_base(kb_path_obj)
            else:
                self.logger.warning(f"⚠️  不支持的知识库格式: {kb_path_obj.suffix}")
                return
            
            # 构建向量索引
            self.build_vector_index()
            
            self.logger.info(f"✅ 知识库加载完成，共{len(self.documents)}个文档")
            
        except Exception as e:
            self.logger.error(f"❌ 知识库加载失败: {str(e)}")
    
    def load_json_knowledge_base(self, json_path: Path):
        """加载JSON格式知识库"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # 文档列表格式
            for item in data:
                if isinstance(item, dict):
                    content = item.get('content', str(item))
                    title = item.get('title', f"文档{len(self.documents) + 1}")
                    source = item.get('source', str(json_path))
                else:
                    content = str(item)
                    title = f"文档{len(self.documents) + 1}"
                    source = str(json_path)
                
                self.documents.append({
                    'content': content,
                    'title': title,
                    'source': source,
                    'id': len(self.documents)
                })
        elif isinstance(data, dict):
            # 键值对格式
            for key, value in data.items():
                self.documents.append({
                    'content': str(value),
                    'title': str(key),
                    'source': str(json_path),
                    'id': len(self.documents)
                })
    
    def load_text_knowledge_base(self, txt_path: Path):
        """加载文本格式知识库"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按段落分割
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            self.documents.append({
                'content': paragraph,
                'title': f"段落{i + 1}",
                'source': str(txt_path),
                'id': len(self.documents)
            })
    
    def load_directory_knowledge_base(self, dir_path: Path):
        """加载目录下的所有文档"""
        supported_formats = ['.txt', '.json', '.md']
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in supported_formats:
                try:
                    if file_path.suffix == '.json':
                        self.load_json_knowledge_base(file_path)
                    else:
                        # 处理文本文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        self.documents.append({
                            'content': content,
                            'title': file_path.stem,
                            'source': str(file_path),
                            'id': len(self.documents)
                        })
                        
                except Exception as e:
                    self.logger.warning(f"⚠️  文件加载失败 {file_path}: {e}")
    
    def build_vector_index(self):
        """构建向量索引"""
        if not self.embedding_model or not self.documents:
            self.logger.warning("⚠️  嵌入模型或文档为空，无法构建索引")
            return
        
        try:
            self.logger.info("🔨 开始构建向量索引...")
            
            # 提取文档内容
            doc_contents = [doc['content'] for doc in self.documents]
            
            # 生成嵌入
            embeddings = self.embedding_model.encode(doc_contents, show_progress_bar=True)
            self.document_embeddings = embeddings
            
            # 构建FAISS索引
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # 内积相似度
            
            # 归一化嵌入向量
            faiss.normalize_L2(embeddings)
            self.vector_index.add(embeddings.astype('float32'))
            
            self.logger.info(f"✅ 向量索引构建完成，维度: {dimension}")
            
        except Exception as e:
            self.logger.error(f"❌ 向量索引构建失败: {str(e)}")
            self.vector_index = None
    
    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        检索相关文档
        Args:
            query: 查询文本
            top_k: 返回文档数量
        Returns:
            List[Dict[str, Any]]: 检索到的文档列表
        """
        if not self.vector_index or not self.embedding_model:
            self.logger.warning("⚠️  向量索引或嵌入模型未就绪")
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # 检索相似文档
            scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
            
            # 构建结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and score >= self.similarity_threshold:
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)
            
            self.logger.info(f"🔍 检索到{len(results)}个相关文档")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 文档检索失败: {str(e)}")
            return []
    
    def format_retrieved_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        格式化检索到的上下文
        Args:
            retrieved_docs: 检索到的文档列表
        Returns:
            str: 格式化的上下文
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            content = doc['content']
            title = doc['title']
            source = doc['source']
            score = doc.get('similarity_score', 0)
            
            # 格式化文档
            doc_text = f"[文档{i+1}] {title}\n来源: {source}\n相似度: {score:.3f}\n内容: {content}\n"
            
            # 检查长度限制
            if current_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n" + "="*50 + "\n".join(context_parts) + "="*50 + "\n"
    
    def process_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        处理用户输入（RAG流程）
        Args:
            user_input: 用户输入
            context: 上下文信息
        Returns:
            str: 处理结果
        """
        try:
            # 1. 检索相关文档
            retrieved_docs = self.retrieve_documents(user_input)
            
            if not retrieved_docs:
                # 没有检索到相关文档，使用基础回答
                if self.model_loaded:
                    base_prompt = f"用户问题: {user_input}\n\n注意：知识库中没有找到相关信息，请基于你的基础知识回答，并说明这不是基于知识库的回答。"
                    return self.infer(base_prompt)
                else:
                    return "抱歉，我在知识库中没有找到相关信息，且模型未加载。请检查问题或联系管理员。"
            
            # 2. 格式化检索上下文
            context_text = self.format_retrieved_context(retrieved_docs)
            
            # 3. 构建增强提示
            enhanced_prompt = f"""基于以下检索到的知识库信息回答用户问题：

{context_text}

用户问题: {user_input}

请基于上述知识库信息回答问题，并在回答末尾注明参考的文档来源。如果知识库信息不足以完全回答问题，请说明。"""
            
            # 4. 生成回答
            if self.model_loaded:
                response = self.infer(enhanced_prompt, max_tokens=1024)
                
                # 5. 后处理：添加来源信息
                sources = list(set([doc['source'] for doc in retrieved_docs]))
                source_info = f"\n\n📚 参考来源:\n" + "\n".join([f"• {source}" for source in sources[:3]])
                
                return response + source_info
            else:
                # 模型未加载时的简化回答
                return f"基于知识库检索，我找到了{len(retrieved_docs)}个相关文档。但由于模型未加载，无法生成详细回答。\n\n检索到的关键信息:\n" + \
                       "\n".join([f"• {doc['title']}: {doc['content'][:100]}..." for doc in retrieved_docs[:3]])
        
        except Exception as e:
            self.logger.error(f"❌ RAG处理失败: {str(e)}")
            return f"抱歉，在处理您的问题时遇到了错误: {str(e)}"
    
    def add_document(self, content: str, title: str, source: str = "manual_add"):
        """
        添加新文档到知识库
        Args:
            content: 文档内容
            title: 文档标题
            source: 文档来源
        """
        try:
            doc = {
                'content': content,
                'title': title,
                'source': source,
                'id': len(self.documents)
            }
            
            self.documents.append(doc)
            
            # 重新构建索引
            self.build_vector_index()
            
            self.logger.info(f"✅ 文档添加成功: {title}")
            
        except Exception as e:
            self.logger.error(f"❌ 文档添加失败: {str(e)}")
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        搜索文档（不生成回答）
        Args:
            query: 搜索查询
            top_k: 返回文档数量
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        return self.retrieve_documents(query, top_k)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.documents:
            return {
                'total_documents': 0,
                'total_characters': 0,
                'sources': [],
                'index_ready': False
            }
        
        total_chars = sum(len(doc['content']) for doc in self.documents)
        sources = list(set(doc['source'] for doc in self.documents))
        
        return {
            'total_documents': len(self.documents),
            'total_characters': total_chars,
            'average_doc_length': total_chars / len(self.documents),
            'sources': sources,
            'index_ready': self.vector_index is not None,
            'embedding_model': getattr(self.embedding_model, 'model_name', 'unknown') if self.embedding_model else None
        }
    
    def get_capabilities(self) -> List[str]:
        """获取RAG智能体能力列表"""
        base_capabilities = super().get_capabilities()
        rag_capabilities = [
            "知识库检索",
            "向量相似度搜索",
            "上下文增强生成",
            "来源引用",
            "文档管理",
            "实时知识更新"
        ]
        return base_capabilities + rag_capabilities

def create_rag_agent(agent_name: str = "RAG助手",
                    config_path: Optional[str] = None,
                    checkpoint_path: Optional[str] = None,
                    knowledge_base_path: Optional[str] = None) -> RAGAgent:
    """
    创建RAG智能体实例
    Args:
        agent_name: 智能体名称
        config_path: 配置文件路径
        checkpoint_path: 模型检查点路径
        knowledge_base_path: 知识库路径
    Returns:
        RAGAgent: RAG智能体实例
    """
    return RAGAgent(agent_name, config_path, checkpoint_path, knowledge_base_path)

if __name__ == "__main__":
    # 示例用法
    print("🧪 RAG智能体测试...")
    
    # 创建测试知识库
    test_kb = [
        {
            "title": "人工智能简介",
            "content": "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
            "source": "AI教程"
        },
        {
            "title": "机器学习",
            "content": "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式，无需明确编程。",
            "source": "ML教程"
        }
    ]
    
    # 保存测试知识库
    test_kb_path = "test_knowledge_base.json"
    with open(test_kb_path, 'w', encoding='utf-8') as f:
        json.dump(test_kb, f, ensure_ascii=False, indent=2)
    
    # 创建RAG智能体
    rag_agent = create_rag_agent(
        agent_name="测试RAG助手",
        knowledge_base_path=test_kb_path
    )
    
    # 测试检索
    results = rag_agent.search_documents("什么是人工智能")
    print(f"🔍 检索结果: {len(results)}个文档")
    
    # 获取统计信息
    stats = rag_agent.get_knowledge_base_stats()
    print(f"📊 知识库统计: {stats}")
    
    # 清理测试文件
    os.remove(test_kb_path)
    
    print("✅ RAG智能体模块加载成功")