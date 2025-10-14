#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好学习智能体 - 基于用户反馈进行偏好学习的智能体
支持从人类反馈中学习（RLHF）和偏好对齐
"""

import sys
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict, deque

# 添加父级路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "1_utils"))

from base_agent import BaseAgent
from logger import Logger
from config_loader import ConfigLoader

class PreferenceAgent(BaseAgent):
    """偏好学习智能体 - 基于用户反馈学习偏好"""
    
    def __init__(self, 
                 agent_name: str = "偏好学习助手",
                 config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 preference_data_path: Optional[str] = None):
        """
        初始化偏好学习智能体
        Args:
            agent_name: 智能体名称
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
            preference_data_path: 偏好数据文件路径
        """
        super().__init__(agent_name, config_path, checkpoint_path)
        
        # 偏好学习相关配置
        self.preference_data_path = preference_data_path
        self.max_history_size = self.config.get('max_history_size', 1000)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.feedback_threshold = self.config.get('feedback_threshold', 3)
        
        # 偏好数据存储
        self.preference_history = deque(maxlen=self.max_history_size)
        self.user_preferences = defaultdict(dict)  # user_id -> preferences
        self.response_quality_scores = defaultdict(list)  # response_pattern -> scores
        self.feedback_patterns = defaultdict(int)  # pattern -> count
        
        # 当前用户会话信息
        self.current_user_id = None
        self.session_feedback = []
        
        # 加载历史偏好数据
        if preference_data_path:
            self.load_preference_data()
        
        self.logger.info("🎯 偏好学习智能体初始化完成")
    
    def get_system_prompt(self) -> str:
        """获取偏好学习智能体的系统提示词"""
        return """你是一个智能的偏好学习助手，具有以下特点：

1. 📊 学习能力：从用户反馈中学习偏好模式
2. 🎯 个性化：根据用户历史偏好调整回答风格
3. 🔄 适应性：持续优化回答质量和相关性
4. 💬 互动性：主动征求用户反馈以改进服务

回答原则：
- 根据已学习的用户偏好调整回答风格
- 优先使用用户偏好的回答模式
- 在适当时候征求用户反馈
- 保持回答的一致性和个性化
- 记录和分析用户的满意度

请基于用户的历史偏好和当前需求，提供个性化的回答。"""
    
    def set_current_user(self, user_id: str):
        """
        设置当前用户
        Args:
            user_id: 用户ID
        """
        self.current_user_id = user_id
        self.session_feedback = []
        self.logger.info(f"👤 当前用户设置为: {user_id}")
    
    def process_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        处理用户输入（考虑偏好）
        Args:
            user_input: 用户输入
            context: 上下文信息
        Returns:
            str: 处理结果
        """
        try:
            # 获取用户偏好
            user_prefs = self.get_user_preferences()
            
            # 构建个性化提示
            personalized_prompt = self.build_personalized_prompt(user_input, user_prefs)
            
            # 生成回答
            if self.model_loaded:
                response = self.infer(personalized_prompt, 
                                    temperature=user_prefs.get('temperature', 0.7),
                                    max_tokens=user_prefs.get('max_tokens', 512))
            else:
                response = f"基于您的偏好设置回答: {user_input}（模型未加载，这是示例回答）"
            
            # 记录交互
            self.record_interaction(user_input, response, context)
            
            # 添加反馈邀请
            if self.should_request_feedback():
                response += "\n\n💡 这个回答对您有帮助吗？您可以给出反馈（好/不好/建议）来帮助我改进。"
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ 偏好处理失败: {str(e)}")
            return f"抱歉，在为您个性化回答时遇到了问题: {str(e)}"
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """
        获取当前用户的偏好设置
        Returns:
            Dict[str, Any]: 用户偏好
        """
        if not self.current_user_id:
            return self.get_default_preferences()
        
        return self.user_preferences.get(self.current_user_id, self.get_default_preferences())
    
    def get_default_preferences(self) -> Dict[str, Any]:
        """获取默认偏好设置"""
        return {
            'response_style': 'balanced',  # formal, casual, balanced
            'detail_level': 'medium',      # brief, medium, detailed
            'temperature': 0.7,
            'max_tokens': 512,
            'include_examples': True,
            'use_emojis': True,
            'language_preference': 'zh',
            'topic_interests': []
        }
    
    def build_personalized_prompt(self, user_input: str, preferences: Dict[str, Any]) -> str:
        """
        构建个性化提示
        Args:
            user_input: 用户输入
            preferences: 用户偏好
        Returns:
            str: 个性化提示
        """
        # 基础提示
        prompt_parts = [f"用户问题: {user_input}"]
        
        # 添加风格指导
        style = preferences.get('response_style', 'balanced')
        if style == 'formal':
            prompt_parts.append("请使用正式、专业的语言风格回答。")
        elif style == 'casual':
            prompt_parts.append("请使用轻松、友好的语言风格回答。")
        else:
            prompt_parts.append("请使用平衡的语言风格，既专业又易懂。")
        
        # 添加详细程度指导
        detail = preferences.get('detail_level', 'medium')
        if detail == 'brief':
            prompt_parts.append("请简洁回答，重点突出关键信息。")
        elif detail == 'detailed':
            prompt_parts.append("请详细回答，包含充分的解释和背景信息。")
        else:
            prompt_parts.append("请适度详细地回答，平衡深度和可读性。")
        
        # 添加示例偏好
        if preferences.get('include_examples', True):
            prompt_parts.append("如果适合，请在回答中包含具体的例子或案例。")
        
        # 添加表情偏好
        if preferences.get('use_emojis', True):
            prompt_parts.append("可以适当使用表情符号使回答更生动。")
        
        # 添加兴趣话题关联
        interests = preferences.get('topic_interests', [])
        if interests and any(interest.lower() in user_input.lower() for interest in interests):
            prompt_parts.append(f"注意用户对以下话题特别感兴趣: {', '.join(interests)}。")
        
        return "\n".join(prompt_parts)
    
    def record_interaction(self, user_input: str, response: str, context: Optional[Dict[str, Any]]):
        """
        记录交互信息
        Args:
            user_input: 用户输入
            response: 回答
            context: 上下文
        """
        interaction = {
            'timestamp': time.time(),
            'user_id': self.current_user_id,
            'user_input': user_input,
            'response': response,
            'context': context,
            'session_id': id(self.session_feedback)  # 简单的会话ID
        }
        
        self.preference_history.append(interaction)
        self.session_feedback.append(interaction)
    
    def should_request_feedback(self) -> bool:
        """
        判断是否应该请求反馈
        Returns:
            bool: 是否请求反馈
        """
        # 每隔几轮对话请求一次反馈
        return len(self.session_feedback) % self.feedback_threshold == 0
    
    def process_feedback(self, feedback: str, interaction_index: int = -1) -> str:
        """
        处理用户反馈
        Args:
            feedback: 用户反馈
            interaction_index: 交互索引（-1表示最后一次）
        Returns:
            str: 反馈处理结果
        """
        try:
            if not self.session_feedback:
                return "没有找到可以反馈的交互记录。"
            
            # 获取目标交互
            if interaction_index == -1 or interaction_index >= len(self.session_feedback):
                target_interaction = self.session_feedback[-1]
            else:
                target_interaction = self.session_feedback[interaction_index]
            
            # 解析反馈
            feedback_score = self.parse_feedback(feedback)
            
            # 更新偏好
            self.update_preferences_from_feedback(target_interaction, feedback, feedback_score)
            
            # 记录反馈
            feedback_record = {
                'timestamp': time.time(),
                'user_id': self.current_user_id,
                'interaction': target_interaction,
                'feedback': feedback,
                'feedback_score': feedback_score
            }
            
            self.preference_history.append(feedback_record)
            
            # 保存偏好数据
            self.save_preference_data()
            
            # 返回确认信息
            if feedback_score > 0:
                return "谢谢您的正面反馈！我会继续保持这种回答风格。"
            elif feedback_score < 0:
                return "感谢您的反馈，我会根据您的建议调整回答方式。"
            else:
                return "谢谢您的反馈，我已记录您的意见。"
                
        except Exception as e:
            self.logger.error(f"❌ 反馈处理失败: {str(e)}")
            return f"处理反馈时出现错误: {str(e)}"
    
    def parse_feedback(self, feedback: str) -> float:
        """
        解析反馈为分数
        Args:
            feedback: 反馈文本
        Returns:
            float: 反馈分数 (-1到1之间)
        """
        feedback_lower = feedback.lower().strip()
        
        # 正面反馈
        positive_keywords = ['好', '很好', '棒', '不错', '满意', '喜欢', 'good', 'great', 'excellent', '👍']
        if any(keyword in feedback_lower for keyword in positive_keywords):
            return 1.0
        
        # 负面反馈
        negative_keywords = ['不好', '差', '不满意', '不喜欢', '糟糕', 'bad', 'poor', '👎']
        if any(keyword in feedback_lower for keyword in negative_keywords):
            return -1.0
        
        # 中性反馈或建议
        neutral_keywords = ['建议', '希望', '可以', '应该', 'suggest', 'recommend']
        if any(keyword in feedback_lower for keyword in neutral_keywords):
            return 0.0
        
        # 默认为中性
        return 0.0
    
    def update_preferences_from_feedback(self, interaction: Dict[str, Any], feedback: str, score: float):
        """
        根据反馈更新用户偏好
        Args:
            interaction: 交互记录
            feedback: 反馈内容
            score: 反馈分数
        """
        if not self.current_user_id:
            return
        
        current_prefs = self.get_user_preferences()
        
        # 根据反馈调整偏好
        if score > 0:
            # 正面反馈：强化当前设置
            self.reinforce_preferences(current_prefs, interaction)
        elif score < 0:
            # 负面反馈：调整设置
            self.adjust_preferences(current_prefs, feedback, interaction)
        
        # 更新用户偏好
        self.user_preferences[self.current_user_id] = current_prefs
    
    def reinforce_preferences(self, preferences: Dict[str, Any], interaction: Dict[str, Any]):
        """
        强化当前偏好设置
        Args:
            preferences: 当前偏好
            interaction: 交互记录
        """
        # 记录成功的模式
        response_pattern = self.extract_response_pattern(interaction['response'])
        self.response_quality_scores[response_pattern].append(1.0)
        
        # 可以在这里实现更复杂的强化学习逻辑
        self.logger.info(f"🔥 强化偏好设置: {response_pattern}")
    
    def adjust_preferences(self, preferences: Dict[str, Any], feedback: str, interaction: Dict[str, Any]):
        """
        调整偏好设置
        Args:
            preferences: 当前偏好
            feedback: 反馈内容
            interaction: 交互记录
        """
        feedback_lower = feedback.lower()
        
        # 根据反馈内容调整特定偏好
        if '太长' in feedback_lower or 'too long' in feedback_lower:
            preferences['detail_level'] = 'brief'
            preferences['max_tokens'] = min(preferences['max_tokens'], 256)
        elif '太短' in feedback_lower or 'too short' in feedback_lower:
            preferences['detail_level'] = 'detailed'
            preferences['max_tokens'] = max(preferences['max_tokens'], 768)
        
        if '正式' in feedback_lower or 'formal' in feedback_lower:
            preferences['response_style'] = 'formal'
            preferences['use_emojis'] = False
        elif '随意' in feedback_lower or 'casual' in feedback_lower:
            preferences['response_style'] = 'casual'
            preferences['use_emojis'] = True
        
        if '例子' in feedback_lower or 'example' in feedback_lower:
            preferences['include_examples'] = True
        elif '简洁' in feedback_lower or 'concise' in feedback_lower:
            preferences['include_examples'] = False
            preferences['detail_level'] = 'brief'
        
        # 记录失败的模式
        response_pattern = self.extract_response_pattern(interaction['response'])
        self.response_quality_scores[response_pattern].append(-1.0)
        
        self.logger.info(f"🔧 调整偏好设置基于反馈: {feedback}")
    
    def extract_response_pattern(self, response: str) -> str:
        """
        提取回答的模式特征
        Args:
            response: 回答文本
        Returns:
            str: 模式标识
        """
        # 简单的模式提取逻辑
        features = []
        
        # 长度特征
        if len(response) < 100:
            features.append("short")
        elif len(response) > 500:
            features.append("long")
        else:
            features.append("medium")
        
        # 表情符号
        if any(emoji in response for emoji in ['😊', '🤖', '💡', '📊', '✅', '❌', '⚠️']):
            features.append("emoji")
        
        # 例子
        if '例如' in response or '比如' in response or 'example' in response.lower():
            features.append("examples")
        
        # 结构化
        if any(marker in response for marker in ['1.', '2.', '•', '-', '▪']):
            features.append("structured")
        
        return "_".join(sorted(features))
    
    def get_preference_summary(self) -> Dict[str, Any]:
        """
        获取偏好学习摘要
        Returns:
            Dict[str, Any]: 偏好摘要
        """
        total_interactions = len(self.preference_history)
        feedback_count = sum(1 for item in self.preference_history if 'feedback_score' in item)
        
        # 统计反馈分数
        feedback_scores = [item.get('feedback_score', 0) for item in self.preference_history if 'feedback_score' in item]
        avg_feedback = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
        
        # 最受欢迎的回答模式
        pattern_scores = {}
        for pattern, scores in self.response_quality_scores.items():
            pattern_scores[pattern] = sum(scores) / len(scores) if scores else 0
        
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1]) if pattern_scores else None
        
        return {
            'total_interactions': total_interactions,
            'feedback_count': feedback_count,
            'feedback_ratio': feedback_count / total_interactions if total_interactions > 0 else 0,
            'average_feedback_score': avg_feedback,
            'registered_users': len(self.user_preferences),
            'best_response_pattern': best_pattern,
            'pattern_scores': pattern_scores
        }
    
    def save_preference_data(self):
        """保存偏好数据"""
        if not self.preference_data_path:
            return
        
        try:
            data = {
                'user_preferences': dict(self.user_preferences),
                'preference_history': list(self.preference_history),
                'response_quality_scores': dict(self.response_quality_scores),
                'feedback_patterns': dict(self.feedback_patterns)
            }
            
            with open(self.preference_data_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"💾 偏好数据已保存: {self.preference_data_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存偏好数据失败: {str(e)}")
    
    def load_preference_data(self):
        """加载偏好数据"""
        if not self.preference_data_path or not Path(self.preference_data_path).exists():
            return
        
        try:
            with open(self.preference_data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.user_preferences = defaultdict(dict, data.get('user_preferences', {}))
            self.preference_history = deque(data.get('preference_history', []), maxlen=self.max_history_size)
            self.response_quality_scores = defaultdict(list, data.get('response_quality_scores', {}))
            self.feedback_patterns = defaultdict(int, data.get('feedback_patterns', {}))
            
            self.logger.info(f"📂 偏好数据已加载: {self.preference_data_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 加载偏好数据失败: {str(e)}")
    
    def get_capabilities(self) -> List[str]:
        """获取偏好学习智能体能力列表"""
        base_capabilities = super().get_capabilities()
        preference_capabilities = [
            "偏好学习",
            "用户反馈处理",
            "个性化回答",
            "适应性调整",
            "模式识别",
            "持续优化"
        ]
        return base_capabilities + preference_capabilities

def create_preference_agent(agent_name: str = "偏好学习助手",
                          config_path: Optional[str] = None,
                          checkpoint_path: Optional[str] = None,
                          preference_data_path: Optional[str] = None) -> PreferenceAgent:
    """
    创建偏好学习智能体实例
    Args:
        agent_name: 智能体名称
        config_path: 配置文件路径
        checkpoint_path: 模型检查点路径
        preference_data_path: 偏好数据文件路径
    Returns:
        PreferenceAgent: 偏好学习智能体实例
    """
    return PreferenceAgent(agent_name, config_path, checkpoint_path, preference_data_path)

if __name__ == "__main__":
    # 示例用法
    print("🧪 偏好学习智能体测试...")
    
    # 创建偏好学习智能体
    pref_agent = create_preference_agent(
        agent_name="测试偏好助手",
        preference_data_path="test_preferences.pkl"
    )
    
    # 设置用户
    pref_agent.set_current_user("test_user")
    
    # 模拟对话和反馈
    response1 = pref_agent.chat("什么是机器学习？")
    print(f"🤖 回答1: {response1}")
    
    # 模拟反馈
    feedback_result = pref_agent.process_feedback("太长了，希望简洁一些")
    print(f"💬 反馈处理: {feedback_result}")
    
    # 再次对话（应该应用偏好调整）
    response2 = pref_agent.chat("什么是深度学习？")
    print(f"🤖 回答2: {response2}")
    
    # 获取偏好摘要
    summary = pref_agent.get_preference_summary()
    print(f"📊 偏好摘要: {summary}")
    
    # 清理测试文件
    test_file = Path("test_preferences.pkl")
    if test_file.exists():
        test_file.unlink()
    
    print("✅ 偏好学习智能体模块加载成功")