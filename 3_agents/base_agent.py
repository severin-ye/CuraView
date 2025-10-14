#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础智能体模块 - 定义智能体的基础接口和通用功能
提供所有专业智能体的父类和共享方法
"""

import sys
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable

# 添加utils和core路径
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / "1_utils"))
sys.path.append(str(current_dir / "2_core"))

from logger import Logger
from config_loader import ConfigLoader
# 修复导入路径
sys.path.append(str(current_dir / "2_core"))
import importlib.util
spec = importlib.util.spec_from_file_location("core_api", current_dir / "2_core" / "__init__.py")
core_api_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_api_module)
get_core_api = core_api_module.get_core_api

class BaseAgent(ABC):
    """基础智能体类 - 所有专业智能体的父类"""
    
    def __init__(self, 
                 agent_name: str,
                 config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None):
        """
        初始化基础智能体
        Args:
            agent_name: 智能体名称
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
        """
        self.agent_name = agent_name
        self.logger = Logger(f"Agent_{agent_name}").get_logger()
        self.config_loader = ConfigLoader()
        
        # 加载配置
        self.config = {}
        if config_path:
            self.config = self.config_loader.load_config(config_path)
        
        # 获取核心API
        self.core_api = get_core_api()
        
        # 模型相关
        self.checkpoint_path = checkpoint_path
        self.model_loaded = False
        
        # 对话历史
        self.conversation_history = []
        
        # 智能体状态
        self.status = "initialized"
        self.metadata = {
            'created_time': time.time(),
            'agent_type': self.__class__.__name__,
            'version': '1.0.0'
        }
        
        self.logger.info(f"🤖 智能体 {agent_name} 初始化完成")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        获取系统提示词
        Returns:
            str: 系统提示词
        """
        pass
    
    @abstractmethod
    def process_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        处理用户输入
        Args:
            user_input: 用户输入
            context: 上下文信息
        Returns:
            str: 处理结果
        """
        pass
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        加载模型
        Args:
            checkpoint_path: 检查点路径
        """
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        
        if not self.checkpoint_path:
            raise ValueError("❌ 未提供模型检查点路径")
        
        try:
            self.core_api.load_model_for_inference(self.checkpoint_path)
            self.model_loaded = True
            self.status = "ready"
            self.logger.info(f"✅ 模型加载成功: {self.checkpoint_path}")
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {str(e)}")
            self.status = "error"
            raise e
    
    def infer(self, 
             prompt: str, 
             max_tokens: int = 512, 
             temperature: float = 0.7) -> str:
        """
        执行推理
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 生成温度
        Returns:
            str: 推理结果
        """
        if not self.model_loaded:
            raise ValueError("❌ 模型未加载，请先调用load_model()")
        
        try:
            # 构建完整提示（包含系统提示）
            system_prompt = self.get_system_prompt()
            full_prompt = f"{system_prompt}\n\n用户: {prompt}\n助手:"
            
            # 执行推理
            response = self.core_api.infer_single(
                full_prompt, 
                self.checkpoint_path,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"❌ 推理失败: {str(e)}")
            return f"抱歉，处理您的请求时出现了错误: {str(e)}"
    
    def chat(self, 
            user_input: str, 
            context: Optional[Dict[str, Any]] = None,
            save_history: bool = True) -> str:
        """
        对话接口
        Args:
            user_input: 用户输入
            context: 上下文信息
            save_history: 是否保存对话历史
        Returns:
            str: 回复内容
        """
        try:
            # 记录用户输入
            if save_history:
                self.conversation_history.append({
                    'timestamp': time.time(),
                    'role': 'user',
                    'content': user_input,
                    'context': context
                })
            
            # 处理输入
            response = self.process_input(user_input, context)
            
            # 记录助手回复
            if save_history:
                self.conversation_history.append({
                    'timestamp': time.time(),
                    'role': 'assistant',
                    'content': response
                })
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ 对话处理失败: {str(e)}")
            error_response = "抱歉，我遇到了一些问题，请稍后重试。"
            
            if save_history:
                self.conversation_history.append({
                    'timestamp': time.time(),
                    'role': 'assistant',
                    'content': error_response,
                    'error': str(e)
                })
            
            return error_response
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取对话历史
        Args:
            limit: 限制返回的对话数量
        Returns:
            List[Dict[str, Any]]: 对话历史列表
        """
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        self.logger.info("🧹 对话历史已清空")
    
    def save_conversation_history(self, filepath: str):
        """
        保存对话历史到文件
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'agent_name': self.agent_name,
                    'agent_type': self.metadata['agent_type'],
                    'export_time': time.time(),
                    'conversation_history': self.conversation_history
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 对话历史已保存到: {filepath}")
        except Exception as e:
            self.logger.error(f"❌ 保存对话历史失败: {str(e)}")
    
    def load_conversation_history(self, filepath: str):
        """
        从文件加载对话历史
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.conversation_history = data.get('conversation_history', [])
            
            self.logger.info(f"📂 对话历史已从文件加载: {filepath}")
        except Exception as e:
            self.logger.error(f"❌ 加载对话历史失败: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取智能体状态
        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            'agent_name': self.agent_name,
            'status': self.status,
            'model_loaded': self.model_loaded,
            'checkpoint_path': self.checkpoint_path,
            'conversation_count': len(self.conversation_history),
            'metadata': self.metadata.copy()
        }
    
    def reset(self):
        """重置智能体状态"""
        self.clear_conversation_history()
        self.status = "ready" if self.model_loaded else "initialized"
        self.logger.info("🔄 智能体状态已重置")
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        更新配置
        Args:
            new_config: 新配置
        """
        self.config.update(new_config)
        self.logger.info("⚙️  配置已更新")
    
    def get_capabilities(self) -> List[str]:
        """
        获取智能体能力列表
        Returns:
            List[str]: 能力列表
        """
        return [
            "自然语言理解",
            "对话交互",
            "上下文记忆",
            "配置管理",
            "状态监控"
        ]
    
    def interactive_session(self):
        """启动交互式会话"""
        self.logger.info("🎮 启动交互式会话")
        print(f"\n{'='*50}")
        print(f"🤖 {self.agent_name} 智能体已就绪")
        print(f"📋 类型: {self.metadata['agent_type']}")
        print(f"💡 输入 'quit', 'exit' 或 '退出' 结束会话")
        print(f"💡 输入 'help' 查看帮助信息")
        print(f"{'='*50}")
        
        while True:
            try:
                user_input = input(f"\n👤 您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 会话结束，再见！")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if not user_input:
                    continue
                
                print(f"🤖 {self.agent_name}: ", end="")
                response = self.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 会话已中断，再见！")
                break
            except Exception as e:
                self.logger.error(f"❌ 会话过程中出错: {str(e)}")
                print(f"😓 抱歉，出现了一些问题: {str(e)}")
    
    def show_help(self):
        """显示帮助信息"""
        print(f"\n📖 {self.agent_name} 帮助信息:")
        print(f"🔹 智能体类型: {self.metadata['agent_type']}")
        print("🔹 可用命令:")
        print("  • help - 显示此帮助信息")
        print("  • quit/exit/退出 - 结束会话")
        print("🔹 能力:")
        for capability in self.get_capabilities():
            print(f"  • {capability}")

class AgentManager:
    """智能体管理器 - 管理多个智能体实例"""
    
    def __init__(self):
        self.logger = Logger("AgentManager").get_logger()
        self.agents: Dict[str, BaseAgent] = {}
        self.logger.info("🎯 智能体管理器初始化完成")
    
    def register_agent(self, agent: BaseAgent):
        """
        注册智能体
        Args:
            agent: 智能体实例
        """
        self.agents[agent.agent_name] = agent
        self.logger.info(f"📝 智能体已注册: {agent.agent_name}")
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        获取智能体
        Args:
            agent_name: 智能体名称
        Returns:
            Optional[BaseAgent]: 智能体实例
        """
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """
        列出所有智能体
        Returns:
            List[str]: 智能体名称列表
        """
        return list(self.agents.keys())
    
    def remove_agent(self, agent_name: str) -> bool:
        """
        移除智能体
        Args:
            agent_name: 智能体名称
        Returns:
            bool: 是否成功移除
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            self.logger.info(f"🗑️  智能体已移除: {agent_name}")
            return True
        return False
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有智能体状态
        Returns:
            Dict[str, Dict[str, Any]]: 所有智能体的状态
        """
        return {name: agent.get_status() for name, agent in self.agents.items()}

# 全局智能体管理器实例
_agent_manager = AgentManager()

def get_agent_manager() -> AgentManager:
    """获取全局智能体管理器"""
    return _agent_manager

if __name__ == "__main__":
    # 示例：基础智能体的简单实现
    class TestAgent(BaseAgent):
        def get_system_prompt(self) -> str:
            return "你是一个测试智能体，友好且乐于助人。"
        
        def process_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
            if not self.model_loaded:
                return "模型未加载，这是一个测试回复。"
            return self.infer(user_input)
    
    print("🧪 基础智能体测试...")
    
    # 创建测试智能体
    test_agent = TestAgent("测试智能体")
    
    # 注册到管理器
    manager = get_agent_manager()
    manager.register_agent(test_agent)
    
    print(f"✅ 智能体状态: {test_agent.get_status()}")
    print(f"📋 已注册的智能体: {manager.list_agents()}")
    print("✅ 基础智能体模块加载成功")