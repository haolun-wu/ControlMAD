"""
Chat History Multi-Agent Debate System

This package contains the enhanced debate system implementation with individual chat histories
for each agent, providing better self-awareness through natural multi-turn dialogue format.
"""

from .debate_system_chat import ChatHistoryDebateSystem
from .agent_chat_manager import AgentChatManager, AgentChatHistory, ChatMessage, MessageRole
from .debate_config import DebateConfig, AgentConfig

__all__ = [
    'ChatHistoryDebateSystem', 
    'AgentChatManager', 
    'AgentChatHistory', 
    'ChatMessage', 
    'MessageRole',
    'DebateConfig', 
    'AgentConfig'
]
