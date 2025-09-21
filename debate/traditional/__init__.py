"""
Traditional Multi-Agent Debate System

This package contains the original debate system implementation with monolithic prompts
and explicit self-awareness instructions.
"""

from .debate_system import MultiAgentDebateSystem
from .debate_config import DebateConfig, AgentConfig
from .debate_visualizer import DebateVisualizer

__all__ = ['MultiAgentDebateSystem', 'DebateConfig', 'AgentConfig', 'DebateVisualizer']
