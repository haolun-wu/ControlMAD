"""
Agent Chat Manager for Multi-Agent Debate System

This module implements a chat history management system for individual agents,
enabling better self-awareness through clear separation of agent messages
and natural multi-turn dialogue format.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class MessageRole(Enum):
    """Roles for messages in agent chat history."""
    SYSTEM = "system"
    USER = "user" 
    ASSISTANT = "assistant"
    OTHER_AGENT = "other_agent"
    DEBATE_MODERATOR = "debate_moderator"


@dataclass
class ChatMessage:
    """A single message in an agent's chat history."""
    role: MessageRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For debate-specific messages
    phase: Optional[str] = None  # "initial", "debate", "self_adjustment", "final"
    player_focus: Optional[str] = None  # Which player is being debated
    agent_name: Optional[str] = None  # Which agent sent this message (for other_agent role)
    round_number: Optional[int] = None  # Which debate round this message belongs to


@dataclass
class AgentChatHistory:
    """Chat history for a single agent throughout the debate session."""
    agent_name: str
    game_id: int
    messages: List[ChatMessage] = field(default_factory=list)
    
    def add_message(self, role: MessageRole, content: str, **kwargs) -> None:
        """Add a message to the chat history."""
        message = ChatMessage(
            role=role,
            content=content,
            **kwargs
        )
        self.messages.append(message)
    
    def get_messages_for_phase(self, phase: str) -> List[ChatMessage]:
        """Get all messages from a specific phase."""
        return [msg for msg in self.messages if msg.phase == phase]
    
    def get_messages_for_round(self, round_number: int) -> List[ChatMessage]:
        """Get all messages from a specific debate round."""
        return [msg for msg in self.messages if msg.round_number == round_number]
    
    def get_conversation_context(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation context in format suitable for API calls."""
        context_messages = self.messages
        if max_messages:
            context_messages = self.messages[-max_messages:]
        
        formatted_messages = []
        for msg in context_messages:
            # Convert MessageRole to API format
            if msg.role == MessageRole.SYSTEM:
                role_str = "system"
            elif msg.role == MessageRole.USER:
                role_str = "user"
            elif msg.role == MessageRole.ASSISTANT:
                role_str = "assistant"
            elif msg.role == MessageRole.OTHER_AGENT:
                role_str = "user"  # Other agents' messages are treated as user input
            elif msg.role == MessageRole.DEBATE_MODERATOR:
                role_str = "user"  # Debate moderator messages are treated as user input
            else:
                role_str = "user"
            
            formatted_messages.append({
                "role": role_str,
                "content": msg.content
            })
        
        return formatted_messages


class AgentChatManager:
    """Manages chat histories for all agents in a debate session."""
    
    def __init__(self, game_id: int, agent_names: List[str]):
        self.game_id = game_id
        self.agent_histories: Dict[str, AgentChatHistory] = {}
        
        # Initialize chat history for each agent
        for agent_name in agent_names:
            self.agent_histories[agent_name] = AgentChatHistory(
                agent_name=agent_name,
                game_id=game_id
            )
    
    def get_agent_history(self, agent_name: str) -> AgentChatHistory:
        """Get chat history for a specific agent."""
        return self.agent_histories.get(agent_name)
    
    def add_system_message(self, agent_name: str, content: str, **kwargs) -> None:
        """Add a system message to an agent's chat history."""
        history = self.get_agent_history(agent_name)
        if history:
            history.add_message(MessageRole.SYSTEM, content, **kwargs)
    
    def add_user_message(self, agent_name: str, content: str, **kwargs) -> None:
        """Add a user message to an agent's chat history."""
        history = self.get_agent_history(agent_name)
        if history:
            history.add_message(MessageRole.USER, content, **kwargs)
    
    def add_assistant_message(self, agent_name: str, content: str, **kwargs) -> None:
        """Add an assistant (agent's own) message to chat history."""
        history = self.get_agent_history(agent_name)
        if history:
            history.add_message(MessageRole.ASSISTANT, content, **kwargs)
    
    def add_other_agent_message(self, agent_name: str, other_agent_name: str, 
                               content: str, **kwargs) -> None:
        """Add another agent's message to an agent's chat history."""
        history = self.get_agent_history(agent_name)
        if history:
            history.add_message(
                MessageRole.OTHER_AGENT, 
                content, 
                agent_name=other_agent_name,
                **kwargs
            )
    
    def add_debate_moderator_message(self, agent_name: str, content: str, **kwargs) -> None:
        """Add a debate moderator message to an agent's chat history."""
        history = self.get_agent_history(agent_name)
        if history:
            history.add_message(MessageRole.DEBATE_MODERATOR, content, **kwargs)
    
    def create_initial_system_message(self, agent_name: str, game_text: str, 
                                    num_players: int, self_reported_confidence: bool = False) -> None:
        """Create the initial system message for an agent."""
        from debate.prompts_chat import get_kks_chat_system_prompt_with_confidence
        
        # Include agent name for clear identity in initial prompt
        system_content = get_kks_chat_system_prompt_with_confidence(num_players, self_reported_confidence, agent_name)
        
        self.add_system_message(
            agent_name, 
            system_content,
            phase="initial"
        )
    
    def create_game_context_message(self, agent_name: str, game_text: str) -> None:
        """Add the game context as a user message."""
        self.add_user_message(
            agent_name,
            game_text,
            phase="initial"
        )
    
    def create_debate_context_message(self, agent_name: str, debate_prompt: str, 
                                    player_focus: str, round_number: int) -> None:
        """Add debate context as a moderator message."""
        self.add_debate_moderator_message(
            agent_name,
            debate_prompt,
            phase="debate",
            player_focus=player_focus,
            round_number=round_number
        )
    
    def create_self_adjustment_context_message(self, agent_name: str, adjustment_prompt: str, 
                                             player_focus: str, round_number: int) -> None:
        """Add self-adjustment context as a moderator message."""
        self.add_debate_moderator_message(
            agent_name,
            adjustment_prompt,
            phase="self_adjustment",
            player_focus=player_focus,
            round_number=round_number
        )
    
    def create_final_discussion_message(self, agent_name: str, final_prompt: str) -> None:
        """Add final discussion context as a moderator message."""
        self.add_debate_moderator_message(
            agent_name,
            final_prompt,
            phase="final"
        )
    
    def add_agent_response(self, agent_name: str, response_content: str, 
                          phase: str, **kwargs) -> None:
        """Add an agent's response to their own chat history."""
        self.add_assistant_message(
            agent_name,
            response_content,
            phase=phase,
            **kwargs
        )
    
    def add_other_agents_context(self, target_agent: str, other_agents_responses: List[Dict[str, Any]], 
                                phase: str, round_number: Optional[int] = None) -> None:
        """Add context about other agents' responses to a target agent's chat history."""
        for response in other_agents_responses:
            if response.get("agent_name") != target_agent:
                # Format the other agent's response for context
                content = self._format_other_agent_response(response, phase)
                self.add_other_agent_message(
                    target_agent,
                    response.get("agent_name", "Unknown"),
                    content,
                    phase=phase,
                    round_number=round_number
                )
    
    def _format_other_agent_response(self, response: Dict[str, Any], phase: str) -> str:
        """Format another agent's response for inclusion in chat history."""
        agent_name = response.get("agent_name", "Unknown")
        
        if phase == "initial":
            assignments = response.get("player_role_assignments", {})
            explanation = response.get("explanation", "")
            confidence = response.get("confidence", 0)
            
            conf_text = f" (confidence: {confidence})" if confidence >= 1 else ""
            
            return f"{agent_name} initially proposed: {assignments}{conf_text}\nTheir reasoning: {explanation}"
        
        elif phase == "debate":
            assignments = response.get("player_role_assignments", {})
            agree_with = response.get("agree_with", [])
            disagree_with = response.get("disagree_with", [])
            agree_reasoning = response.get("agree_reasoning", "")
            disagree_reasoning = response.get("disagree_reasoning", "")
            
            content = f"{agent_name} thinks: {assignments}\n"
            if agree_with:
                content += f"Agrees with: {', '.join(agree_with)}\n"
                if agree_reasoning:
                    content += f"Agree reasoning: {agree_reasoning}\n"
            if disagree_with:
                content += f"Disagrees with: {', '.join(disagree_with)}\n"
                if disagree_reasoning:
                    content += f"Disagree reasoning: {disagree_reasoning}\n"
            
            return content
        
        elif phase == "self_adjustment":
            assignments = response.get("player_role_assignments", {})
            explanation = response.get("explanation", "")
            confidence = response.get("confidence", 0)
            
            conf_text = f" (confidence: {confidence})" if confidence >= 1 else ""
            
            return f"{agent_name} adjusted to: {assignments}{conf_text}\nTheir reasoning: {explanation}"
        
        elif phase == "final":
            assignments = response.get("player_role_assignments", {})
            explanation = response.get("explanation", "")
            confidence = response.get("confidence", 0)
            
            conf_text = f" (confidence: {confidence})" if confidence >= 1 else ""
            
            return f"{agent_name} final decision: {assignments}{conf_text}\nTheir reasoning: {explanation}"
        
        else:
            return f"{agent_name} response: {response}"
    
    def get_chat_context_for_agent(self, agent_name: str, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Get formatted chat context for an agent, suitable for API calls."""
        history = self.get_agent_history(agent_name)
        if not history:
            return []
        
        return history.get_conversation_context(max_messages)
    
