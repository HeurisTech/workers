from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Type, TypeVar
from langchain_core.runnables import RunnableConfig, ensure_config

from langgraph_mcp.planner_style import prompts

def _get_merged_mcp_config(runtime_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load default tools and merge with runtime config."""
    default_tools = json.load(open(Path(__file__).parent.parent.parent.parent / "default-tools.json"))
    if runtime_config is None:
        return default_tools
    return {**default_tools, **runtime_config}

@dataclass(kw_only=True)
class Configuration:

    mcp_server_config: dict[str, Any] = field(
        default_factory=lambda: _get_merged_mcp_config(),
        metadata={"description": "Dictionary mapping MCP server name to its configuration."},
    )

    planner_system_prompt: str = field(
        default=prompts.PLANNER_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for forming the plan to address the current state of conversation with experts."},
    )

    planner_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for planning. Should be in the form: provider/model-name."
        },
    )

    execute_task_system_prompt: str = field(
        default=prompts.EXECUTE_TASK_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for executing a task by an expert."},
    )

    execute_task_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for executing a task by an expert. Should be in the form: provider/model-name."
        },
    )

    generate_response_system_prompt: str = field(
        default=prompts.GENERATE_RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating final responses after plan completion."},
    )

    generate_response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The language model used for generating final responses. Should be in the form: provider/model-name."
        },
    )
    organization_id: str = field(
        default="",
        metadata={"description": "The organization ID to use for the conversation."},
    )

    def __post_init__(self):
        """Ensure mcp_server_config always has merged default + runtime values."""
        self.mcp_server_config = _get_merged_mcp_config(self.mcp_server_config)

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an Configuration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of Configuration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
    
    def get_mcp_server_descriptions(self) -> list[tuple[str, str]]:
        """Get a list of descriptions of all MCP servers in the specified configuration."""
        descriptions = []
        for server_name, server_config in self.mcp_server_config.items():
            description = server_config.get('description', '')
            descriptions.append((server_name, description))
        return descriptions
    
    def build_experts_context(self) -> str:
        """Build the experts part of the prompt for the planning task.
        
        Here's the format to use:
        - <server_name>: <server_description>
        - <server_name>: <server_description>
        ...

        Returns:
            str: The experts part of the prompt.
        """
        return "\n".join([f"- {server_name}: {server_description}" for server_name, server_description in self.get_mcp_server_descriptions()])
    
    def get_server_config(self, server_name: str) -> Dict[str, Any] | None:
        """Get server configuration for the specified server.

        Args:
            server_name (str): Name of the server to get configuration for

        Returns:
            Dict[str, Any]: Server configuration for the specified server or None if not found
        """
        return self.mcp_server_config.get(server_name, None)
        

T = TypeVar("T", bound=Configuration)
