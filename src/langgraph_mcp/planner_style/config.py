from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Dict, Optional, Type, TypeVar
from langchain_core.runnables import RunnableConfig, ensure_config

from langgraph_mcp.planner_style import prompts

@dataclass(kw_only=True)
class Configuration:

    mcp_server_config: dict[str, Any] = field(
        default_factory=dict,
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
            if server_config is None:
                description = ''
            else:
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
        # First try exact match
        if server_name in self.mcp_server_config:
            return self.mcp_server_config[server_name]
        
        # For Composio MCP configs, try matching by mcp_server_name field
        for config_key, server_config in self.mcp_server_config.items():
            if server_config is not None and isinstance(server_config, dict):
                if server_config.get('mcp_server_name') == server_name:
                    return server_config
        
        return None
        

T = TypeVar("T", bound=Configuration)
