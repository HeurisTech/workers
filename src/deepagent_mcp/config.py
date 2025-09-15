"""Configuration management for the MCP Orchestrator."""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar
from langchain_core.runnables import RunnableConfig, ensure_config

T = TypeVar("T", bound="Configuration")

# Supported models for Phase 2
SUPPORTED_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "claude-4": "anthropic/claude-3-5-sonnet-20241022", 
    "gemini-2.5": "google/gemini-2.0-flash-exp",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    # Add more models as needed
}

@dataclass(kw_only=True)
class Configuration:
    """Main configuration for the MCP Orchestrator."""

    # Phase 1: Base model configuration
    model: str = field(
        default="gpt-4o",
        metadata={"description": "The language model to use (gpt-4o, claude-4, gemini-2.5)"}
    )

    # MCP server configurations
    mcp_server_config: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Dictionary mapping MCP server name to its configuration"}
    )

    # Phase 1: Tool filtering
    max_tools_per_step: int = field(
        default=15,
        metadata={"description": "Maximum tools to load per execution step"}
    )

    tool_filtering_enabled: bool = field(
        default=False,
        metadata={"description": "Whether to enable intelligent tool filtering"}
    )

    # Phase 3: Advanced filtering controls
    max_tools_before_filtering: int = field(
        default=50,
        metadata={"description": "Maximum tools before advanced filtering kicks in"}
    )

    relevance_threshold: float = field(
        default=0.3,
        metadata={"description": "Minimum relevance score for tool inclusion (0.0-1.0)"}
    )

    enable_semantic_filtering: bool = field(
        default=True,
        metadata={"description": "Enable semantic similarity filtering"}
    )

    enable_keyword_prefiltering: bool = field(
        default=True,
        metadata={"description": "Enable fast keyword-based pre-filtering"}
    )

    # Phase 1: Execution settings
    max_concurrent_agents: int = field(
        default=3,
        metadata={"description": "Maximum number of concurrent sub-agents"}
    )

    # Phase 2 NEW: User system prompt
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"description": "Custom system prompt to prepend to base instructions"}
    )

    # Phase 3 NEW: Tool filtering and human-in-loop  
    enable_advanced_filtering: bool = field(
        default=True,  # Enable by default now
        metadata={"description": "Enable Phase 3 advanced tool filtering with planning"}
    )

    interrupt_before_execution: bool = field(
        default=False, 
        metadata={"description": "Require human approval before tool execution"}
    )

    interrupt_for_operations: List[str] = field(
        default_factory=list,
        metadata={"description": "List of operations that require human approval"}
    )

    # Debug mode
    debug_mode: bool = field(
        default=False,
        metadata={"description": "Enable debug logging"}
    )

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Normalize model name to full provider format
        if self.model in SUPPORTED_MODELS:
            self.model = SUPPORTED_MODELS[self.model]
        elif "/" not in self.model:
            # If it's just a model name without provider, try to infer
            if "gpt" in self.model.lower():
                self.model = f"openai/{self.model}"
            elif "claude" in self.model.lower():
                self.model = f"anthropic/{self.model}"
            elif "gemini" in self.model.lower():
                self.model = f"google/{self.model}"

    @classmethod
    def from_runnable_config(cls: Type[T], config: Optional[RunnableConfig] = None) -> T:
        """Create configuration from RunnableConfig."""
        config = ensure_config(config)
        configurable = config.get("configurable", {}) if config else {}

        field_names = {f.name for f in fields(cls) if f.init}
        config_values = {k: v for k, v in configurable.items() if k in field_names}

        return cls(**config_values)

    def get_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific MCP server."""
        return self.mcp_server_config.get(server_name)

    def validate_model(self) -> bool:
        """Validate that the configured model is supported."""
        # Basic validation - in production you might want to test actual connectivity
        supported_providers = ["openai", "anthropic", "google"]
        if "/" in self.model:
            provider = self.model.split("/")[0]
            return provider in supported_providers
        return self.model in SUPPORTED_MODELS


# Export main configuration class
__all__ = ["Configuration", "SUPPORTED_MODELS"]
