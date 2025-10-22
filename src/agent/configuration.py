"""Configuration utilities for the Computer-Use Agent.

All static settings are centralised here. Values are primarily sourced from
environment variables to keep the public API surface minimal and avoid passing
secrets across function boundaries.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Type, TypeVar
from dataclasses import dataclass, field, fields

from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationInfo
from langchain_core.runnables import RunnableConfig, ensure_config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

__all__ = ["AgentConfig", "get_config"]

T = TypeVar("T", bound="AgentConfig")


@dataclass(kw_only=True)
class AgentConfig:
    """Static configuration for the Computer-Use Agent.

    Most fields are optional and will fall back to sane defaults when omitted.
    You can supply values either via constructor kwargs *or* via environment
    variables whose names are documented alongside each field.
    """

    # ---------------------------------------------------------------------
    # Credential & API keys
    # ---------------------------------------------------------------------
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        metadata={"description": "Secret key for the OpenAI API. Set via $OPENAI_API_KEY."},
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""),
        metadata={"description": "Secret key for the Anthropic API. Set via $ANTHROPIC_API_KEY."},
    )
    e2b_api_key: str = field(
        default_factory=lambda: os.getenv("E2B_API_KEY", ""),
        metadata={"description": "Secret key for the e2b API. Set via $E2B_API_KEY."},
    )

    # ---------------------------------------------------------------------
    # Model names
    # ---------------------------------------------------------------------
    model_planner: str = field(
        default=os.getenv("CUA_PLANNER_MODEL", "gpt-4o-mini"),
        metadata={"description": "Chat model used for the planner node."},
    )
    model_executor: str = field(
        default=os.getenv("CUA_EXECUTOR_MODEL", "computer-use-preview"),
        metadata={"description": "Computer-Use model used for the executor node."},
    )
    brain_provider: str = field(
        default=os.getenv("CUA_BRAIN_PROVIDER", "openai"),
        metadata={"description": "LLM provider for the brain node. Options: 'openai' or 'anthropic'."},
    )
    brain_model: str = field(
        default=os.getenv("CUA_BRAIN_MODEL", "gpt-5-nano"),
        metadata={"description": "Model name for the brain node."},
    )

    # ---------------------------------------------------------------------
    # Sandbox settings
    # ---------------------------------------------------------------------
    sandbox_template: str = field(
        default=os.getenv("CUA_SANDBOX_TEMPLATE", "browser-python"),
        metadata={"description": "Name of the e2b sandbox template to launch when a sandbox is required."},
    )
    sandbox_timeout: int = field(
        default_factory=lambda: int(os.getenv("CUA_SANDBOX_TIMEOUT", "900")),
        metadata={"description": "Maximum sandbox lifetime in seconds (default 900 = 15 min)."},
    )

    # If the caller already has a sandbox they want to re-use, they can
    # populate these two fields in the initial GraphInput instead of relying
    # on configuration.
    sandbox_id: Optional[str] = field(
        default=None,
        metadata={"description": "Pre-existing sandbox id (optional)."}
    )
    sandbox_url: Optional[str] = field(
        default=None,
        metadata={"description": "Pre-existing sandbox URL (optional)."}
    )

    # ---------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------
    iteration_limit: int = field(
        default_factory=lambda: int(os.getenv("CUA_ITERATION_LIMIT", "30")),
        metadata={"description": "Safety cap on max executor iterations in a single invocation."},
    )
    
    recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("CUA_RECURSION_LIMIT", "50")),
        metadata={"description": "LangGraph recursion limit for brain-executor feedback loops."},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an AgentConfig instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of AgentConfig with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


@lru_cache(maxsize=1)
def get_config() -> AgentConfig:
    """Return a **cached** instance of `AgentConfig`.

    This ensures expensive env parsing occurs only once per process.
    """
    return AgentConfig()
