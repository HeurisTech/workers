"""Simple single-input/output LangGraph agent.

This module provides a basic agent that takes a user message and system prompt,
then generates a response using an LLM.
"""

from simple_agent.graph import graph

__all__ = ["graph"]
