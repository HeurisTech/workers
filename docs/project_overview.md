# Project Overview

## Purpose

This project demonstrates building a universal assistant using LangGraph and the Model Context Protocol (MCP). It combines LangGraph's workflow orchestration capabilities with MCP's standardized interface for connecting AI models to external tools and data sources.

The core idea is to implement a multi-agent pattern where an assistant routes user requests to appropriate agents. These agents then interact with MCP servers to utilize their offered tools, prompts, and resources.

## Key Components

*   **LangGraph:** Used to define and execute the assistant's workflow as a graph. Nodes represent actions (like routing, calling agents, or interacting with MCP), and edges define the control flow.
*   **MCP:** Provides a standardized way for LangGraph agents to communicate with external services (MCP Servers) offering tools and data.
*   **Strategies:** Different implementations of the LangGraph assistant workflow are provided as distinct "strategies" (e.g., using a planner, a retriever, etc.).
*   **MCP Wrapper (`src/langgraph_mcp/mcp_wrapper.py`):** A generic module to handle communication with MCP servers (both standard and Smithery-based) using a strategy pattern.

## Directory Structure

-   `src/langgraph_mcp/`: Contains the core source code.
    -   `with_planner/`: Old planner agent implementation strategy.
    -   `planner_style/`: Enhanced planner style agent with goal tracking, custom tools, and async input
    -   `tools.py`: All tools functionality (framework, registry, implementations, merging)
    -   `state.py`: Defines the common state shared across graphs.
    -   `utils.py`: Common utility functions and AsyncInputManager.
    -   `mcp_wrapper.py`: Handles interaction with MCP servers.
-   `pyproject.toml`: Project metadata and dependencies.
-   `langgraph.json`: LangGraph-specific configuration, including graph entry points.
-   `README.md`: General project description for human readers.
-   `llms.txt`: Entry point for AI/LLM-readable documentation.
-   `docs/`: Directory containing detailed markdown documentation linked from `llms.txt`. 