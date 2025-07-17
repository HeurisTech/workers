# LangGraph Solution Template for MCP based agents

> This project provides multiple LangGraph framework based implementations (strategies) for a universal assistant powered by MCP (Model Context Protocol) servers. It aims to showcase how LangGraph and MCP can be combined to build modular and extensible AI agents capable of interacting with various tools and data sources.

This documentation provides details necessary for Large Language Models (LLMs) and AI Agents to understand, diagnose, and enhance this project workspace.

## Project Overview

- [Project Structure and Purpose](./docs/project_overview.md): High-level description of the project, its goals, and the directory structure.

## Dependencies

- [Key Dependencies](./docs/dependencies.md): Information about project dependencies, Python version, and build system details extracted from `pyproject.toml`.

## LangGraph Integration

- [LangGraph Setup](./docs/langgraph_integration.md): Details on how LangGraph is configured and integrated, based on `langgraph.json`.

## Strategies

- [Implementation Strategies](./docs/strategies/index.md): Overview of the different LangGraph-based assistant strategies implemented and links to their specific documentation.

## Code Patterns

- [Common Code Patterns](./docs/code_patterns.md): Documentation of recurring code patterns and conventions used throughout the project, including LLM node implementation, model loading, and MCP interaction.

## Setting it up

1. Create and activate a virtual environment

    ```bash
    git clone {{REPO_URL}}
    cd langgraph-mcp
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. Install Langgraph CLI

    ```bash
    pip install -U "langgraph-cli[inmem]"
    ```

    Note: "inmem" extra(s) are needed to run LangGraph API server in development mode (without requiring Docker installation)

3. Install the dependencies

    ```bash
    source .venv/bin/activate
    pip install -e .
    pip install -e ".[dev]"
    pip install -e ".[test]"
    ```

4. Configure environment variables

    ```bash
    cp env.example .env
    ```

5. Run tests

    ```bash
    source .venv/bin/activate
    pytest tests/test_assistant_with_planner.py
    pytest tests/test_goal_oriented_agent.py
    ```