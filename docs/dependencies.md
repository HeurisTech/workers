# Project Dependencies and Metadata

This document outlines the key dependencies and metadata for the `langgraph-mcp` project, primarily sourced from `pyproject.toml`.

## Project Information

- **Name:** `langgraph-mcp`
- **Version:** `0.0.1`
- **Description:** LangGraph Solution Template for MCP
- **Authors:** Heuris Labs <hello@heuris.co>
- **License:** MIT
- **Readme:** `README.md`

## Requirements

- **Python Version:** >=3.12

## Core Dependencies

- `asyncio>=3.4.3`
- `langchain>=0.2.17`
- `langchain-core>=0.3.21`
- `langchain-openai>=0.2.11`
- `langgraph>=0.2.56`
- `mcp>=1.0.0` (Model Context Protocol library)
- `openai>=1.57.0`
- `python-dotenv>=1.0.1`
- `smithery` (For interacting with Smithery-based MCP servers)

## Development Dependencies (`dev`)

- `debugpy`
- `mypy>=1.11.1`
- `ruff>=0.6.1`

## Build System

- **Requires:** `setuptools>=73.0.0`, `wheel`
- **Build Backend:** `setuptools.build_meta`
- **Packages:** `langgraph_mcp` (sourced from `src/langgraph_mcp`)
