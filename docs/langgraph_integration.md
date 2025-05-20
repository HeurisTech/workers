# LangGraph Integration Details

This document details how LangGraph is configured and integrated into the project, based on the `langgraph.json` file.

## Configuration (`langgraph.json`)

```json
{
    "name": "LangGraph Solution Template for MCP",
    "version": "1.0.0",
    "python_version": "3.12",
    "dependencies": ["."],
    "graphs": {
      "assist_with_planner": "./src/langgraph_mcp/with_planner/graph.py:graph"
    },
    "env": ".env"
}
```

## Key Aspects

*   **Name & Version:** Defines the LangGraph project name and version (distinct from the Python package version in `pyproject.toml`).
*   **Python Version:** Specifies the target Python version (`3.12`).
*   **Dependencies:** Indicates project dependencies relevant to LangGraph CLI deployment (`.` means the current project).
*   **Graphs:** This is the core mapping that defines the available LangGraph graphs and their entry points.
    *   Each key (e.g., `assist_with_planner`) is an identifier for a graph.
    *   The value specifies the Python file and the graph object within that file (e.g., `./src/langgraph_mcp/with_planner/graph.py:graph` points to the `graph` object in `graph.py`).
    *   This allows the LangGraph CLI and API server to discover and serve these specific graphs.
*   **Environment:** Specifies the environment file (`.env`) to load for configuration variables. 