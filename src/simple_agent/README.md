# Workflow Enhancement Agent

An AI-powered workflow enhancement agent that transforms simple task descriptions into comprehensive, well-structured workflows that effectively utilize available integrations.

## Purpose

This agent is specifically designed to:
- Analyze task descriptions and identify objectives
- Evaluate available integrations (email, calendar, files, communication, database, etc.)
- Generate detailed, step-by-step workflows
- Specify which integration to use at each step
- Include error handling and success criteria
- Transform vague requests into executable workflows

## Architecture

The workflow enhancement agent follows the standard LangGraph pattern:

```
InputState (task + integrations) → generate_response → State (enhanced workflow) → END
```

## Files

- `__init__.py` - Package initialization, exports the graph
- `state.py` - State definitions (InputState, State)
- `config.py` - Configuration management (Configuration dataclass)
- `agent.py` - Main agent implementation with graph definition
- `graph.py` - Graph entry point for external use

## Configuration

The agent supports the following configuration options:

- `model` (str): The language model to use (default: "openai/gpt-4o")
- `instructions` (str): System prompt for the agent (default: "You are a helpful assistant.")

## Usage

### Python API

```python
import asyncio
from langchain_core.messages import HumanMessage
from simple_agent.graph import graph

async def run_agent():
    # Prepare input
    input_data = {
        "messages": [HumanMessage(content="What is the capital of France?")]
    }

    # Prepare configuration
    config = {
        "configurable": {
            "thread_id": "my-thread-1",
            "model": "openai/gpt-4o",
            "instructions": "You are a helpful geography assistant."
        }
    }

    # Invoke the agent
    result = await graph.ainvoke(input_data, config=config)

    # Get the response
    response = result["messages"][-1]
    print(response.content)

asyncio.run(run_agent())
```

### LangGraph Platform

Deploy the agent using LangGraph Platform:

```bash
# Add to langgraph.json
{
  "graphs": {
    "simple_agent": "./src/simple_agent/graph.py:graph"
  }
}
```

Then deploy:

```bash
langgraph deploy
```

## Features

- **Simple Architecture**: Single node for straightforward LLM interactions
- **Configurable Model**: Support for any LangChain-compatible LLM
- **Custom System Prompts**: Easily customize agent behavior
- **PostgreSQL Checkpointing**: Automatic conversation persistence
- **Type-Safe**: Full type hints throughout
- **Async Support**: Built with async/await for performance

## Requirements

- Python >= 3.12
- LangChain Core
- LangGraph
- PostgreSQL (for checkpointing)

## Environment Variables

- `DATABASE_URI`: PostgreSQL connection string (optional, for checkpointing)

## Example

See `test_simple_agent.py` in the workers directory for a complete example.
