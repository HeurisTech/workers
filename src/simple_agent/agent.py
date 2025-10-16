"""Main agent implementation for the simple agent."""

import os
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from simple_agent.config import Configuration
from simple_agent.state import State, InputState
from langgraph_mcp.utils import load_chat_model


async def generate_response(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Generate a response using the configured LLM.

    This node takes the current state messages and system instructions,
    invokes the LLM, and returns the response message.

    Args:
        state: Current state containing messages.
        config: Runnable configuration containing agent settings.

    Returns:
        Dictionary with the response message to add to state.
    """
    # Load configuration
    cfg = Configuration.from_runnable_config(config)

    # Create prompt template with system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg.instructions),
        ("placeholder", "{messages}")
    ])

    # Load the chat model
    model = load_chat_model(cfg.model)

    # Prepare the prompt with current messages
    formatted_prompt = await prompt.ainvoke({"messages": state["messages"]}, config)

    # Invoke the model to get response
    response = await model.ainvoke(formatted_prompt.messages, config)

    # Return the response message
    return {"messages": [response]}


# Create the state graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the single node
builder.add_node("generate_response", generate_response)

# Add edges: START -> generate_response -> END
builder.add_edge(START, "generate_response")
builder.add_edge("generate_response", END)

# Set up checkpointing with PostgreSQL
DB_URI = os.getenv("DATABASE_URI")
checkpointer = None
if DB_URI:
    checkpointer = AsyncPostgresSaver.from_conn_string(DB_URI)

# Compile the graph
graph = builder.compile(checkpointer=checkpointer)

__all__ = ["graph", "generate_response"]
