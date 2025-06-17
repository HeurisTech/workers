from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from contextlib import asynccontextmanager

@asynccontextmanager
async def make_graph():
    client = MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
                "transport": "stdio",
            }
        }
    )

    async with client.session("playwright") as session:
        tools = await load_mcp_tools(session)
        agent = create_react_agent("openai:gpt-4.1", tools)
        yield agent  # Execution pauses here until caller finishes its async with

    # This part executes **only when** the caller exits their async with.
    # At this point, the session will be closed automatically.