from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from contextlib import asynccontextmanager
from typing import cast, Any, Dict

@asynccontextmanager
async def make_graph(model: str, expert: str, expert_config: Dict[str, Any]):
    # Filter out non-MCP parameters like 'description' before passing to MCP client
    mcp_config = {k: v for k, v in expert_config.items() if k != 'description'}
    
    # Extract or set description (if not present, use server name)
    description = expert_config.get('description', expert)
    
    client = MultiServerMCPClient(
        cast(Dict[str, Any], {
            expert: mcp_config
        })
    )

    async with client.session(expert) as session:
        tools = await load_mcp_tools(session)
        agent = create_react_agent(model, tools)
        yield agent  # Execution pauses here until caller finishes its async with

    # This part executes **only when** the caller exits their async with.
    # At this point, the session will be closed automatically.