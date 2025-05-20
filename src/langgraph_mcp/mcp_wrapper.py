import os
from abc import ABC, abstractmethod
from typing import Any
from langchain_core.tools import ToolException
from mcp import ClientSession, ListPromptsResult, ListResourcesResult, ListToolsResult, StdioServerParameters, stdio_client
import mcp
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client
import pydantic_core
import smithery
from urllib.parse import urlparse


# Abstract base class for MCP session functions
class MCPSessionFunction(ABC):
    @abstractmethod
    async def __call__(self, server_name: str, env: dict, session: ClientSession) -> Any:
        pass

class RoutingDescription(MCPSessionFunction):
    async def __call__(self, server_name: str, env: dict, session: ClientSession) -> str:
        tools: ListToolsResult | None = None
        prompts: ListPromptsResult | None = None
        resources: ListResourcesResult | None = None
        content = ""
        try:
            tools = await session.list_tools()
            if tools:
                content += "Provides tools:\n"
                for tool in tools.tools:
                    content += f"- {tool.name}: {tool.description}\n"
                content += "---\n"
        except Exception as e:
            print(f"Failed to fetch tools from server '{server_name}': {e}")
        
        try:
            prompts = await session.list_prompts()
            if prompts:
                content += "Provides prompts:\n"
                for prompt in prompts.prompts:
                    content += f"- {prompt.name}: {prompt.description}\n"
                content += "---\n"
        except Exception as e:
            print(f"Failed to fetch prompts from server '{server_name}': {e}")

        try:
            resources = await session.list_resources()
            if resources:
                content += "Provides resources:\n"
                for resource in resources.resources:
                    content += f"- {resource.name}: {resource.description}\n"
                content += "---\n"
        except Exception as e:
            print(f"Failed to fetch resources from server '{server_name}': {e}")

        return server_name, content

class GetTools(MCPSessionFunction):
    async def __call__(self, server_name: str, env: dict, session: ClientSession) -> list[dict[str, Any]]:
        tools = await session.list_tools()
        if tools is None:
            return []
        return [
            {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description or "",
                    'parameters': tool.inputSchema or {}
                }
            }
            for tool in tools.tools
        ]

class GetPrompts(MCPSessionFunction):
    async def __call__(self, server_name: str, env: dict, session: ClientSession) -> dict[str, Any]:
        prompts = await session.list_prompts()
        if prompts is None:
            return {"prompts": []}
        return {
            "prompts": [
                {
                    "name": prompt.name,
                    "description": prompt.description or "",
                    "arguments": prompt.arguments or []
                }
                for prompt in prompts.prompts
            ]
        }

class RunTool(MCPSessionFunction):
    def __init__(self, tool_name: str, **kwargs):
        self.tool_name = tool_name
        self.kwargs = kwargs

    async def __call__(self, server_name: str, env: dict, session: ClientSession) -> Any:
        arguments = dict(self.kwargs)
        if env:
            arguments["env_override"] = env
        result = await session.call_tool(self.tool_name, arguments=arguments)
        content = pydantic_core.to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content

# Utility function to infer transport from config
def infer_transport_from_config(server_config: dict) -> str | None:
    if "command" in server_config:
        return "stdio"
    elif "url" in server_config:
        return "streamable_http"
    return None

async def apply(server_name: str, server_config: dict, fn: MCPSessionFunction) -> Any:
    """Apply a function to an MCP server session, handling stdio, streamable_http, and sse transports.
    
    Args:
        server_name: Name of the server to connect to
        server_config: Configuration for the server (should include 'transport', but can be inferred)
        fn: Function to apply to the server session
    """
    env = server_config.get("env") or {}
    transport = server_config.get("transport")
    if not transport:
        transport = infer_transport_from_config(server_config)
    if not transport:
        raise ValueError(f"No 'transport' specified and cannot be inferred from server_config for server '{server_name}'")

    if transport == "stdio":
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"],
            env={**os.environ, **env}
        )
        print(f"Starting stdio session with (server: {server_name})")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(server_name, {}, session)

    elif transport == "streamable_http":
        url = server_config["url"]
        parsed = urlparse(url)
        if parsed.hostname and parsed.hostname.endswith("smithery.ai"):
            url = smithery.create_smithery_url(url, env) + f"&api_key={os.getenv('SMITHERY_API_KEY')}"
            print(f"Starting Smithery (streamable_http) session with (server: {server_name})")
        else:
            print(f"Starting streamable_http session with (server: {server_name})")
        async with streamablehttp_client(url) as (read_stream, write_stream, _):
            async with mcp.ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await fn(server_name, env, session)

    elif transport == "sse":
        url = server_config["url"]
        print(f"Starting SSE session with (server: {server_name})")
        async with sse_client(url) as (read_stream, write_stream, _):
            async with mcp.ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await fn(server_name, env, session)

    else:
        raise ValueError(f"Unknown transport '{transport}' for server '{server_name}'")
