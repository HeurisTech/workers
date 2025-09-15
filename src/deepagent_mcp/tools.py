"""MCP tool integration for Phases 1, 2, and 3.

Uses langchain-mcp-adapters to connect to MCP servers and load tools.
Includes intelligent tool filtering for Phase 3.
"""

import json
from typing import Dict, List, Any, Optional
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagent_mcp.state import MCPOrchestratorState, MCPToolInfo, MCPServerStatus
from deepagent_mcp.utils import setup_logging, analyze_with_llm


class MCPToolManager:
    """Manages MCP server connections and tool loading."""

    def __init__(self, server_configs: Dict[str, Any]):
        """Initialize with MCP server configurations.

        Args:
            server_configs: Dictionary mapping server names to connection configs
        """
        self.server_configs = server_configs
        self.client = MultiServerMCPClient(server_configs) if server_configs else None
        self._tools_cache: List[BaseTool] = []
        self._tool_info_cache: List[MCPToolInfo] = []

    async def discover_tools(self, state: MCPOrchestratorState) -> Dict[str, Any]:
        """Discover and catalog all available MCP tools.

        Args:
            state: Current orchestrator state

        Returns:
            State updates with discovered tools and server status
        """
        if not self.client:
            return {
                "last_error": "No MCP servers configured",
                "mcp_servers": {},
                "available_tools": []
            }

        try:
            # Load all tools from all servers
            tools = await self.client.get_tools()

            # Validate and filter tools with invalid schemas that would be rejected by providers (e.g., OpenAI)
            def _get_schema_dict(t: BaseTool) -> Dict[str, Any]:
                if not getattr(t, 'args_schema', None):
                    return {}
                schema = t.args_schema
                if hasattr(schema, 'model_json_schema'):
                    try:
                        return schema.model_json_schema()
                    except Exception:
                        return {}
                return schema if isinstance(schema, dict) else {}

            def _schema_is_valid_for_llm(schema: Dict[str, Any]) -> bool:
                # Recursively ensure array types declare 'items'; this is required by OpenAI tools API.
                if not isinstance(schema, dict):
                    return True
                if schema.get("type") == "array" and "items" not in schema:
                    return False
                # Recurse into nested structures
                for key in ("properties", "definitions"):
                    props = schema.get(key)
                    if isinstance(props, dict):
                        for v in props.values():
                            if not _schema_is_valid_for_llm(v):
                                return False
                # Recurse into common fields
                for key in ("items", "allOf", "anyOf", "oneOf"):
                    val = schema.get(key)
                    if isinstance(val, dict):
                        if not _schema_is_valid_for_llm(val):
                            return False
                    elif isinstance(val, list):
                        for v in val:
                            if isinstance(v, dict) and not _schema_is_valid_for_llm(v):
                                return False
                return True

            valid_tools: List[BaseTool] = []
            for t in tools:
                schema_dict = _get_schema_dict(t)
                if _schema_is_valid_for_llm(schema_dict):
                    valid_tools.append(t)
                else:
                    # Drop invalid tool to prevent provider 400s (e.g., arrays without 'items')
                    continue

            self._tools_cache = valid_tools

            # Convert to our tool info format
            tool_infos = []
            server_status = {}

            # Group tools by server (this is a simplified approach)
            for tool in self._tools_cache:
                # Extract server name from tool metadata if available
                server_name = getattr(tool, 'server_name', 'unknown')

                # Handle schema - it might be a dict or a Pydantic model
                schema = {}
                if tool.args_schema:
                    if hasattr(tool.args_schema, 'model_json_schema'):
                        schema = tool.args_schema.model_json_schema()
                    else:
                        schema = tool.args_schema  # It's already a dict

                tool_info = MCPToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    server_name=server_name,
                    schema=schema
                )
                tool_infos.append(tool_info)

                # Update server status
                if server_name not in server_status:
                    server_status[server_name] = MCPServerStatus(
                        name=server_name,
                        connected=True,
                        tools_count=0
                    )
                server_status[server_name].tools_count += 1

            self._tool_info_cache = tool_infos

            return {
                "available_tools": tool_infos,
                "mcp_servers": server_status,
                "last_error": None
            }

        except Exception as e:
            return {
                "last_error": f"Failed to discover tools: {str(e)}",
                "mcp_servers": {},
                "available_tools": [],
                "error_count": getattr(state, 'error_count', 0) + 1
            }

    async def get_tools_for_execution(self, 
                                    state: MCPOrchestratorState,
                                    max_tools: Optional[int] = None) -> List[BaseTool]:
        """Get tools for execution, optionally filtered.

        Args:
            state: Current orchestrator state
            max_tools: Maximum number of tools to return

        Returns:
            List of LangChain tools ready for execution
        """
        if not self._tools_cache:
            # Try to load tools if not cached, then filter invalid ones
            try:
                if self.client:
                    raw_tools = await self.client.get_tools()

                    def _get_schema_dict(t: BaseTool) -> Dict[str, Any]:
                        if not getattr(t, 'args_schema', None):
                            return {}
                        schema = t.args_schema
                        if hasattr(schema, 'model_json_schema'):
                            try:
                                return schema.model_json_schema()
                            except Exception:
                                return {}
                        return schema if isinstance(schema, dict) else {}

                    def _schema_is_valid_for_llm(schema: Dict[str, Any]) -> bool:
                        if not isinstance(schema, dict):
                            return True
                        if schema.get("type") == "array" and "items" not in schema:
                            return False
                        for key in ("properties", "definitions"):
                            props = schema.get(key)
                            if isinstance(props, dict):
                                for v in props.values():
                                    if not _schema_is_valid_for_llm(v):
                                        return False
                        for key in ("items", "allOf", "anyOf", "oneOf"):
                            val = schema.get(key)
                            if isinstance(val, dict):
                                if not _schema_is_valid_for_llm(val):
                                    return False
                            elif isinstance(val, list):
                                for v in val:
                                    if isinstance(v, dict) and not _schema_is_valid_for_llm(v):
                                        return False
                        return True

                    filtered: List[BaseTool] = []
                    for t in raw_tools:
                        if _schema_is_valid_for_llm(_get_schema_dict(t)):
                            filtered.append(t)
                    self._tools_cache = filtered
            except Exception:
                return []

        tools = self._tools_cache.copy()

        # Apply max_tools limit if specified
        if max_tools and len(tools) > max_tools:
            # For Phase 1, just take the first N tools
            # In Phase 3, we'll implement intelligent filtering
            tools = tools[:max_tools]

        return tools

    async def get_tools_for_filtered_execution(self, filtered_tool_infos: List[MCPToolInfo]) -> List[BaseTool]:
        """Get tools for execution based on filtered tool info list.

        Args:
            filtered_tool_infos: List of tool info objects to include

        Returns:
            List of corresponding BaseTool objects
        """
        if not self._tools_cache:
            return []

        filtered_names = {tool_info.name for tool_info in filtered_tool_infos}
        filtered_tools = [tool for tool in self._tools_cache if tool.name in filtered_names]

        return filtered_tools

    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all MCP servers.

        Returns:
            Dictionary mapping server names to health status
        """
        if not self.client:
            return {}

        # Simple health check - try to get tools
        try:
            await self.client.get_tools()
            # If successful, all servers in config are healthy
            return {name: True for name in self.server_configs.keys()}
        except Exception:
            # If failed, mark all as unhealthy (simplified for Phase 1)
            return {name: False for name in self.server_configs.keys()}


class IntelligentToolFilter:
    """Phase 3: Intelligent tool filtering with planning and relevance analysis."""

    def __init__(self, max_tools_per_step: int = 10):
        self.max_tools_per_step = max_tools_per_step

    async def create_plan_and_filter_tools(self, 
                                         user_request: str,
                                         available_tools: List[MCPToolInfo],
                                         model: str) -> Dict[str, Any]:
        """Main filtering pipeline with planning.

        Args:
            user_request: User's request to fulfill
            available_tools: All available tools
            model: Model to use for analysis

        Returns:
            Dictionary with execution plan and filtered tools
        """
        if len(available_tools) <= self.max_tools_per_step:
            # No filtering needed
            return {
                "execution_steps": ["Execute user request with available tools"],
                "filtered_tools": available_tools,
                "planned_operations": ["general"]
            }

        try:
            # Stage 1: Score relevance, filter out irrelevant tools
            relevance_scores = await self._analyze_tool_relevance(user_request, available_tools, model)
            relevant_tools = [
                tool for tool in available_tools 
                if relevance_scores.get(tool.name, 0) >= 0.1
            ]

            # Stage 2: Create execution plan
            execution_plan = await self._create_execution_plan(user_request, relevant_tools, model)

            # Stage 3: Select top tools based on plan and relevance
            final_tools = await self._select_final_tools(
                execution_plan, relevant_tools, relevance_scores
            )

            return {
                "execution_steps": execution_plan.get("steps", []),
                "filtered_tools": final_tools,
                "planned_operations": execution_plan.get("operations", []),
                "tool_count_reduced": len(available_tools) - len(final_tools)
            }

        except Exception as e:
            # Fallback to simple filtering
            return {
                "execution_steps": ["Execute user request (filtering failed)"],
                "filtered_tools": available_tools[:self.max_tools_per_step],
                "planned_operations": ["general"],
                "filtering_error": str(e)
            }

    async def _analyze_tool_relevance(self, 
                                    user_request: str,
                                    available_tools: List[MCPToolInfo],
                                    model: str) -> Dict[str, float]:
        """Stage 1: Analyze tool relevance using LLM."""
        tools_info = []
        for tool in available_tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "server": tool.server_name
            })

        analysis_prompt = f"""User Request: {user_request}

Available Tools: {json.dumps(tools_info, indent=2)}

Rate each tool's relevance (0.0-1.0) for this specific request.
Consider:
- Does it directly help accomplish the user's goal?
- Is it the best tool for this task?
- Would the request be harder without it?

Return JSON mapping tool names to relevance scores (0.0-1.0).
Example: {{"tool1": 0.8, "tool2": 0.2, "tool3": 0.9}}"""

        try:
            response = await analyze_with_llm(analysis_prompt, model)
            # Parse JSON response with robust parsing
            if isinstance(response, str):
                from .utils import parse_json_response
                parsed_response = parse_json_response(response)
                relevance_scores = parsed_response if isinstance(parsed_response, dict) else {}
            else:
                relevance_scores = response

            # Ensure all tools have scores
            for tool in available_tools:
                if tool.name not in relevance_scores:
                    relevance_scores[tool.name] = 0.0

            return relevance_scores

        except Exception as e:
            # Fallback: give all tools equal moderate relevance
            return {tool.name: 0.5 for tool in available_tools}

    async def _create_execution_plan(self,
                                   user_request: str, 
                                   relevant_tools: List[MCPToolInfo],
                                   model: str) -> Dict[str, Any]:
        """Stage 2: Create execution plan."""
        tools_list = [{"name": tool.name, "description": tool.description} for tool in relevant_tools]

        planning_prompt = f"""User Request: {user_request}

Available Relevant Tools: {json.dumps(tools_list, indent=2)}

Create an execution plan by:
1. Breaking the request into logical steps
2. Identifying which tools are needed for each step
3. Determining operation types (read, write, search, analyze, etc.)
4. Flagging steps that might need human approval

Return JSON with:
- steps: Array of step descriptions
- tool_assignments: Object mapping step numbers to tool names
- operations: Array of operation types
- requires_approval: Boolean for sensitive operations

Example:
{{
  "steps": ["Step 1: Search for information", "Step 2: Analyze results"],
  "tool_assignments": {{"1": ["search_tool"], "2": ["analyze_tool"]}},
  "operations": ["search", "analyze"],
  "requires_approval": false
}}"""

        try:
            response = await analyze_with_llm(planning_prompt, model)
            if isinstance(response, str):
                from .utils import parse_json_response
                parsed_response = parse_json_response(response)
                plan = parsed_response if isinstance(parsed_response, dict) else {"steps": ["Execute user request"], "requires_approval": False}
            else:
                plan = response

            return plan

        except Exception as e:
            # Fallback plan
            return {
                "steps": ["Execute user request with relevant tools"],
                "tool_assignments": {"1": [tool.name for tool in relevant_tools[:5]]},
                "operations": ["general"],
                "requires_approval": False
            }

    async def _select_final_tools(self,
                                execution_plan: Dict[str, Any],
                                relevant_tools: List[MCPToolInfo], 
                                relevance_scores: Dict[str, float]) -> List[MCPToolInfo]:
        """Stage 3: Select final tools based on plan and relevance."""

        # Get tools mentioned in plan
        plan_tools = set()
        tool_assignments = execution_plan.get("tool_assignments", {})
        for step_tools in tool_assignments.values():
            if isinstance(step_tools, list):
                plan_tools.update(step_tools)
            else:
                plan_tools.add(step_tools)

        # Prioritize tools mentioned in plan
        final_tools = []
        tool_names_added = set()

        # First, add tools from the plan
        for tool in relevant_tools:
            if tool.name in plan_tools and len(final_tools) < self.max_tools_per_step:
                final_tools.append(tool)
                tool_names_added.add(tool.name)

        # Then add highest scoring tools not in plan
        remaining_tools = [
            tool for tool in relevant_tools 
            if tool.name not in tool_names_added
        ]
        remaining_tools.sort(key=lambda t: relevance_scores.get(t.name, 0), reverse=True)

        for tool in remaining_tools:
            if len(final_tools) < self.max_tools_per_step:
                final_tools.append(tool)
            else:
                break

        return final_tools


# Export the classes
__all__ = ["MCPToolManager", "IntelligentToolFilter"]
