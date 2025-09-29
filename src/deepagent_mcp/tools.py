"""MCP tool integration for Phases 1, 2, and 3.

Uses langchain-mcp-adapters to connect to MCP servers and load tools.
Includes intelligent tool filtering for Phase 3.
"""

import json
from typing import Dict, List, Any, Optional
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagent_mcp.state import MCPOrchestratorState, MCPToolInfo, MCPServerStatus


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

    async def _ensure_tools_loaded(self) -> None:
        """Ensure tools are loaded in cache. Used for LangGraph Platform compatibility."""
        if self._tools_cache or not self.client:
            return
            
        try:
            raw_tools = await self.client.get_tools()
            
            # Apply schema validation
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

            # Filter valid tools
            valid_tools = []
            invalid_count = 0
            for t in raw_tools:
                try:
                    schema_dict = _get_schema_dict(t)
                    if _schema_is_valid_for_llm(schema_dict):
                        valid_tools.append(t)
                    else:
                        invalid_count += 1
                except Exception:
                    invalid_count += 1
            
            self._tools_cache = valid_tools
            
            # Log tool loading results for debugging
            from deepagent_mcp.utils import setup_logging
            logger = setup_logging(True)  # Enable debug for tool loading issues
            logger.info(f"Tool cache loaded: {len(valid_tools)} valid tools, {invalid_count} invalid/failed")
            
        except Exception as e:
            # Log the error for debugging but don't raise
            from deepagent_mcp.utils import setup_logging
            logger = setup_logging(True)
            logger.error(f"Failed to ensure tools loaded: {e}")
            pass

    async def discover_tools(self, state: MCPOrchestratorState) -> Dict[str, Any]:
        """Discover and catalog all available MCP tools.

        Args:
            state: Current orchestrator state

        Returns:
            State updates with discovered tools and server status
        """
        if not self.client:
            return {
                "last_error": None,  # No error - just no MCP servers configured
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
        await self._ensure_tools_loaded()
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
        await self._ensure_tools_loaded()
        
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
        
        # Define tool categories for better filtering
        self.tool_categories = {
            "email": ["gmail", "mail", "email", "message", "send", "inbox", "draft"],
            "calendar": ["calendar", "event", "schedule", "meeting", "appointment"],
            "files": ["drive", "file", "document", "folder", "upload", "download"],
            "communication": ["slack", "teams", "chat", "notify", "sms", "call"],
            "database": ["db", "sql", "query", "table", "record", "data"],
            "web": ["http", "api", "request", "fetch", "scrape", "url"],
            "analytics": ["analyze", "report", "chart", "metric", "dashboard"],
            "automation": ["workflow", "trigger", "action", "webhook", "integration"],
            "ai": ["gpt", "ai", "llm", "generate", "analyze", "summarize"],
            "social": ["twitter", "linkedin", "facebook", "social", "post", "share"]
        }
        
        # Define operation types
        self.operation_types = {
            "read": ["get", "fetch", "list", "search", "find", "retrieve", "read"],
            "write": ["create", "send", "post", "add", "insert", "write", "upload"],
            "update": ["update", "edit", "modify", "change", "patch", "set"],
            "delete": ["delete", "remove", "trash", "clear", "clean"],
            "analyze": ["analyze", "process", "calculate", "compute", "evaluate"]
        }

    async def create_plan_and_filter_tools(self, 
                                         user_request: str,
                                         available_tools: List[MCPToolInfo],
                                         model: str) -> Dict[str, Any]:
        """Main filtering pipeline with multi-stage intelligent filtering.

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
                "planned_operations": ["general"],
                "filtering_strategy": "none_needed"
            }

        try:
            # Stage 1: Fast keyword-based pre-filtering (reduces load for LLM)
            pre_filtered_tools = await self._pre_filter_by_keywords(user_request, available_tools)
            
            # Stage 2: Semantic similarity filtering
            semantically_filtered = await self._filter_by_semantic_similarity(
                user_request, pre_filtered_tools, model
            )
            
            # Stage 3: LLM-based relevance scoring (only for remaining tools)
            relevance_scores = await self._analyze_tool_relevance(
                user_request, semantically_filtered, model
            )
            
            # Stage 4: Filter by relevance threshold (adaptive based on tool count)
            threshold = self._calculate_relevance_threshold(len(semantically_filtered))
            relevant_tools = [
                tool for tool in semantically_filtered 
                if relevance_scores.get(tool.name, 0) >= threshold
            ]

            # Stage 5: Create execution plan with filtered tools
            execution_plan = await self._create_execution_plan(user_request, relevant_tools, model)

            # Stage 6: Final tool selection based on plan and scores
            final_tools = await self._select_final_tools(
                execution_plan, relevant_tools, relevance_scores
            )

            return {
                "execution_steps": execution_plan.get("steps", []),
                "filtered_tools": final_tools,
                "planned_operations": execution_plan.get("operations", []),
                "tool_count_reduced": len(available_tools) - len(final_tools),
                "filtering_strategy": "multi_stage",
                "filtering_stats": {
                    "original_count": len(available_tools),
                    "after_keywords": len(pre_filtered_tools),
                    "after_semantic": len(semantically_filtered), 
                    "after_relevance": len(relevant_tools),
                    "final_count": len(final_tools),
                    "relevance_threshold": threshold
                }
            }

        except Exception as e:
            # Multi-level fallback strategy
            fallback_tools = await self._fallback_filter(user_request, available_tools)
            return {
                "execution_steps": ["Execute user request (using fallback filtering)"],
                "filtered_tools": fallback_tools,
                "planned_operations": ["general"],
                "filtering_error": str(e),
                "filtering_strategy": "fallback"
            }

    async def _pre_filter_by_keywords(self,
                                    user_request: str,
                                    available_tools: List[MCPToolInfo]) -> List[MCPToolInfo]:
        """Stage 1: Fast keyword-based pre-filtering to reduce tool set."""
        request_lower = user_request.lower()
        
        # Extract key intent categories from request
        detected_categories = []
        for category, keywords in self.tool_categories.items():
            if any(keyword in request_lower for keyword in keywords):
                detected_categories.append(category)
        
        # Extract operation types 
        detected_operations = []
        for operation, keywords in self.operation_types.items():
            if any(keyword in request_lower for keyword in keywords):
                detected_operations.append(operation)
        
        # If no categories detected, return top tools by name similarity
        if not detected_categories:
            return self._filter_by_name_similarity(request_lower, available_tools)
        
        # Filter tools by detected categories
        filtered_tools = []
        for tool in available_tools:
            tool_name_desc = f"{tool.name.lower()} {tool.description.lower()}"
            
            # Check if tool matches detected categories
            category_match = False
            for category in detected_categories:
                if any(keyword in tool_name_desc for keyword in self.tool_categories[category]):
                    category_match = True
                    break
            
            # Check if tool matches detected operations
            operation_match = False
            if detected_operations:
                for operation in detected_operations:
                    if any(keyword in tool_name_desc for keyword in self.operation_types[operation]):
                        operation_match = True
                        break
            else:
                operation_match = True  # No specific operation required
            
            if category_match and operation_match:
                filtered_tools.append(tool)
        
        # If still too many tools, take top by relevance heuristics
        if len(filtered_tools) > self.max_tools_per_step * 3:  # 3x buffer for next stage
            filtered_tools = self._rank_by_heuristics(request_lower, filtered_tools)
            filtered_tools = filtered_tools[:self.max_tools_per_step * 3]
        
        return filtered_tools if filtered_tools else available_tools[:self.max_tools_per_step * 2]

    def _filter_by_name_similarity(self, request_lower: str, tools: List[MCPToolInfo]) -> List[MCPToolInfo]:
        """Fallback: filter by name/description similarity when no categories detected."""
        request_words = set(request_lower.split())
        
        scored_tools = []
        for tool in tools:
            tool_words = set(f"{tool.name.lower()} {tool.description.lower()}".split())
            # Simple word overlap scoring
            overlap = len(request_words.intersection(tool_words))
            if overlap > 0:
                scored_tools.append((tool, overlap))
        
        # Sort by overlap score and return top candidates
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in scored_tools[:self.max_tools_per_step * 2]]

    def _rank_by_heuristics(self, request_lower: str, tools: List[MCPToolInfo]) -> List[MCPToolInfo]:
        """Rank tools using simple heuristics for better filtering."""
        scored_tools = []
        
        for tool in tools:
            score = 0
            tool_name_desc = f"{tool.name.lower()} {tool.description.lower()}"
            
            # Exact name matches get highest score
            if any(word in tool.name.lower() for word in request_lower.split()):
                score += 10
            
            # Description matches get medium score
            if any(word in tool.description.lower() for word in request_lower.split()):
                score += 5
                
            # Prefer shorter, more specific tool names
            score += max(0, 20 - len(tool.name))
            
            scored_tools.append((tool, score))
        
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in scored_tools]

    async def _filter_by_semantic_similarity(self,
                                           user_request: str,
                                           tools: List[MCPToolInfo],
                                           model: str) -> List[MCPToolInfo]:
        """Stage 2: Semantic similarity filtering using lightweight LLM call."""
        if len(tools) <= self.max_tools_per_step * 2:
            return tools
        
        # Create a simplified prompt for fast semantic filtering
        tools_summary = []
        for i, tool in enumerate(tools):
            tools_summary.append(f"{i}: {tool.name} - {tool.description[:100]}...")
        
        prompt = f"""User Request: {user_request}

Tools (numbered list):
{chr(10).join(tools_summary)}

Return ONLY the numbers (comma-separated) of the most relevant tools for this request.
Focus on tools that directly help accomplish the user's goal.
Maximum {self.max_tools_per_step * 2} tools.

Example response: 1,5,12,18"""

        try:
            from deepagent_mcp.utils import analyze_with_llm
            response = await analyze_with_llm(prompt, model)
            
            # Parse the response to get tool indices
            if isinstance(response, str):
                # Extract numbers from response
                import re
                numbers = re.findall(r'\d+', response.strip())
                selected_indices = [int(n) for n in numbers if int(n) < len(tools)]
                
                if selected_indices:
                    return [tools[i] for i in selected_indices[:self.max_tools_per_step * 2]]
        
        except Exception:
            pass
        
        # Fallback: return first portion of tools
        return tools[:self.max_tools_per_step * 2]

    def _calculate_relevance_threshold(self, tool_count: int) -> float:
        """Calculate adaptive relevance threshold based on tool count."""
        if tool_count <= self.max_tools_per_step:
            return 0.1  # Very permissive
        elif tool_count <= self.max_tools_per_step * 2:
            return 0.3  # Moderate
        elif tool_count <= self.max_tools_per_step * 4:
            return 0.5  # Stricter
        else:
            return 0.7  # Very strict

    async def _fallback_filter(self,
                             user_request: str,
                             available_tools: List[MCPToolInfo]) -> List[MCPToolInfo]:
        """Multi-level fallback when main filtering fails."""
        try:
            # Try keyword-based filtering
            filtered = await self._pre_filter_by_keywords(user_request, available_tools)
            if len(filtered) <= self.max_tools_per_step:
                return filtered
            
            # If still too many, use heuristic ranking
            request_lower = user_request.lower()
            ranked = self._rank_by_heuristics(request_lower, filtered)
            return ranked[:self.max_tools_per_step]
            
        except Exception:
            # Ultimate fallback: return first N tools
            return available_tools[:self.max_tools_per_step]

    async def _analyze_tool_relevance(self, 
                                    user_request: str,
                                    available_tools: List[MCPToolInfo],
                                    model: str) -> Dict[str, float]:
        """Stage 3: Analyze tool relevance using LLM with enhanced prompt."""
        # Limit tools to prevent context overflow
        tools_to_analyze = available_tools[:50]  # Reasonable limit for LLM analysis
        
        tools_info = []
        for tool in tools_to_analyze:
            tools_info.append({
                "name": tool.name,
                "description": tool.description[:200],  # Truncate long descriptions
                "server": tool.server_name
            })

        analysis_prompt = f"""User Request: "{user_request}"

Available Tools ({len(tools_info)} tools):
{json.dumps(tools_info, indent=2)}

Rate each tool's relevance (0.0-1.0) for this SPECIFIC request:

Scoring Guidelines:
- 0.9-1.0: Essential/Primary tools directly needed for the task
- 0.7-0.8: Important supporting tools
- 0.5-0.6: Potentially useful but not critical
- 0.3-0.4: Tangentially related
- 0.0-0.2: Not relevant

Consider:
1. Does the tool directly accomplish the user's primary goal?
2. Is this tool necessary for the specific action requested?
3. Would completing this request be impossible/much harder without this tool?
4. Does the tool name/description match the request intent?

Return ONLY valid JSON mapping tool names to scores (0.0-1.0):
{{"TOOL_NAME_1": 0.85, "TOOL_NAME_2": 0.12, "TOOL_NAME_3": 0.93}}"""

        try:
            from deepagent_mcp.utils import analyze_with_llm
            response = await analyze_with_llm(analysis_prompt, model)
            # Parse JSON response with robust parsing
            if isinstance(response, str):
                from deepagent_mcp.utils import parse_json_response
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
            from deepagent_mcp.utils import analyze_with_llm
            response = await analyze_with_llm(planning_prompt, model)
            if isinstance(response, str):
                from deepagent_mcp.utils import parse_json_response
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
