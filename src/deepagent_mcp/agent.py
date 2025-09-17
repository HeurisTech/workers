"""Main MCP Orchestrator Agent - Phases 1, 2, and 3.

Implements the core orchestration logic using deepagents architecture
with MCP tool integration via langchain-mcp-adapters.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Union

# Load environment variables
load_dotenv()
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from deepagents import async_create_deep_agent
from deepagents.tools import write_todos, write_file, read_file, edit_file, ls

from deepagent_mcp.config import Configuration
from deepagent_mcp.state import MCPOrchestratorState, MCPToolInfo
from deepagent_mcp.prompts import get_system_prompt, get_classification_prompt, get_planning_prompt
from deepagent_mcp.tools import MCPToolManager, IntelligentToolFilter
from deepagent_mcp.utils import setup_logging, classify_request_with_llm, create_execution_plan


async def classify_request(state: MCPOrchestratorState, *, config: RunnableConfig) -> Dict[str, Any]:
    """Phase 3: Fast classification to route requests efficiently."""
    cfg = Configuration.from_runnable_config(config)
    logger = setup_logging(cfg.debug_mode)

    if not cfg.enable_advanced_filtering:
        # Skip classification in basic mode, go straight to discovery
        return {"request_classification": "execution"}

    # Handle both dict and object state
    messages = state.get('messages', []) if isinstance(state, dict) else getattr(state, 'messages', [])
    user_message = messages[-1].content if messages else ""

    # Use lightweight classification
    classification = await classify_request_with_llm(user_message, cfg.model)
    logger.info(f"Request classified as: {classification}")

    return {"request_classification": classification}


async def simple_response(state: MCPOrchestratorState, *, config: RunnableConfig) -> Dict[str, Any]:
    """Phase 3: Handle simple questions without tool execution - FAST path.""" 
    cfg = Configuration.from_runnable_config(config)
    
    # Handle both dict and object state
    if isinstance(state, dict):
        messages = state.get('messages', [])
        available_tools = state.get('available_tools', [])
    else:
        messages = state.messages
        available_tools = state.available_tools
    
    user_message = messages[-1].content if messages else ""

    # For simple responses, we can use a lightweight approach
    # In a full implementation, you'd use a simpler model here
    response_text = f"Based on available tools: {[tool.name for tool in available_tools]}, I can help you with various tasks. What specific assistance do you need?"

    return {
        "messages": [AIMessage(content=response_text)],
        "todos": []
    }


async def ask_clarification(state: MCPOrchestratorState, *, config: RunnableConfig) -> Dict[str, Any]:
    """Phase 3: Request clarification for ambiguous requests."""
    # Handle both dict and object state
    messages = state.get('messages', []) if isinstance(state, dict) else getattr(state, 'messages', [])
    user_message = messages[-1].content if messages else ""

    clarification_text = f"I need more details about your request: '{user_message}'. Could you please clarify what specific task you'd like me to help with?"

    return {
        "messages": [AIMessage(content=clarification_text)], 
        "awaiting_clarification": True
    }


async def discover_mcp_tools(state: MCPOrchestratorState, *, config: RunnableConfig) -> Dict[str, Any]:
    """Discovery node: Connect to MCP servers and catalog available tools."""
    cfg = Configuration.from_runnable_config(config)
    logger = setup_logging(cfg.debug_mode)

    logger.info("Starting MCP tool discovery...")

    # Initialize MCP tool manager
    tool_manager = MCPToolManager(cfg.mcp_server_config)

    # Discover tools from all configured servers
    discovery_result = await tool_manager.discover_tools(state)

    logger.info(f"Discovery complete. Found {len(discovery_result.get('available_tools', []))} tools")

    # Return the discovery result - LangGraph will merge the dict into state
    return discovery_result


async def plan_and_filter_tools(state: MCPOrchestratorState, *, config: RunnableConfig) -> Dict[str, Any]:
    """Phase 3: Create execution plan and filter relevant tools."""
    cfg = Configuration.from_runnable_config(config)
    logger = setup_logging(cfg.debug_mode)

    # Handle both dict and object state
    available_tools = state.get('available_tools', []) if isinstance(state, dict) else getattr(state, 'available_tools', [])
    messages = state.get('messages', []) if isinstance(state, dict) else getattr(state, 'messages', [])
    
    # Check if filtering is needed
    should_filter = (
        cfg.enable_advanced_filtering and 
        available_tools and 
        len(available_tools) > cfg.max_tools_before_filtering
    )
    
    if not should_filter:
        # Use basic filtering or no filtering
        basic_filtered = available_tools[:cfg.max_tools_per_step] if len(available_tools) > cfg.max_tools_per_step else available_tools
        return {
            "filtered_tools": basic_filtered, 
            "execution_plan": ["Execute user request with available tools"],
            "filtering_strategy": "basic" if basic_filtered != available_tools else "none"
        }

    user_message = messages[-1].content if messages else ""
    logger.info(f"Advanced filtering needed: {len(available_tools)} tools -> target: {cfg.max_tools_per_step}")

    # Initialize intelligent tool filter with configuration
    tool_filter = IntelligentToolFilter(cfg.max_tools_per_step)

    # Create execution plan and filter tools
    try:
        plan_result = await tool_filter.create_plan_and_filter_tools(
            user_message, available_tools, cfg.model
        )

        logger.info(f"Created plan with {len(plan_result['execution_steps'])} steps")
        logger.info(f"Filtered to {len(plan_result['filtered_tools'])} relevant tools")

        # Check if human approval is needed
        execution_steps = plan_result['execution_steps']
        needs_approval = cfg.interrupt_before_execution
        
        # Check if any step requires approval (handle both string and dict formats)
        if not needs_approval and execution_steps:
            for step in execution_steps:
                if isinstance(step, dict) and step.get('requires_approval', False):
                    needs_approval = True
                    break

        return {
            "filtered_tools": plan_result["filtered_tools"],
            "execution_plan": plan_result["execution_steps"], 
            "planned_operations": plan_result.get("planned_operations", []),
            "approval_required": needs_approval
        }

    except Exception as e:
        logger.error(f"Planning and filtering failed: {str(e)}")
        # Fallback to simple filtering
        max_tools = min(cfg.max_tools_per_step, len(available_tools))
        return {
            "filtered_tools": available_tools[:max_tools],
            "execution_plan": ["Execute user request with available tools"],
            "last_error": f"Advanced planning failed: {str(e)}"
        }


async def execute_with_mcp_tools(state: MCPOrchestratorState, *, config: RunnableConfig) -> Dict[str, Any]:
    """Main execution node: Use MCP tools to fulfill user requests."""
    cfg = Configuration.from_runnable_config(config)
    logger = setup_logging(cfg.debug_mode)

    # Recreate tool manager (since objects don't persist well in LangGraph state)
    tool_manager = MCPToolManager(cfg.mcp_server_config)

    # Initialize empty MCP tools list if no servers configured
    mcp_tools = []

    # Determine which tools to use - handle both dict and object state
    tools_to_use = []
    if isinstance(state, dict):
        tools_to_use = state.get('filtered_tools', []) if state.get('filtered_tools') else state.get('available_tools', [])
    else:
        tools_to_use = state.filtered_tools if hasattr(state, 'filtered_tools') and state.filtered_tools else state.available_tools

    # Only try to load MCP tools if we have a client configured
    if tool_manager.client:
        # Get tools for execution
        if cfg.tool_filtering_enabled and len(tools_to_use) > cfg.max_tools_per_step:
            # Use filtered tools
            logger.info(f"Using filtered tools: {len(tools_to_use)} tool infos")
            mcp_tools = await tool_manager.get_tools_for_filtered_execution(tools_to_use)
            logger.info(f"Retrieved {len(mcp_tools)} actual tool objects for filtered execution")
        else:
            # Use all available tools (Phase 1 behavior)
            max_tools = None if not cfg.tool_filtering_enabled else cfg.max_tools_per_step
            logger.info(f"Using all available tools (max: {max_tools})")
            mcp_tools = await tool_manager.get_tools_for_execution(state, max_tools)
            logger.info(f"Retrieved {len(mcp_tools)} tool objects for execution")
    else:
        logger.info("No MCP servers configured, proceeding with built-in tools only")

    # If only one MCP server is configured, scope tools to that server by name prefix convention.
    # This avoids binding unrelated tools that may have incompatible schemas.
    try:
        if isinstance(cfg.mcp_server_config, dict) and len(cfg.mcp_server_config) == 1 and mcp_tools:
            only_server = next(iter(cfg.mcp_server_config.keys()))
            prefix = f"{only_server.upper()}_"
            scoped = [t for t in mcp_tools if getattr(t, 'name', '').upper().startswith(prefix)]
            if scoped:
                mcp_tools = scoped
    except Exception:
        # Best-effort scoping; continue with unfiltered tools on any error
        pass

    logger.info(f"Executing with {len(mcp_tools)} available MCP tools")

    # Check if we have tools available (only warn if we have an MCP client but no tools)
    if tool_manager.client and not mcp_tools and len(tools_to_use) > 0:
        logger.warning(f"Expected {len(tools_to_use)} tools but got 0. This may indicate a tool loading issue in LangGraph Platform.")

        # Try to reload tools directly as a fallback
        try:
            if tool_manager.client:
                fallback_tools = await tool_manager.client.get_tools()
                logger.info(f"Fallback: loaded {len(fallback_tools)} tools directly from client")
                mcp_tools = fallback_tools[:cfg.max_tools_per_step] if fallback_tools else []
        except Exception as e:
            logger.error(f"Fallback tool loading failed: {e}")

    # Get built-in tools (from deepagents)
    built_in_tools = [write_todos, write_file, read_file, edit_file, ls]
    all_tools = built_in_tools + mcp_tools

    # Create system prompt - Phase 2 feature: prepend user system prompt
    instructions = get_system_prompt()
    if cfg.system_prompt:
        instructions = f"{cfg.system_prompt}\n\n{instructions}"
        logger.info("Using custom system prompt from configuration")

    # Add execution plan context if available
    execution_plan = state.get('execution_plan', []) if isinstance(state, dict) else getattr(state, 'execution_plan', [])
    if execution_plan:
        plan_context = "\n\nExecution Plan:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(execution_plan)])
        instructions += plan_context

    # Phase 2 feature: Support multiple models
    model = cfg.model
    # Extract just the model name for deepagents (remove provider prefix)
    model_name = model.split("/")[-1] if "/" in model else model
    logger.info(f"Using model: {model} (deepagents format: {model_name})")

    # Create deep agent for execution
    try:
        agent = async_create_deep_agent(
            tools=all_tools,
            instructions=instructions,
            model=model_name
        )

        # Execute the request
        result = await agent.ainvoke(state)

        return {
            "messages": result["messages"],
            "files": result.get("files", {}),
            "todos": result.get("todos", []),
            "execution_complete": True
        }

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        error_count = state.get('error_count', 0) if isinstance(state, dict) else getattr(state, 'error_count', 0)
        return {
            "last_error": f"Execution failed: {str(e)}",
            "error_count": error_count + 1
        }


def route_after_classification(state: MCPOrchestratorState) -> str:
    """Route to appropriate path based on classification."""
    classification = getattr(state, 'request_classification', 'execution')

    if classification == "simple":
        return "simple_response"
    elif classification == "clarification":
        return "ask_clarification" 
    else:
        return "discover_tools"  # Default to full execution path


# def should_interrupt_for_approval(state: MCPOrchestratorState) -> str:
#     """Determine if human approval is needed before execution."""
#     if getattr(state, 'approval_required', False):
#         return "interrupt"
#     return "continue"


async def create_mcp_orchestrator(config: Union[Configuration, Dict[str, Any]]) -> Any:
    """Create the main MCP Orchestrator agent.

    Args:
        config: Orchestrator configuration (Configuration object or dict)

    Returns:
        Compiled LangGraph agent
    """
    from langgraph.graph import StateGraph

    # Handle both Configuration objects and dictionaries from LangGraph
    if isinstance(config, dict):
        # Convert dict to Configuration object
        config = Configuration.from_runnable_config({"configurable": config})

    # Create state graph
    builder = StateGraph(MCPOrchestratorState, config_schema=Configuration)

    if config.enable_advanced_filtering:
        # Phase 3: Advanced workflow with classification and filtering
        builder.add_node("classify_request", classify_request)
        builder.add_node("simple_response", simple_response)
        builder.add_node("ask_clarification", ask_clarification)
        builder.add_node("discover_tools", discover_mcp_tools)
        builder.add_node("plan_and_filter", plan_and_filter_tools)
        builder.add_node("execute", execute_with_mcp_tools)

        # Set entry point to classification
        builder.set_entry_point("classify_request")

        # Route based on classification
        builder.add_conditional_edges(
            "classify_request",
            route_after_classification,
            {
                "simple_response": "simple_response",
                "ask_clarification": "ask_clarification", 
                "discover_tools": "discover_tools"
            }
        )

        # Simple paths end immediately
        builder.add_edge("simple_response", "__end__")
        builder.add_edge("ask_clarification", "classify_request")  # Loop back for clarification

        # Full execution path
        builder.add_edge("discover_tools", "plan_and_filter")

        # Direct flow from plan_and_filter to execute
        builder.add_edge("plan_and_filter", "execute")
        builder.add_edge("execute", "__end__")

        # Compile with interrupt support - interrupt before execute if needed
        interrupt_before = ["execute"] if config.interrupt_before_execution else []
        return builder.compile(interrupt_before=interrupt_before)

    else:
        # Phase 1 & 2: Simple workflow (existing implementation)
        builder.add_node("discover_tools", discover_mcp_tools)
        builder.add_node("execute", execute_with_mcp_tools)

        # Set entry point
        builder.set_entry_point("discover_tools")

        # Add edges
        builder.add_edge("discover_tools", "execute")
        builder.add_edge("execute", "__end__")

        # Compile and return
        return builder.compile()


# Export the main creation function
__all__ = ["create_mcp_orchestrator"]
