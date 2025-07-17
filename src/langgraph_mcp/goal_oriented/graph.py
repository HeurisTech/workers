# src/langgraph_mcp/goal_oriented/graph.py

from datetime import datetime, timezone
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy

from langgraph_mcp.state import InputState
from langgraph_mcp.mcp_wrapper import apply, GetTools, RunTool
from langgraph_mcp.tools import get_merged_tools, execute_merged_tool, is_custom_tool
from langgraph_mcp.utils import load_chat_model

from langgraph_mcp.mcp_react_graph import make_graph

from langgraph_mcp.goal_oriented.config import Configuration
from langgraph_mcp.goal_oriented.state import PlannerResult, State, GoalAssessmentResult

# Tags for special message responses
ASK_USER_FOR_INFO_TAG = "[ASK_USER]"
TASK_COMPLETE_TAG = "[TASK_COMPLETE]"
IDK_TAG = "[IDK]"

EXPERTS_NEEDING_MULTI_GRAPH_RUNS_WITHIN_AN_MCP_SESSION = ["playwright"]


async def planner(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Build the plan or advance to the next task."""
    
    # If a task was just completed, advance to the next task.
    if state.task_completed and state.planner_result:
        planner_result = state.planner_result.model_copy()
        planner_result.next_task += 1
        planner_result.decision = "continue"  # continue with the plan
        return {
            "planner_result": planner_result,
            "task_completed": False,  # Reset task completion status
        }

    # Let LLM build a plan or reflect on why the current task is not complete
    # plan / re-plan / clarify
    cfg = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg.planner_system_prompt),
        ("placeholder", "{messages}")
    ])
    model = load_chat_model(cfg.planner_model)
    experts = cfg.build_experts_context()
    context = await prompt.ainvoke(
        {
            "messages": state.messages,
            "experts": experts,
            "plan": [],
            "user_goal": state.user_goal or "No specific goal set",
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.with_structured_output(PlannerResult).ainvoke(context, config)
    result: Dict[str, Any] = {"planner_result": response}
    if isinstance(response, PlannerResult) and response.clarification:
        result["messages"] = [AIMessage(content=response.clarification)]
    return result

def decide_planner_edge(state: State) -> str:
    # Check for pending user input first
    if has_pending_user_input(state):
        return "async_user_input_handler"
    
    if state.planner_result and state.planner_result.get_current_task():
        # there is a task to execute next, but first assess goal alignment
        if state.user_goal:
            return "assess_goal_alignment"
        else:
            return "execute_task"
    # couldn't plan # no task to execute next, so we need to respond to the user
    return "respond"


def decide_goal_assessment_edge(state: State) -> str:
    """Decide what to do after goal assessment."""
    # Check for pending user input first
    if has_pending_user_input(state):
        return "async_user_input_handler"
    
    if state.goal_assessment and not state.goal_assessment.is_aligned:
        # Goal is not aligned, ask for clarification
        return "respond"
    # Goal is aligned or no assessment, proceed to execute task
    return "execute_task"


async def execute_task(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    if not state.planner_result:
        return {"messages": [AIMessage(content='We should not be in execute_task node without a plan.')]}

    task = state.planner_result.get_current_task()
    if not task:
        return {"messages": [AIMessage(content='We should not be in execute_task node without a current task.')]}

    task_expert = task.expert
    task_description = task.task

    cfg = Configuration.from_runnable_config(config)
    server_cfg = cfg.get_server_config(task_expert)  # expert mcp server config
    if not server_cfg:
        return {"messages": [AIMessage(content=f'No configuration found for the expert {task_expert}.')]}

    # Get merged tools (both MCP and custom tools)
    tools = await get_merged_tools(task_expert, server_cfg) if server_cfg else []
    if not tools:
        return {"messages": [AIMessage(content=f'No tools available with the expert {task_expert}.')]}
    
    #######################################################################
    if task_expert in EXPERTS_NEEDING_MULTI_GRAPH_RUNS_WITHIN_AN_MCP_SESSION:
        async with make_graph(cfg.execute_task_model.replace('/', ':'), task_expert, server_cfg) as subgraph:
            subgraph_result = await subgraph.ainvoke({"messages": state.messages})
            return {
                "messages": subgraph_result["messages"][len(state.messages):],
                "task_completed": True
            }
    #######################################################################
    
    model = load_chat_model(cfg.execute_task_model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg.execute_task_system_prompt),
        ("placeholder", "{messages}")
    ])
    context = await prompt.ainvoke(
        {
            "messages": state.messages,
            "expert": task_expert,
            "task": task_description,
            "ask_user_for_info_tag": ASK_USER_FOR_INFO_TAG,
            "task_complete_tag": TASK_COMPLETE_TAG,
            "idk_tag": IDK_TAG,
            "system_time": datetime.now(tz=timezone.utc).isoformat()
        },
        config
    )
    response = await model.bind_tools(tools).ainvoke(context, config)
    result: Dict[str, Any] = {"messages": [response]}
    if isinstance(response, AIMessage):
        if content := response.content:
            if TASK_COMPLETE_TAG in content:
                result["task_completed"] = True
    return result

def decide_execute_task_edge(state: State) -> str:
    """
    Routing the outcomes of the execute_task node.
    |----------------|------------------------------------------------|
    | Outcome        | How to recognize the outcome                   |
    |----------------|------------------------------------------------|
    | Call tool      | tool_calls section (JSON function call only)   |
    | Ask user       | Message starting with {ask_user_for_info_tag}  |
    | Task complete  | Message starting with {task_complete_tag}      |
    | Don't know     | Message starting with {idk_tag}                |
    |----------------|------------------------------------------------|
    """
    # Check for pending user input first
    if has_pending_user_input(state):
        return "async_user_input_handler"
    
    last_msg = state.messages[-1]
    # If it's a tool call, go to tools
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    # If it's a plain message, check the content for tags
    content = getattr(last_msg, "content", "") or ""
    if content.startswith(ASK_USER_FOR_INFO_TAG):
        return "human_input"
    # task complete or don't know, and default: go to planner
    return "planner"


async def tools(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    if not state.planner_result:
        return {"messages": [AIMessage(content='Error: We should not be in tools node without a plan.')]}

    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": [AIMessage(content="Error: We should not be in tools node without tool_calls.")]}
    
    task = state.planner_result.get_current_task()
    if not task:
        return {"messages": [AIMessage(content='We should not be in tools node without a current task.')]}

    task_expert = task.expert

    cfg = Configuration.from_runnable_config(config)
    server_cfg = cfg.get_server_config(task_expert)  # expert mcp server config
    if not server_cfg:
        return {"messages": [AIMessage(content=f'No configuration found for the expert {task_expert}.')]}

    # Handle ALL tool calls, not just the first one
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']

        # ##TODO: Hardcoded organization_id injection for knowledge retrieval tool
        # This should be made more generic and configurable
        if tool_name == "retrieve_organizational_knowledge":
            if 'organization_id' not in tool_args or not tool_args.get('organization_id'):
                tool_args['organization_id'] = cfg.organization_id
                
        try:
            tool_output = await execute_merged_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                server_name=task_expert,
                server_config=server_cfg
            )
        except Exception as e:
            tool_output = f"Error: {e}"
        
        tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call['id']))

    return {"messages": tool_messages}


def human_input(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    last = state.messages[-1]
    return {"messages": [HumanMessage(content=last.content)]}


async def respond(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    cfg = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg.generate_response_system_prompt),
        ("placeholder", "{messages}")
    ])
    context = await prompt.ainvoke(
        {"messages": state.messages, "system_time": datetime.now(tz=timezone.utc).isoformat()},
        config
    )
    model = load_chat_model(cfg.generate_response_model)
    response = await model.ainvoke(context, config)
    return {"messages": [response]}


async def assess_goal_alignment(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Assess whether the current plan aligns with the user's goal."""
    if not state.user_goal or not state.planner_result:
        # No goal or plan to assess, return default alignment
        return {"goal_assessment": GoalAssessmentResult(
            is_aligned=True,
            alignment_score=1.0,
            explanation="No goal or plan to assess",
            missing_elements=[],
            suggested_improvements=[]
        )}
    
    cfg = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg.goal_assessment_system_prompt),
        ("placeholder", "{messages}")
    ])
    
    model = load_chat_model(cfg.goal_assessment_model)
    
    # Prepare context for goal assessment
    context = await prompt.ainvoke(
        {
            "messages": state.messages,
            "goal": state.user_goal,
            "plan": [task.model_dump() for task in state.planner_result.plan],
            "system_time": datetime.now(tz=timezone.utc).isoformat()
        },
        config
    )
    
    # Get goal assessment result
    assessment = await model.with_structured_output(GoalAssessmentResult).ainvoke(context, config)
    
    # If the plan is not aligned with the goal, add a message about it
    result = {"goal_assessment": assessment}
    if not assessment.is_aligned:
        alignment_message = f"Goal alignment check: {assessment.explanation}"
        if assessment.suggested_improvements:
            alignment_message += f"\n\nSuggested improvements: {', '.join(assessment.suggested_improvements)}"
        result["messages"] = [AIMessage(content=alignment_message)]
    
    return result


async def async_user_input_handler(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Handle asynchronous user input during task execution.
    
    This node processes user inputs that have been added to the pending_user_input
    queue externally (via AsyncInputManager). It does NOT interrupt the graph execution.
    """
    
    # Check if there are pending user inputs
    if state.pending_user_input:
        # Process the first pending input
        current_input = state.pending_user_input[0]
        remaining_inputs = state.pending_user_input[1:]
        
        # Check if this is a goal update
        if any(keyword in current_input.lower() for keyword in ['goal', 'objective', 'purpose', 'want to']):
            # This might be a goal update, process it
            return {
                "pending_user_input": remaining_inputs,
                "user_input_timestamp": datetime.now(),
                "messages": [HumanMessage(content=current_input)],
                "user_goal": current_input,  # Simple goal extraction
                "goal_last_updated": datetime.now()
            }
        else:
            # Regular user input
            return {
                "pending_user_input": remaining_inputs,
                "user_input_timestamp": datetime.now(),
                "messages": [HumanMessage(content=current_input)]
            }
    
    # No pending input to process - this shouldn't happen since
    # conditional edges only route here when pending input exists
    return {}


def has_pending_user_input(state: State) -> bool:
    """Check if there are pending user inputs."""
    return len(state.pending_user_input) > 0


def check_for_async_input(state: State) -> str:
    """Check if there's pending user input that needs processing."""
    if has_pending_user_input(state):
        return "async_user_input_handler"
    return "continue"


def decide_async_input_edge(state: State) -> str:
    """Decide what to do after processing async user input."""
    # After processing user input, check if goal was updated
    if state.user_input_timestamp and state.goal_last_updated:
        # Goal may have been updated, re-plan
        return "planner"
    
    # If there's still pending input, process it
    if has_pending_user_input(state):
        return "async_user_input_handler"
    
    # No more pending input, continue with current task
    if state.planner_result and state.planner_result.get_current_task():
        return "execute_task"
    
    # No current task, go to planner
    return "planner"


builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node("planner", planner, cache_policy=CachePolicy())
builder.add_node("assess_goal_alignment", assess_goal_alignment, cache_policy=CachePolicy())
builder.add_node("execute_task", execute_task, cache_policy=CachePolicy())
builder.add_node("tools", tools, cache_policy=CachePolicy())
builder.add_node("human_input", human_input)
builder.add_node("async_user_input_handler", async_user_input_handler)
builder.add_node("respond", respond, cache_policy=CachePolicy())

builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    decide_planner_edge,
    {
        "assess_goal_alignment": "assess_goal_alignment", 
        "execute_task": "execute_task", 
        "respond": "respond",
        "async_user_input_handler": "async_user_input_handler"
    }
)
builder.add_conditional_edges(
    "assess_goal_alignment",
    decide_goal_assessment_edge,
    {
        "execute_task": "execute_task", 
        "respond": "respond",
        "async_user_input_handler": "async_user_input_handler"
    }
)
builder.add_conditional_edges(
    "execute_task",
    decide_execute_task_edge,
    {
        "tools": "tools", 
        "human_input": "human_input", 
        "planner": "planner",
        "async_user_input_handler": "async_user_input_handler"
    }
)
builder.add_conditional_edges(
    "async_user_input_handler",
    decide_async_input_edge,
    {
        "planner": "planner",
        "execute_task": "execute_task",
        "async_user_input_handler": "async_user_input_handler"
    }
)
builder.add_edge("human_input", "execute_task")
builder.add_edge("tools", "execute_task")
builder.add_edge("respond", END)

graph = builder.compile(cache=InMemoryCache())