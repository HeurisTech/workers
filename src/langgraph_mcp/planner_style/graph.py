# src/langgraph_mcp/planner_style/graph.py

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
from langgraph_mcp.utils import load_chat_model

from langgraph_mcp.mcp_react_graph import make_graph

from langgraph_mcp.planner_style.config import Configuration
from langgraph_mcp.planner_style.state import PlannerResult, State

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
    if state.planner_result and state.planner_result.get_current_task():
        # there is a task to execute next
        return "execute_task"
    # couldn't plan # no task to execute next, so we need to respond to the user
    return "respond"


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

    tools = await apply(task_description, server_cfg, GetTools()) if server_cfg else []  # expert tools list
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
    
    tool_call = last_message.tool_calls[0]  # TODO: handle multiple tool calls
    
    task = state.planner_result.get_current_task()
    if not task:
        return {"messages": [AIMessage(content='We should not be in tools node without a current task.')]}

    task_expert = task.expert

    cfg = Configuration.from_runnable_config(config)
    server_cfg = cfg.get_server_config(task_expert)  # expert mcp server config
    if not server_cfg:
        return {"messages": [AIMessage(content=f'No configuration found for the expert {task_expert}.')]}

    # RunTool could raise an exception, so we need to handle it
    try:
        tool_output = await apply(task_expert, server_cfg, RunTool(tool_call['name'], **tool_call['args']))
    except Exception as e:
        tool_output = f"Error: {e}"

    return {"messages": [ToolMessage(content=tool_output, tool_call_id=tool_call['id'])]}


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


builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node("planner", planner, cache_policy=CachePolicy())
builder.add_node("execute_task", execute_task, cache_policy=CachePolicy())
builder.add_node("tools", tools, cache_policy=CachePolicy())
builder.add_node("human_input", human_input)
builder.add_node("respond", respond, cache_policy=CachePolicy())

builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    decide_planner_edge,
    {"execute_task": "execute_task", "respond": "respond"}
)
builder.add_conditional_edges(
    "execute_task",
    decide_execute_task_edge,
    {"tools": "tools", "human_input": "human_input", "planner": "planner"}
)
builder.add_edge("human_input", "execute_task")
builder.add_edge("tools", "execute_task")
builder.add_edge("respond", END)

graph = builder.compile(cache=InMemoryCache())
