from datetime import datetime, timezone
import os
from typing import Literal, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from langgraph_mcp import mcp_wrapper as mcp
from langgraph_mcp.state import InputState
from langgraph_mcp.utils import load_chat_model

from langgraph_mcp.with_planner.config import Configuration
from langgraph_mcp.with_planner.state import State, PlannerResult

# Tags for special message responses
IDK_TAG = "[::IDK::]"

class TaskAssessmentResult(BaseModel):
    """Output schema for task assessment LLM evaluation."""
    is_completed: bool = Field(description="Boolean indicating if the task is complete")
    explanation: str = Field(description="Explanation for the assessment")
    confidence: float = Field(description="Confidence score between 0 and 1")

async def planner(state: State, *, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    # get configurations
    configuration = Configuration.from_runnable_config(config)
    # build a chat prompt template with instructions for the plan evaluation task and messages in the memory
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.planner_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    # load the chat model configured for the planning task
    model = load_chat_model(configuration.planner_model)
    # let's get the current plan from the state
    current_plan = state.planner_result.plan if state.planner_result else []
    # let's build the experts list available for the planning task
    experts = configuration.build_experts_context()
    # let's build the final prompt with all the context and memory
    context = await prompt.ainvoke(
        {
            "messages": state.messages,
            "experts": experts,
            "plan": current_plan,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    # lets convey the output structure desired and call the model
    response = await model.with_structured_output(PlannerResult).ainvoke(context, config)
    result = {"planner_result": response}
    # if the model asks for clarification, we'll add the clarification seeking message to the state as well
    if response.clarification:
        result["messages"] = [AIMessage(content=response.clarification)]
    # add planner result to state
    return result

def decide_planner_edge(state: State) -> str:
    if state.planner_result.clarification:
        return END
    if state.planner_result.plan:
        return "orchestrate_tools"
    return END

async def orchestrate_tools(state: State, *, config: RunnableConfig, special_instructions: str = None) -> dict[str, list[BaseMessage]]:
    # get configurations
    configuration = Configuration.from_runnable_config(config)
    # build a chat prompt template with instructions for orchestrating across expert tools
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.orchestrate_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    # load the chat model configured for the task of orchestrating across expert tools
    model = load_chat_model(configuration.orchestrate_model)
    # get the current plan from the state
    current_plan = state.planner_result.plan
    # get the current task from the plan
    current_task = state.planner_result.get_current_task()
    # build the final prompt with all the context and memory
    context = await prompt.ainvoke(
        {
            "messages": state.messages,
            "plan": [task.model_dump() for task in current_plan],
            "task": current_task.task if current_task else "",
            "idk_tag": IDK_TAG,
            "special_instructions": special_instructions,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    # get the tools for the current expert using our MCP wrapper and bind them to the model
    if current_task:
        server_config = configuration.get_server_config(current_task.expert)
        tools = await mcp.apply(current_task.expert, server_config, mcp.GetTools())
        model = model.bind_tools(tools)
    # call the model
    response = await model.ainvoke(context, config)

    return {"messages": [response]}

def decide_orchestrate_tools_edge(state: State) -> Literal["call_tool", "assess_task", "end"]:
    """Decide what to do after orchestration."""
    if not state.messages:
        return "end"
    
    last_message = state.messages[-1]
    
    # Check if the last message has tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"
    
    # If the message is asking for human input, end the current flow
    if "I need more information from you" in last_message.content:
        return "human_input"
    
    # If this is a task completion, check assessment
    if isinstance(last_message, (AIMessage, ToolMessage)):
        return "assess_task"
    
    # Default end: this covers IDK_TAG case as well as ASK_USER case
    return "end"

def human_input(state: State, *, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    last_message = state.messages[-1]
    human_message = interrupt(last_message.content)
    return {
        "messages": [HumanMessage(content=human_message)]
    }


async def call_tool(state: State, *, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    # Get the current task
    current_task = state.planner_result.get_current_task()
    if not current_task:
        return {"messages": [AIMessage(content="Error: No current task available for tool execution.")]}

    # Fetch mcp server config
    configuration = Configuration.from_runnable_config(config)
    server_config = configuration.get_server_config(current_task.expert)

    # Execute MCP server Tool
    tool_call = state.messages[-1].tool_calls[0]
    try:
        tool_output = await mcp.apply(
            current_task.expert, 
            server_config, 
            mcp.RunTool(tool_call['name'], **tool_call['args'])
        )
    except Exception as e:
        tool_output = f"Error: {e}"
        
    return {"messages": [ToolMessage(content=tool_output, tool_call_id=tool_call['id'])]}

async def assess_task_completion(state: State, *, config: RunnableConfig) -> dict[str, bool]:
    """Assess whether the current task has been completed."""
    # Get configurations
    configuration = Configuration.from_runnable_config(config)
    
    # Get the current task
    current_task = state.planner_result.get_current_task()
    if not current_task:
        return {"task_completed": True}  # If no task, consider it completed
    
    # Build a chat prompt for task assessment
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.task_assessment_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    
    # Load the chat model for task assessment
    model = load_chat_model(configuration.task_assessment_model)
    
    # Build the context
    context = await prompt.ainvoke(
        {
            "task": current_task.task,
            "messages": state.messages,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    
    # Call the model with the structured output parser
    result = await model.with_structured_output(TaskAssessmentResult).ainvoke(context, config)
    
    # Return the task completion status
    return {"task_completed": result.is_completed and result.confidence >= 0.7}

def decide_task_assessment_edge(state: State) -> Literal["next_task", "orchestrate_tools"]:
    """Decide whether to move to the next task or continue with the current one."""
    if state.task_completed:
        return "next_task"
    return "orchestrate_tools"

def advance_to_next_task(state: State) -> dict[str, Union[PlannerResult, bool]]:
    """Advance to the next task in the plan."""
    if not state.planner_result:
        return {}
        
    # Create a copy of the current planner result
    planner_result = state.planner_result.model_copy()
    
    # Advance to the next task
    planner_result.next_task += 1
    
    # Reset task-related state
    result = {
        "planner_result": planner_result,
        "task_completed": False
    }
    
    return result

def decide_next_task_edge(state: State) -> Literal["orchestrate_tools", "generate_response"]:
    """Decide whether there are more tasks or the plan is complete."""
    if (state.planner_result and 
        state.planner_result.next_task < len(state.planner_result.plan)):
        return "orchestrate_tools"
    return "generate_response"

async def generate_response(state: State, *, config: RunnableConfig) -> dict[str, Union[list[BaseMessage], PlannerResult]]:
    """Generate a final AI response at the end of plan execution."""
    # Get configurations
    configuration = Configuration.from_runnable_config(config)
    
    # Build a chat prompt for final response
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.generate_response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    
    # Load the chat model for generating the final response
    model = load_chat_model(configuration.generate_response_model)
    
    # Build the context
    context = await prompt.ainvoke(
        {
            "messages": state.messages,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    
    # Call the model
    response = await model.ainvoke(context, config)
    
    # Reset planner state for future interactions
    return {
        "messages": [response],
        "planner_result": None
    }

# Define the state graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add all the nodes
builder.add_node("planner", planner)
builder.add_node("orchestrate_tools", orchestrate_tools)
builder.add_node("human_input", human_input)
builder.add_node("call_tool", call_tool)
builder.add_node("assess_task_completion", assess_task_completion)
builder.add_node("advance_to_next_task", advance_to_next_task)
builder.add_node("generate_response", generate_response)

# Add the edges
builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    decide_planner_edge,
    {
        "orchestrate_tools": "orchestrate_tools",
        END: END,
    }
)
builder.add_conditional_edges(
    "orchestrate_tools",
    decide_orchestrate_tools_edge,
    {
        "call_tool": "call_tool",
        "human_input": "human_input",
        "assess_task": "assess_task_completion",
        "end": END,
    }
)
builder.add_edge("call_tool", "assess_task_completion")
builder.add_edge("human_input", "orchestrate_tools")
builder.add_conditional_edges(
    "assess_task_completion",
    decide_task_assessment_edge,
    {
        "next_task": "advance_to_next_task",
        "orchestrate_tools": "orchestrate_tools",
    }
)
builder.add_conditional_edges(
    "advance_to_next_task",
    decide_next_task_edge,
    {
        "orchestrate_tools": "orchestrate_tools",
        "generate_response": "generate_response",
    }
)
builder.add_edge("generate_response", END)

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
DB_URI = os.getenv("DATABASE_URI")
checkpointer = AsyncPostgresSaver.from_conn_string(DB_URI)
# Compile the graph
graph = builder.compile(checkpointer=checkpointer)
graph.name = "AssistantGraphWithPlanner"