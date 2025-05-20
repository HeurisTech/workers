# Strategy: With Planner

This strategy uses a dedicated "planner" agent to analyze the conversation history and decide the next steps, including which MCP server (expert) and tool to invoke.

## Files

*   **Configuration (`config.py`):** Defines configurable parameters for this strategy.
*   **State (`state.py`):** Defines the specific state managed by this graph, extending the common `InputState`.
*   **Graph (`graph.py`):** Implements the LangGraph workflow for this strategy.
*   **Prompts (`prompts.py`):** Contains the system prompts used by the planner, orchestrator, task assessment, and response generation LLMs.

## Configuration (`config.py`)

Key configurable aspects:

*   `mcp_server_config`: Dictionary holding configurations for all available MCP servers.
*   `planner_system_prompt`: System prompt guiding the planner LLM.
*   `planner_model`: LLM used for the planning step (e.g., `openai/gpt-4o`).
*   `orchestrate_system_prompt`: System prompt guiding the orchestrator LLM.
*   `orchestrate_model`: LLM used for the orchestration/tool-calling step (e.g., `openai/gpt-4o`).
*   `task_assessment_system_prompt`: System prompt for evaluating task completion status.
*   `task_assessment_model`: LLM used for task completion assessment.
*   `generate_response_system_prompt`: System prompt for generating final responses after plan completion.
*   `generate_response_model`: LLM used for generating final responses.
*   Includes methods (`get_mcp_server_descriptions`, `build_experts_context`) to format MCP server info for the planner prompt.

## State (`state.py`)

Extends the base `InputState` (which contains `messages`) with:

*   `planner_result`: An optional `PlannerResult` object containing:
    *   `decision`: Whether to `continue` the current plan or `replace` it.
    *   `plan`: A list of `Task` objects (each with `expert` name and `task` description).
    *   `next_task`: Index of the task to execute next.
    *   `clarification`: Optional message if the planner needs more info.
*   `task_completed`: Boolean flag indicating if the current task has been completed.

## Graph Workflow (`graph.py`)

1.  **START -> `planner`:**
    *   Receives the current `State` (messages).
    *   Uses the `planner_model` and `planner_system_prompt`.
    *   Considers the message history, available MCP servers (`experts`), and any existing plan.
    *   Generates a `PlannerResult` (plan, next task, decision, optional clarification).
    *   Updates the state with the `planner_result` and any clarification message.
2.  **`planner` -> `decide_planner_edge`:**
    *   If the `planner_result` contains a plan (`plan` list is not empty), transitions to `orchestrate`.
    *   Otherwise (no plan generated or planner decided to end), transitions to `END`.
3.  **`orchestrate`:**
    *   Receives the `State` including the `planner_result`.
    *   Identifies the `current_task` based on `planner_result.next_task`.
    *   Uses the `orchestrate_model` and `orchestrate_system_prompt`.
    *   Formats a prompt including message history, the full plan, and the specific `current_task` description.
    *   Fetches available tools for the `current_task.expert` (MCP server) using `mcp.GetTools()` via the `mcp_wrapper.apply` function.
    *   Binds these tools to the `orchestrate_model`.
    *   Invokes the model. The model might:
        *   Respond directly to the user.
        *   Decide to call one of the bound tools.
        *   Indicate it doesn't know how to proceed (`IDK_TAG`).
    *   Updates the `messages` in the state with the AI response (which might include tool calls).
4.  **`orchestrate` -> `decide_orchestrate_edge`:**
    *   If the last message contains `tool_calls`, transitions to `call_tool`.
    *   If the last message contains `IDK_TAG` or is asking for human input, transitions to `END`.
    *   Otherwise, transitions to `assess_task`.
5.  **`call_tool`:**
    *   Identifies the `current_task.expert` (MCP server).
    *   Extracts the `tool_call` details from the last AI message.
    *   Uses `mcp.RunTool()` via the `mcp_wrapper.apply` function to execute the tool on the specified MCP server.
    *   Adds the `ToolMessage` (containing the result or error) to the state.
6.  **`call_tool` -> `assess_task_completion`:** After executing the tool, transitions to assess if the task is complete.
7.  **`assess_task_completion`:**
    *   Uses the `task_assessment_model` and `task_assessment_system_prompt`.
    *   Provides the `current_task.task` description and recent messages as context.
    *   Evaluates if the task has been completed successfully.
    *   Updates the `task_completed` flag in the state based on the assessment.
8.  **`assess_task_completion` -> `decide_task_assessment_edge`:**
    *   If `task_completed` is `True`, transitions to `advance_to_next_task`.
    *   If `task_completed` is `False`, transitions back to `orchestrate` to continue working on the current task.
9.  **`advance_to_next_task`:**
    *   Increments the `next_task` index in `planner_result`.
    *   Resets `task_completed` to `False`.
10. **`advance_to_next_task` -> `decide_next_task_edge`:**
    *   If there are more tasks in the plan, transitions back to `orchestrate` to work on the next task.
    *   If all tasks are completed, transitions to `generate_response`.
11. **`generate_response`:**
    *   Uses the `generate_response_model` and `generate_response_system_prompt`.
    *   Considers the full conversation history.
    *   Generates a final summary response for the user.
    *   Resets the `planner_result` to `None` for the next conversation.
12. **`generate_response` -> `END`:** The workflow concludes. 