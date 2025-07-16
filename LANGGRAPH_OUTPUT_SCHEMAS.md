# LangGraph Graph Output Schemas

This document describes the output formats returned by each LangGraph graph/assistant type in our system via Server-Sent Events (SSE).

## Overview

LangGraph returns streaming responses using Server-Sent Events with different schemas depending on the graph type. All responses start with metadata and then send `values` events containing the actual data.

## Common SSE Structure

```
event: metadata
data: {"run_id":"...", "attempt":1}

event: values
data: { /* graph-specific output schema */ }
```

## Output Schema Categories

### 1. Chat-Based Graphs (Messages Format)

These graphs return conversational message arrays compatible with standard chat interfaces.

#### `assist_with_playwright`

**Output Schema:**

```json
{
  "messages": [
    {
      "content": "user message",
      "additional_kwargs": {},
      "response_metadata": {},
      "type": "human",
      "name": null,
      "id": "uuid",
      "example": false
    },
    {
      "content": "AI response",
      "additional_kwargs": {"refusal": null},
      "response_metadata": {
        "token_usage": {
          "completion_tokens": 11,
          "prompt_tokens": 1159,
          "total_tokens": 1170,
          "completion_tokens_details": {...},
          "prompt_tokens_details": {...}
        },
        "model_name": "gpt-4.1-2025-04-14",
        "system_fingerprint": null,
        "id": "chatcmpl-...",
        "service_tier": "default",
        "finish_reason": "stop",
        "logprobs": null
      },
      "type": "ai",
      "name": null,
      "id": "run--...",
      "example": false,
      "tool_calls": [],
      "invalid_tool_calls": [],
      "usage_metadata": {
        "input_tokens": 1159,
        "output_tokens": 11,
        "total_tokens": 1170,
        "input_token_details": {"audio": 0, "cache_read": 0},
        "output_token_details": {"audio": 0, "reasoning": 0}
      }
    }
  ]
}
```

**Characteristics:**

- ✅ Standard chat message format
- ✅ Compatible with existing chat UIs
- ✅ Includes token usage metadata
- ❌ No planning or orchestration data

---

### 2. Planning-Based Graphs (Messages + Planning Result)

These graphs return message arrays plus additional planning/orchestration metadata.

#### `planner_style_agent`

**Output Schema:**

```json
{
  "messages": [
    /* Same message format as chat-based graphs */
  ],
  "planner_result": {
    "decision": "replace",
    "plan": [],
    "next_task": 0,
    "clarification": "Response message explaining the plan"
  }
}
```

#### `assist_with_planner`

**Output Schema:**

```json
{
  "messages": [
    /* Same message format as chat-based graphs */
  ],
  "planner_result": {
    "decision": "replace",
    "plan": [],
    "next_task": 0,
    "clarification": "Response message about task matching"
  }
}
```

**Characteristics:**

- ✅ Standard chat message format
- ✅ Compatible with existing chat UIs
- ✅ Additional planning metadata
- ✅ Can orchestrate multi-step tasks
- 📋 `planner_result.decision`: Action to take ("replace", "append", etc.)
- 📋 `planner_result.plan`: Array of planned tasks
- 📋 `planner_result.next_task`: Index of next task to execute
- 📋 `planner_result.clarification`: Human-readable explanation

---

### 3. Computer Use Graphs (Task Execution Format)

Computer automation graphs use a completely different output schema focused on task execution status and results.

#### `computer_use`

**Output Schema (In Progress):**

```json
{
  "status": "in_progress",
  "user_request": "Original user request",
  "human_intervention_reason": ""
}
```

**Output Schema (Completed):**

```json
{
  "status": "completed",
  "answer": "Description of what was accomplished",
  "user_request": "Original user request",
  "total_interactions": 1,
  "execution_time_seconds": 0.0,
  "error_message": "",
  "screenshots_taken": 1,
  "actions_performed": [
    "ActionType.wait:Waiting for user to provide more details...",
    "ActionType.screenshot:Capture the current state..."
  ],
  "human_intervention_reason": "",
  "sandbox_preserved": false,
  "task_completed_successfully": true,
  "requires_human_intervention": false
}
```

**Characteristics:**

- ❌ Not compatible with standard chat UIs
- ✅ Rich execution metadata
- ✅ Task completion status tracking
- ✅ Action logging and screenshots
- ✅ Human intervention detection
- 🔄 Status progression: "in_progress" → "completed"
- 📊 Execution metrics: time, interactions, actions
- 🖼️ Screenshot and action tracking
- ⚠️ Error handling and intervention flags

---

## Frontend Integration Implications

### Chat UI Compatibility Matrix

| Graph Type               | Standard Chat UI        | Custom Handler Needed           |
| ------------------------ | ----------------------- | ------------------------------- |
| `assist_with_playwright` | ✅ Direct support       | ❌ Not needed                   |
| `planner_style_agent`    | ✅ Use `messages` field | ⚠️ Optional: show planning info |
| `assist_with_planner`    | ✅ Use `messages` field | ⚠️ Optional: show planning info |
| `computer_use`           | ❌ Incompatible         | ✅ Required                     |

### Recommended Implementation Strategy

1. **Default Chat Handler**: Extract `messages` field for standard chat display
2. **Planning Enhancement**: Optionally display `planner_result` for planning graphs
3. **Computer Use Handler**: Custom UI for task execution tracking

### Code Example

```typescript
function handleGraphOutput(data: any, graphType: string) {
  switch (graphType) {
    case "assist_with_playwright":
      return {
        type: "chat",
        messages: data.messages,
      };

    case "planner_style_agent":
    case "assist_with_planner":
      return {
        type: "planning",
        messages: data.messages,
        plannerResult: data.planner_result,
      };

    case "computer_use":
      return {
        type: "computer_use",
        status: data.status,
        answer: data.answer,
        executionDetails: {
          totalInteractions: data.total_interactions,
          executionTime: data.execution_time_seconds,
          screenshotsTaken: data.screenshots_taken,
          actionsPerformed: data.actions_performed,
        },
      };
  }
}
```

## Testing Commands

To test these schemas yourself:

```bash
# Standard message format test
curl -X POST "http://localhost:2024/threads/{thread_id}/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "assist_with_playwright",
    "input": {"messages": [{"role": "user", "content": "test"}]}
  }'

# Planning format test
curl -X POST "http://localhost:2024/threads/{thread_id}/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "planner_style_agent",
    "input": {"messages": [{"role": "user", "content": "test"}]}
  }'

# Computer use format test
curl -X POST "http://localhost:2024/threads/{thread_id}/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "computer_use",
    "input": {"messages": [{"role": "user", "content": "test"}]}
  }'
```

## Summary

- **3 distinct output schema categories** based on graph purpose
- **Chat-based graphs** return standard message arrays
- **Planning graphs** add orchestration metadata to messages
- **Computer use graphs** use specialized task execution format
- **Frontend adaptation required** for computer use graphs
- **Backward compatibility maintained** for existing chat interfaces
