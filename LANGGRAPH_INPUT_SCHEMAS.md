# LangGraph Graph Input Schemas

This document describes the expected input formats for each LangGraph graph/assistant type in our system.

## Overview

Different LangGraph graphs expect different input schemas. The main difference is:

- **Standard graphs**: Use `messages` array format (compatible with chat interfaces)
- **Computer Use graphs**: Use `user_request` string format (specialized for computer automation)

## Available Graphs

Based on `/assistants/search` endpoint, we have these graphs:

| Graph ID                 | Assistant ID                           | Input Format   | Description                                 |
| ------------------------ | -------------------------------------- | -------------- | ------------------------------------------- |
| `goal_oriented_agent`    | `08ab9ba0-aaee-594b-9fb0-d7a8e3afc738` | `messages`     | General planning and orchestration          |
| `assist_with_planner`    | `940edd11-f3fc-5023-8844-ad56d5df745a` | `messages`     | Task planning with expert delegation        |
| `assist_with_playwright` | `d1e5b1c1-917d-5050-a39f-ae9e03a6a9ff` | `messages`     | Browser automation specialist               |
| `computer_use`           | Multiple IDs                           | `user_request` | Computer interaction and desktop automation |

## Input Schema Details

### 1. Messages Format (Standard Graphs)

**Used by:** `goal_oriented_agent`, `assist_with_planner`, `assist_with_playwright`

```json
{
  "assistant_id": "goal_oriented_agent",
  "input": {
    "messages": [
      {
        "role": "user",
        "content": "Your message here"
      }
    ]
  }
}
```

**Example Request:**

```bash
curl -X POST http://localhost:2024/threads/{thread_id}/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "goal_oriented_agent",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "hello, how are you?"
        }
      ]
    }
  }'
```

**Response Format:**

- Returns streaming events with `messages` array
- Compatible with assistant-ui components
- Standard LangChain message format

### 2. User Request Format (Computer Use Graphs)

**Used by:** `computer_use`

```json
{
  "assistant_id": "computer_use",
  "input": {
    "user_request": "Your request as a string"
  }
}
```

**Example Request:**

```bash
curl -X POST http://localhost:2024/threads/{thread_id}/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "computer_use",
    "input": {
      "user_request": "open google chrome on computer"
    }
  }'
```

**Response Format:**

- Returns streaming events with status updates
- Includes `user_request`, `status`, and `human_intervention_reason` fields
- Specialized for computer automation workflows

## Validation Errors

### Computer Use with Wrong Format

If you send `messages` to `computer_use`, you get:

```json
{
  "error": "ValidationError",
  "message": "1 validation error for GraphInput\nuser_request\n  Field required [type=missing, input_value={}, input_type=dict]"
}
```

### Standard Graphs with Wrong Format

Standard graphs are more flexible and may accept various formats, but `messages` is the recommended format.

## Current Frontend Issue

**Problem:** Our frontend chat components always send the `messages` format, but computer-use workers use the `computer_use` graph which expects `user_request` format.

**Impact:** Computer-use workers receive validation errors and cannot process chat messages.

**Solution Required:** We need input format transformation based on the worker's graph type:

- `computer_use` graph → Transform messages to `user_request` format
- All other graphs → Keep `messages` format

## Implementation Requirements

1. **Detection Logic:** Check worker's `assistant_id` or `graph_id` to determine input format
2. **Transformation:** Convert between formats in the API layer
3. **Response Handling:** Handle different response formats appropriately

### Message to User Request Transformation

```javascript
// For computer_use graphs
const transformToUserRequest = (messages) => {
  const lastUserMessage = messages.filter((msg) => msg.role === "user").pop();

  return {
    user_request: lastUserMessage?.content || "",
  };
};
```

### User Request to Messages Transformation (for responses)

```javascript
// Convert computer_use responses back to message format for UI
const transformToMessages = (computerUseResponse) => {
  return {
    messages: [
      // ... existing messages,
      {
        role: "assistant",
        content:
          computerUseResponse.status === "completed"
            ? "Task completed successfully"
            : "Task in progress...",
      },
    ],
  };
};
```

## Testing Commands

```bash
# Test goal_oriented_agent (messages format)
curl -X POST http://localhost:2024/threads/{thread_id}/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "goal_oriented_agent", "input": {"messages": [{"role": "user", "content": "hello"}]}}'

# Test computer_use (user_request format)
curl -X POST http://localhost:2024/threads/{thread_id}/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "computer_use", "input": {"user_request": "open chrome"}}'
```
