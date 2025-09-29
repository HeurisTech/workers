"""Utility functions for the MCP Orchestrator."""

import logging
import json
import asyncio
import os
import re
from typing import Dict, Any, List, Optional, Union
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


def parse_json_response(response: str) -> Union[Dict[str, Any], List[Any], str]:
    """Robustly parse JSON from LLM responses that may contain markdown or extra text.
    
    Args:
        response: Raw response string from LLM
        
    Returns:
        Parsed JSON object, or original string if parsing fails
    """
    if not response:
        return response
    
    logger = logging.getLogger("mcp_orchestrator")
    
    # First try direct parsing
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parsing failed: {e}")
        pass
    
    # Look for JSON in markdown code blocks
    json_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(.*?)\n```',      # ``` ... ```
        r'`([^`]*)`',               # `...`
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Better { ... } block matching
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Better [ ... ] block matching
    ]
    
    for i, pattern in enumerate(json_patterns):
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                # Clean up the match
                cleaned = match.strip()
                result = json.loads(cleaned)
                logger.debug(f"JSON parsing succeeded with pattern {i+1}")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Pattern {i+1} match failed: {e}")
                continue
    
    # Last resort: try to find any JSON-like structure and fix common issues
    try:
        # Remove common markdown artifacts
        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Try parsing the cleaned version
        result = json.loads(cleaned)
        logger.debug("JSON parsing succeeded after cleaning")
        return result
    except json.JSONDecodeError:
        pass
    
    # If all parsing fails, log and return the original response
    logger.debug(f"All JSON parsing attempts failed for response: {response[:200]}...")
    return response


def setup_logging(debug_mode: bool = False) -> logging.Logger:
    """Set up logging for the MCP Orchestrator.

    Args:
        debug_mode: Whether to enable debug logging

    Returns:
        Configured logger instance
    """
    log_level = os.getenv("MCP_ORCHESTRATOR_LOG_LEVEL", "INFO")
    if debug_mode:
        log_level = "DEBUG"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger("mcp_orchestrator")

    # Suppress noisy loggers unless in debug mode
    if not debug_mode:
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("anthropic").setLevel(logging.ERROR)

    return logger


async def analyze_with_llm(prompt: str, model: str, temperature: float = 0.3) -> Any:
    """Analyze a prompt using the specified language model.

    Args:
        prompt: The prompt to analyze
        model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022")
        temperature: Temperature for generation

    Returns:
        Parsed response (usually dict from JSON)
    """
    try:
        # Initialize the chat model
        llm = init_chat_model(model, temperature=temperature)

        # Create message and invoke
        message = HumanMessage(content=prompt)
        response = await llm.ainvoke([message])

        # Try to parse as JSON with robust parsing
        try:
            return parse_json_response(response.content)
        except (json.JSONDecodeError, AttributeError):
            # Return raw content if not JSON
            return response.content

    except Exception as e:
        raise Exception(f"LLM analysis failed with model {model}: {str(e)}")


async def classify_request_with_llm(user_request: str, model: str) -> str:
    """Classify a user request using LLM.

    Args:
        user_request: The user's request to classify
        model: Model to use for classification

    Returns:
        Classification: "simple", "clarification", or "execution"
    """
    classification_prompt = f"""You are a request classifier. Classify this user request quickly and accurately.

User Request: "{user_request}"

Categories:
1. "simple" - Simple questions, greetings, information requests that don't need tools
2. "clarification" - Ambiguous requests that need more details
3. "execution" - Clear requests requiring tool usage and task execution

Respond with ONLY: simple, clarification, or execution"""

    try:
        response = await analyze_with_llm(classification_prompt, model, temperature=0.1)
        classification = response.strip().lower() if isinstance(response, str) else "execution"

        # Validate response
        valid_classifications = {"simple", "clarification", "execution"}
        if classification not in valid_classifications:
            return "execution"  # Default to execution for safety

        return classification

    except Exception:
        # Default to execution if classification fails
        return "execution"


async def create_execution_plan(user_request: str, available_tools: List[Any], model: str) -> Dict[str, Any]:
    """Create an execution plan for the user request.

    Args:
        user_request: User's request
        available_tools: List of available tools
        model: Model to use for planning

    Returns:
        Dictionary with execution plan
    """
    tools_summary = []
    for tool in available_tools[:20]:  # Limit to prevent context overflow
        tools_summary.append({
            "name": getattr(tool, 'name', str(tool)),
            "description": getattr(tool, 'description', '')[:100]  # Truncate long descriptions
        })

    planning_prompt = f"""Create an execution plan for this user request.

User Request: {user_request}

Available Tools (sample): {json.dumps(tools_summary, indent=2)}

Create a step-by-step plan that:
1. Breaks the request into logical steps
2. Identifies which tools might be needed
3. Considers dependencies between steps
4. Flags any steps that might need human approval

Return JSON with:
- steps: Array of step descriptions (2-5 steps max)
- estimated_tools: Array of tool names that might be needed
- requires_approval: Boolean for sensitive operations
- complexity: "low", "medium", or "high"

Keep it concise but thorough."""

    try:
        plan = await analyze_with_llm(planning_prompt, model, temperature=0.2)

        # Ensure required fields exist
        if not isinstance(plan, dict):
            plan = {"steps": ["Execute user request"], "estimated_tools": [], "requires_approval": False}

        plan.setdefault("steps", ["Execute user request"])
        plan.setdefault("estimated_tools", [])
        plan.setdefault("requires_approval", False)
        plan.setdefault("complexity", "medium")

        return plan

    except Exception as e:
        # Return fallback plan
        return {
            "steps": ["Execute user request with available tools"],
            "estimated_tools": [],
            "requires_approval": False,
            "complexity": "medium",
            "planning_error": str(e)
        }


def validate_tool_schema(tool_info: Dict[str, Any]) -> bool:
    """Validate that a tool info dictionary has required fields.

    Args:
        tool_info: Dictionary containing tool information

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["name", "description"]
    return all(field in tool_info for field in required_fields)


def categorize_tools(tools: List[Any]) -> Dict[str, List[Any]]:
    """Categorize tools by type/functionality.

    Args:
        tools: List of tool objects

    Returns:
        Dictionary mapping categories to tool lists
    """
    categories = {
        "file_operations": [],
        "web_operations": [],
        "data_analysis": [],
        "communication": [],
        "development": [],
        "other": []
    }

    # Simple categorization based on tool names and descriptions
    for tool in tools:
        name = getattr(tool, 'name', '').lower()
        description = getattr(tool, 'description', '').lower()

        if any(keyword in name or keyword in description for keyword in ['file', 'read', 'write', 'edit', 'ls']):
            categories["file_operations"].append(tool)
        elif any(keyword in name or keyword in description for keyword in ['web', 'http', 'search', 'fetch', 'url']):
            categories["web_operations"].append(tool)
        elif any(keyword in name or keyword in description for keyword in ['analyze', 'data', 'query', 'process']):
            categories["data_analysis"].append(tool)
        elif any(keyword in name or keyword in description for keyword in ['send', 'email', 'message', 'notify']):
            categories["communication"].append(tool)
        elif any(keyword in name or keyword in description for keyword in ['git', 'code', 'build', 'deploy', 'test']):
            categories["development"].append(tool)
        else:
            categories["other"].append(tool)

    return categories


def estimate_context_usage(tools: List[Any], user_request: str) -> Dict[str, int]:
    """Estimate context window usage for tools and request.

    Args:
        tools: List of tools
        user_request: User's request

    Returns:
        Dictionary with token estimates
    """
    # Rough token estimation (4 chars ≈ 1 token)
    request_tokens = len(user_request) // 4

    tools_tokens = 0
    for tool in tools:
        name_tokens = len(getattr(tool, 'name', '')) // 4
        desc_tokens = len(getattr(tool, 'description', '')) // 4
        schema_tokens = 50  # Rough estimate for tool schema
        tools_tokens += name_tokens + desc_tokens + schema_tokens

    return {
        "request_tokens": request_tokens,
        "tools_tokens": tools_tokens,
        "total_estimated": request_tokens + tools_tokens,
        "tools_count": len(tools)
    }


async def health_check_server(server_name: str, server_config: Dict[str, Any]) -> bool:
    """Perform health check on a specific MCP server.

    Args:
        server_name: Name of the server
        server_config: Server configuration

    Returns:
        True if healthy, False otherwise
    """
    try:
        # This is a placeholder - in a real implementation, you'd
        # actually connect to the server and test it
        # For now, we'll just check if the config looks valid
        if "command" in server_config or "url" in server_config:
            return True
        return False
    except Exception:
        return False


def format_execution_summary(todos: List[Any], files: Dict[str, str], messages: List[Any]) -> str:
    """Format a summary of execution results.

    Args:
        todos: List of todos created
        files: Files created/modified
        messages: Message history

    Returns:
        Formatted summary string
    """
    summary_parts = []

    if todos:
        summary_parts.append(f"✅ Created {len(todos)} todos")

    if files:
        summary_parts.append(f"📁 Modified {len(files)} files")

    if messages:
        summary_parts.append(f"💬 Exchanged {len(messages)} messages")

    if not summary_parts:
        summary_parts.append("🤖 Processed request")

    return " | ".join(summary_parts)


# Export utility functions
__all__ = [
    "setup_logging",
    "analyze_with_llm", 
    "classify_request_with_llm",
    "create_execution_plan",
    "validate_tool_schema",
    "categorize_tools",
    "estimate_context_usage",
    "health_check_server",
    "format_execution_summary"
]
