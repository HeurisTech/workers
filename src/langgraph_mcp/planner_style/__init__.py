"""Enhanced planner style agent with goal tracking, custom tools, and async input.

This module provides the enhanced planner style agent which includes:
- Goal tracking and plan-goal alignment assessment
- Custom tools infrastructure alongside MCP tools
- Asynchronous user input handling during task execution
"""

# Initialize tools when the module is imported
try:
    from langgraph_mcp.tools import register_tools
    register_tools()
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register custom tools: {e}", UserWarning)
