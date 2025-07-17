"""Model Context Protocol (MCP) Orchestration Graphs for LangGraph."""

from langgraph_mcp.with_planner.graph import graph as assistant_with_planner
from langgraph_mcp.planner_style.graph import graph as planner_style_agent
from langgraph_mcp.goal_oriented.graph import graph as goal_oriented_agent
from langgraph_mcp.playwright_react_graph import make_graph as assistant_with_playwright

__all__ = ["assistant_with_planner", "assistant_with_playwright", "goal_oriented_agent", "planner_style_agent"]
