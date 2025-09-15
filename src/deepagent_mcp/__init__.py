"""MCP Orchestrator - An intelligent orchestrator for Model Context Protocol servers.

Complete implementation across all phases:
- Phase 1: MCP server discovery, tool loading, and intelligent execution using deepagents
- Phase 2: Enhanced configuration with custom system prompts and multi-model support  
- Phase 3: Advanced tool filtering with intelligent planning and human-in-loop capabilities

Built on langchain-mcp-adapters for MCP integration with configurable RunnableConfig pattern.
"""

from deepagent_mcp.agent import create_mcp_orchestrator
from deepagent_mcp.config import Configuration, SUPPORTED_MODELS
from deepagent_mcp.state import (
    MCPOrchestratorState, 
    Todo, 
    MCPToolInfo, 
    MCPServerStatus,
    ExecutionStep,
    ToolFilteringResult
)
from deepagent_mcp.tools import MCPToolManager, IntelligentToolFilter
from deepagent_mcp.prompts import (
    get_system_prompt,
    get_classification_prompt, 
    get_planning_prompt,
    get_tool_analysis_prompt,
    get_subagent_prompt
)
from deepagent_mcp.utils import (
    setup_logging,
    analyze_with_llm,
    classify_request_with_llm,
    create_execution_plan,
    validate_tool_schema,
    categorize_tools,
    estimate_context_usage,
    health_check_server,
    format_execution_summary
)

# Version info
__version__ = "0.3.0"  # Updated to reflect Phase 3 completion
__phase__ = "3"

# Main exports organized by phase
__all__ = [
    # Core orchestrator (All Phases)
    "create_mcp_orchestrator",

    # Configuration (Phase 1 + 2)
    "Configuration",
    "SUPPORTED_MODELS",

    # State management (All Phases)
    "MCPOrchestratorState",
    "Todo",
    "MCPToolInfo", 
    "MCPServerStatus",
    "ExecutionStep",        # Phase 3
    "ToolFilteringResult",  # Phase 3

    # Tools and managers (All Phases)
    "MCPToolManager",
    "IntelligentToolFilter",  # Phase 3

    # Prompts (All Phases)
    "get_system_prompt",
    "get_classification_prompt",    # Phase 3
    "get_planning_prompt",          # Phase 3
    "get_tool_analysis_prompt",     # Phase 3
    "get_subagent_prompt",          # Phase 3

    # utilities (All Phases)
    "setup_logging",
    "analyze_with_llm",             # Phase 3
    "classify_request_with_llm",    # Phase 3
    "create_execution_plan",        # Phase 3
    "validate_tool_schema",         # Phase 3
    "categorize_tools",             # Phase 3
    "estimate_context_usage",       # Phase 3
    "health_check_server",          # Phase 3
    "format_execution_summary",     # Phase 3

    # Metadata
    "__version__",
    "__phase__",
]

# Phase completion status
PHASE_STATUS = {
    "Phase 1": "✅ COMPLETED - Core MCP orchestration with deepagents integration",
    "Phase 2": "✅ COMPLETED - Enhanced config with system prompts and multi-model support", 
    "Phase 3": "✅ COMPLETED - Advanced filtering with planning and human-in-loop capabilities"
}

def get_phase_info() -> dict:
    """Get information about implementation phases."""
    return {
        "current_version": __version__,
        "current_phase": __phase__,
        "phase_status": PHASE_STATUS,
        "features": {
            "Phase 1": [
                "MCP server discovery and connection",
                "Tool loading and caching",
                "Intelligent task execution with deepagents",
                "Virtual filesystem and todo management",
                "Error handling and recovery"
            ],
            "Phase 2": [
                "Custom system prompt injection",
                "Multi-model support (GPT-4o, Claude-4, Gemini-2.5)",
                "Model validation and normalization",
                "Enhanced configuration management"
            ],
            "Phase 3": [
                "Intelligent request classification and routing",
                "Advanced tool filtering with relevance scoring",
                "Execution planning and step management", 
                "Human-in-loop approval workflow",
                "Performance analytics and monitoring",
                "Context-aware tool selection"
            ]
        }
    }

def print_status():
    """Print current implementation status."""
    info = get_phase_info()
    print(f"🤖 MCP Orchestrator v{info['current_version']}")
    print(f"📊 Current Phase: {info['current_phase']}")
    print()
    for phase, status in info["phase_status"].items():
        print(f"{status}")
    print()
    print("🔧 Available Features:")
    for phase, features in info["features"].items():
        print(f"\n{phase}:")
        for feature in features:
            print(f"  ✅ {feature}")

if __name__ == "__main__":
    print_status()
