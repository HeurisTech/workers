"""State management for the MCP Orchestrator.

Following the deepagents pattern with virtual filesystem and todo tracking,
combined with the workers pattern for clean state inheritance.
Enhanced with Phase 3 features for advanced filtering and human-in-loop.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def file_reducer(existing: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    """Merge file system state across agent calls."""
    if existing is None:
        existing = {}
    if new is None:
        return existing
    return {**existing, **new}


def list_reducer(existing: List[Any], new: List[Any]) -> List[Any]:
    """Merge lists across agent calls."""
    if existing is None:
        existing = []
    if new is None:
        return existing
    return existing + new


@dataclass
class Todo:
    """Individual todo item for task tracking."""
    content: str
    status: str = "pending"  # pending, in_progress, completed
    activeForm: str = ""
    priority: str = "medium"  # low, medium, high
    estimated_time: Optional[int] = None  # minutes


@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""
    name: str
    description: str
    server_name: str
    schema: Dict[str, Any]
    category: Optional[str] = None
    relevance_score: Optional[float] = None  # Phase 3: relevance for current request
    usage_frequency: int = field(default=0)  # Phase 3: how often this tool is used


@dataclass
class MCPServerStatus:
    """Status information for an MCP server."""
    name: str
    connected: bool
    tools_count: int
    last_health_check: Optional[str] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None  # Phase 3: server performance


@dataclass
class ExecutionStep:
    """Phase 3: Individual step in execution plan."""
    step_number: int
    description: str
    required_tools: List[str] = field(default_factory=list)
    depends_on: List[int] = field(default_factory=list)  # Step dependencies
    estimated_time: Optional[int] = None  # minutes
    requires_approval: bool = False
    status: str = "pending"  # pending, in_progress, completed, failed


@dataclass
class ToolFilteringResult:
    """Phase 3: Result of tool filtering process."""
    original_tool_count: int
    filtered_tool_count: int
    filtering_strategy: str  # "relevance", "plan-based", "simple"
    context_tokens_saved: Optional[int] = None
    filtering_time_ms: Optional[float] = None


@dataclass(kw_only=True)
class MCPOrchestratorState(MessagesState):
    """Main state for the MCP Orchestrator.

    Inherits from MessagesState to get message handling,
    adds virtual filesystem and MCP-specific state.
    Enhanced with Phase 3 features.
    """

    # Phase 1: Virtual filesystem (deepagents pattern)
    files: Dict[str, str] = field(default_factory=dict, metadata={"reducer": file_reducer})

    # Phase 1: Todo tracking (deepagents pattern)
    todos: List[Todo] = field(default_factory=list)

    # Phase 1: MCP server and tool information
    mcp_servers: Dict[str, MCPServerStatus] = field(default_factory=dict)
    available_tools: List[MCPToolInfo] = field(default_factory=list)

    # Phase 3: Enhanced tool and execution state
    filtered_tools: List[MCPToolInfo] = field(default_factory=list)
    execution_steps: List[ExecutionStep] = field(default_factory=list)
    tool_filtering_result: Optional[ToolFilteringResult] = field(default=None)

    # Phase 1: Basic execution state
    current_step: int = field(default=0)
    execution_plan: List[str] = field(default_factory=list)
    step_results: Dict[int, Any] = field(default_factory=dict)

    # Phase 3: Request classification and routing
    request_classification: Optional[str] = field(default=None)  # simple, clarification, execution
    awaiting_clarification: bool = field(default=False)
    clarification_history: List[str] = field(default_factory=list)

    # Phase 3: Planning and intelligence
    planned_operations: List[str] = field(default_factory=list)  # read, write, search, etc.
    complexity_estimate: str = field(default="medium")  # low, medium, high
    estimated_completion_time: Optional[int] = field(default=None)  # minutes

    # Phase 1: Error handling
    last_error: Optional[str] = field(default=None)
    error_count: int = field(default=0)
    error_history: List[str] = field(default_factory=list, metadata={"reducer": list_reducer})

    # Phase 3: Human-in-loop state
    pending_approval: Optional[Dict[str, Any]] = field(default=None)
    approval_required: bool = field(default=False)
    approved_operations: List[str] = field(default_factory=list)
    rejected_operations: List[str] = field(default_factory=list)
    human_feedback: List[str] = field(default_factory=list, metadata={"reducer": list_reducer})

    # Phase 3: Performance and analytics
    execution_start_time: Optional[float] = field(default=None)
    execution_end_time: Optional[float] = field(default=None)
    total_tokens_used: Optional[int] = field(default=None)
    tools_used_count: Dict[str, int] = field(default_factory=dict)

    # Phase 3: Advanced features
    sub_agent_results: Dict[str, Any] = field(default_factory=dict)
    context_preservation: Dict[str, Any] = field(default_factory=dict)  # For sub-agent context
    execution_complete: bool = field(default=False)

    def add_execution_step(self, description: str, required_tools: List[str] = None, 
                          requires_approval: bool = False) -> int:
        """Add a new execution step and return its number."""
        step_number = len(self.execution_steps) + 1
        step = ExecutionStep(
            step_number=step_number,
            description=description,
            required_tools=required_tools or [],
            requires_approval=requires_approval
        )
        self.execution_steps.append(step)
        return step_number

    def update_step_status(self, step_number: int, status: str) -> None:
        """Update the status of an execution step."""
        for step in self.execution_steps:
            if step.step_number == step_number:
                step.status = status
                break

    def get_active_step(self) -> Optional[ExecutionStep]:
        """Get the currently active execution step."""
        for step in self.execution_steps:
            if step.status == "in_progress":
                return step
        return None

    def get_pending_approval_steps(self) -> List[ExecutionStep]:
        """Get steps that require human approval."""
        return [step for step in self.execution_steps if step.requires_approval and step.status == "pending"]

    def record_tool_usage(self, tool_name: str) -> None:
        """Record usage of a tool for analytics."""
        if tool_name not in self.tools_used_count:
            self.tools_used_count[tool_name] = 0
        self.tools_used_count[tool_name] += 1

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution state."""
        completed_steps = [step for step in self.execution_steps if step.status == "completed"]
        failed_steps = [step for step in self.execution_steps if step.status == "failed"]

        return {
            "total_steps": len(self.execution_steps),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "current_step": self.current_step,
            "tools_available": len(self.available_tools),
            "tools_filtered": len(self.filtered_tools) if self.filtered_tools else 0,
            "error_count": self.error_count,
            "execution_complete": self.execution_complete,
            "approval_pending": self.approval_required,
            "classification": self.request_classification
        }


# Export state classes
__all__ = [
    "MCPOrchestratorState", 
    "Todo", 
    "MCPToolInfo", 
    "MCPServerStatus",
    "ExecutionStep",
    "ToolFilteringResult"
]
