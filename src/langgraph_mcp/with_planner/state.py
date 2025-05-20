from dataclasses import dataclass, field
from typing import Optional, Literal

from pydantic import BaseModel

from langgraph_mcp.state import InputState


class Task(BaseModel):
    """Represents a task assigned to an expert for execution."""
    
    expert: str
    """The name of the expert responsible for executing the task."""
    
    task: str
    """A brief description of the task to be performed."""


class PlannerResult(BaseModel):
    """Represents the output of the planner when determining the next course of action."""

    decision: Literal["continue", "replace"]
    """Indicates whether to continue with the existing plan or replace it with a new one."""

    plan: list[Task]
    """Ordered list of tasks to be executed, assigned to available experts."""

    next_task: int
    """Index of the next task to execute within the `tasks` list."""

    clarification: Optional[str] = None
    """Optional clarification message if user input requires further disambiguation."""

    def get_current_task(self) -> Optional[Task]:
        """Returns the current task based on next_task index, or None if out of bounds."""
        if 0 <= self.next_task < len(self.plan):
            return self.plan[self.next_task]
        return None


@dataclass(kw_only=True)
class State(InputState):
    """Extends InputState to include planner result and task completion status.

    This state variant maintains the planner's decision, task plan, and tracks
    task completion status to manage plan execution flow.
    """

    planner_result: Optional[PlannerResult] = field(default=None)
    """The result from the planner, including task assignments and execution decisions.

    If no planner result is available, this remains `None`.
    """

    task_completed: bool = field(default=False)
    """Indicates whether the current task has been completed.
    
    This is used to determine whether to advance to the next task or continue with the current one.
    """