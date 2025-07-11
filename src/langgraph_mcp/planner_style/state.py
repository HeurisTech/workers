from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime

from pydantic import BaseModel

from langgraph_mcp.state import InputState


class Task(BaseModel):
    """Represents a task assigned to an expert for execution."""
    
    expert: str
    """The name of the expert responsible for executing the task."""
    
    task: str
    """A brief description of the task to be performed."""


class GoalAssessmentResult(BaseModel):
    """Represents the output of goal assessment when evaluating plan-goal alignment."""

    is_aligned: bool
    """Boolean indicating if the plan aligns with the goal (alignment_score >= 0.7)."""

    alignment_score: float
    """Confidence score between 0 and 1 for how well the plan aligns with the goal."""

    explanation: str
    """Detailed explanation of the assessment reasoning."""

    missing_elements: list[str]
    """List of important elements missing from the plan."""

    suggested_improvements: list[str]
    """List of ways to improve the plan's alignment with the goal."""


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

    user_goal: Optional[str] = field(default=None)
    """The user's overarching goal that persists across plan changes.
    
    This goal is used to ensure all plans and tasks align with the user's intent.
    """

    goal_last_updated: Optional[datetime] = field(default=None)
    """Timestamp of when the user goal was last updated.
    
    Used to track goal changes and ensure plan-goal alignment validation.
    """

    goal_assessment: Optional[GoalAssessmentResult] = field(default=None)
    """The result of goal-plan alignment assessment.
    
    Contains information about whether the current plan aligns with the user's goal.
    """

    pending_user_input: list[str] = field(default_factory=list)
    """Queue of pending user inputs received asynchronously.
    
    Allows user to provide input at any time during task execution.
    """

    user_input_timestamp: Optional[datetime] = field(default=None)
    """Timestamp of the last user input received.
    
    Used to track when user provided input during ongoing task execution.
    """