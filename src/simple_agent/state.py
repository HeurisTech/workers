"""State definitions for the simple agent."""

from dataclasses import dataclass
from typing import Annotated, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages


@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the simple agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages.

    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`.
    """


class State(MessagesState):
    """Main state for the simple agent.

    Inherits from MessagesState to get message handling capabilities.
    This provides the `messages` field with proper add_messages reducer.
    """
    pass
