from datetime import datetime
from typing import Optional, Dict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message.

    This function extracts the text content from various message formats.

    Args:
        msg: The message object to extract text from.

    Returns:
        The extracted text content of the message.

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content={"text": "World"}))
        'World'
        >>> get_message_text(HumanMessage(content=[{"text": "Hello"}, " ", {"text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name: String in the format 'provider/model'.
        
    Returns:
        The loaded chat model instance.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    if provider == "lm-studio" or model == "lm-studio":
        return ChatOpenAI(
            base_url="http://localhost:1234/v1",
            model=model,
            temperature=0.7,
            api_key="lm-studio"
        )
    else:
        return init_chat_model(model, model_provider=provider)


class AsyncInputManager:
    """Manages asynchronous user input for running LangGraph instances.
    
    This class provides functionality to add user input to a running graph's state
    without interrupting the graph execution. This enables truly asynchronous
    user input during task execution.
    """
    
    def __init__(self, langgraph_url: str = "http://localhost:2024"):
        """Initialize the async input manager.
        
        Args:
            langgraph_url: URL of the LangGraph API server
        """
        from langgraph_sdk import get_client
        self.client = get_client(url=langgraph_url)
    
    async def add_user_input(
        self, 
        thread_id: str, 
        user_input: str,
        detect_goal_change: bool = True
    ) -> bool:
        """Add user input to a running graph's pending input queue.
        
        This function updates the graph's state while it's running, adding
        the user input to the pending_user_input queue. The graph will
        pick up this input at the next conditional edge check.
        
        Args:
            thread_id: The thread ID of the running graph
            user_input: The user's input text
            detect_goal_change: Whether to detect if this input contains a goal change
            
        Returns:
            True if input was successfully added, False otherwise
        """
        try:
            # Get current state to read existing pending inputs
            current_state = self.client.threads.get_state(thread_id)
            
            # Get existing pending inputs or create empty list
            existing_pending = current_state.values.get("pending_user_input", [])
            
            # Add new input to the queue
            updated_pending = list(existing_pending) + [user_input.strip()]
            
            # Prepare state update
            state_update = {
                "pending_user_input": updated_pending,
                "user_input_timestamp": datetime.now().isoformat()
            }
            
            # If goal change detection is enabled, check for goal keywords
            if detect_goal_change:
                goal_keywords = ['goal', 'objective', 'purpose', 'want to', 'aim to']
                if any(keyword in user_input.lower() for keyword in goal_keywords):
                    state_update["user_goal"] = user_input.strip()
                    state_update["goal_last_updated"] = datetime.now().isoformat()
            
            # Update the running graph's state
            self.client.threads.update_state(
                thread_id=thread_id,
                values=state_update,
                as_node="async_input_injector"
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to add user input: {e}")
            return False
    
    async def add_goal_change(
        self, 
        thread_id: str, 
        new_goal: str
    ) -> bool:
        """Explicitly update the user's goal while the graph is running.
        
        Args:
            thread_id: The thread ID of the running graph
            new_goal: The new goal text
            
        Returns:
            True if goal was successfully updated, False otherwise
        """
        try:
            # Get current pending inputs
            current_state = self.client.threads.get_state(thread_id)
            existing_pending = current_state.values.get("pending_user_input", [])
            
            # Add goal change as input and update goal
            updated_pending = list(existing_pending) + [new_goal.strip()]
            
            state_update = {
                "pending_user_input": updated_pending,
                "user_input_timestamp": datetime.now().isoformat(),
                "user_goal": new_goal.strip(),
                "goal_last_updated": datetime.now().isoformat()
            }
            
            self.client.threads.update_state(
                thread_id=thread_id,
                values=state_update,
                as_node="goal_change_injector"
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to update goal: {e}")
            return False
    
    async def get_pending_inputs(self, thread_id: str) -> list[str]:
        """Get the current pending inputs for a thread.
        
        Args:
            thread_id: The thread ID to check
            
        Returns:
            List of pending user inputs
        """
        try:
            current_state = self.client.threads.get_state(thread_id)
            return current_state.values.get("pending_user_input", [])
        except Exception as e:
            print(f"Failed to get pending inputs: {e}")
            return []
    
    async def clear_pending_inputs(self, thread_id: str) -> bool:
        """Clear all pending inputs for a thread.
        
        Args:
            thread_id: The thread ID to clear inputs for
            
        Returns:
            True if inputs were successfully cleared, False otherwise
        """
        try:
            self.client.threads.update_state(
                thread_id=thread_id,
                values={"pending_user_input": []},
                as_node="input_clearer"
            )
            return True
        except Exception as e:
            print(f"Failed to clear pending inputs: {e}")
            return False


async def add_user_input_to_running_graph(
    thread_id: str, 
    user_input: str,
    langgraph_url: str = "http://localhost:2024",
    detect_goal_change: bool = True
) -> bool:
    """Add user input to a running graph.
    
    Args:
        thread_id: The thread ID of the running graph
        user_input: The user's input text
        langgraph_url: URL of the LangGraph API server
        detect_goal_change: Whether to detect goal changes
        
    Returns:
        True if input was successfully added, False otherwise
    """
    manager = AsyncInputManager(langgraph_url)
    return await manager.add_user_input(thread_id, user_input, detect_goal_change)


async def update_goal_in_running_graph(
    thread_id: str, 
    new_goal: str,
    langgraph_url: str = "http://localhost:2024"
) -> bool:
    """Update the goal in a running graph.
    
    Args:
        thread_id: The thread ID of the running graph
        new_goal: The new goal text
        langgraph_url: URL of the LangGraph API server
        
    Returns:
        True if goal was successfully updated, False otherwise
    """
    manager = AsyncInputManager(langgraph_url)
    return await manager.add_goal_change(thread_id, new_goal)


def add_user_input_sync(
    thread_id: str, 
    user_input: str,
    langgraph_url: str = "http://localhost:2024",
    detect_goal_change: bool = True
) -> bool:
    """Synchronous version of add_user_input_to_running_graph.
    
    Args:
        thread_id: The thread ID of the running graph
        user_input: The user's input text
        langgraph_url: URL of the LangGraph API server
        detect_goal_change: Whether to detect goal changes
        
    Returns:
        True if input was successfully added, False otherwise
    """
    import asyncio
    return asyncio.run(
        add_user_input_to_running_graph(thread_id, user_input, langgraph_url, detect_goal_change)
    )


def update_goal_sync(
    thread_id: str, 
    new_goal: str,
    langgraph_url: str = "http://localhost:2024"
) -> bool:
    """Synchronous version of update_goal_in_running_graph.
    
    Args:
        thread_id: The thread ID of the running graph
        new_goal: The new goal text
        langgraph_url: URL of the LangGraph API server
        
    Returns:
        True if goal was successfully updated, False otherwise
    """
    import asyncio
    return asyncio.run(
        update_goal_in_running_graph(thread_id, new_goal, langgraph_url)
    )


