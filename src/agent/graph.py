"""LangGraph workflow for the Computer Use Agent."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy
from langchain_core.runnables import RunnableConfig

from agent.configuration import AgentConfig
from agent.cua.sandbox import start_desktop_sandbox, get_desktop, _DESKTOP_CACHE
from agent.prompts import create_strategic_brain_prompt
from agent.state import (
    ExecutionState, 
    GraphInput, 
    OutputState,
    TaskStatus, 
    SemanticInstruction, 
    ActionType,
    SemanticInstructionSchema
)
from agent.vision_utils import (
    execute_semantic_instruction,
    build_brain_context,
    handle_repeated_action,
    should_request_human_intervention
)
from agent.utils import utcnow

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_state_field(state, field_name, default=None):
    """Get a field from state, handling both dict and object formats."""
    if isinstance(state, dict):
        return state.get(field_name, default)
    else:
        return getattr(state, field_name, default)

def set_state_field(state, field_name, value):
    """Set a field in state, handling both dict and object formats."""
    if isinstance(state, dict):
        state[field_name] = value
    else:
        setattr(state, field_name, value)

def safe_get_field(obj, field_name, default_value):
    """Safely get a field from an object with a default value if missing or None."""
    if hasattr(obj, field_name):
        value = getattr(obj, field_name)
        if value is not None:
            return value
    return default_value

def map_platform_to_environment(platform: str) -> str:
    """Map platform selection to CUA environment name."""
    mapping = {
        "google-chrome": "browser",
        "firefox": "browser", 
        "vscode": "editor",
    }
    return mapping.get(platform, "browser")

def cleanup_sandbox(sandbox_id: str, graceful: bool = True) -> None:
    """Clean up sandbox resources gracefully."""
    
    desktop = get_desktop(sandbox_id)
    if not desktop:
        return  # Already cleaned up
    
    try:
        if graceful:
            # Stop streaming gracefully
            if hasattr(desktop, 'stream') and desktop.stream:
                try:
                    desktop.stream.stop()
                except Exception:
                    pass  # Stream might already be stopped
            
            # Save any work or close applications gracefully
            try:
                # Save any open documents (Ctrl+S)
                desktop.key_down("ctrl")
                desktop.key_down("s")
                desktop.key_up("s")
                desktop.key_up("ctrl")
                desktop.wait(1000)
                
                # Close applications gracefully (Alt+F4)
                desktop.key_down("alt")
                desktop.key_down("f4")
                desktop.key_up("f4")
                desktop.key_up("alt")
                desktop.wait(1000)
            except Exception:
                pass  # Ignore errors during graceful cleanup
        
        # Kill the sandbox
        desktop.kill()
        
    except Exception as e:
        # Log error but don't raise - cleanup should be best effort
        print(f"Error during sandbox cleanup: {e}")
    
    finally:
        # Remove from cache
        if sandbox_id in _DESKTOP_CACHE:
            del _DESKTOP_CACHE[sandbox_id]

# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def setup_sandbox(state: ExecutionState, *, config: RunnableConfig) -> ExecutionState:
    """Set up the sandbox environment for execution."""
    
    # Get configuration from runnable config
    cfg = AgentConfig.from_runnable_config(config)
    configurable = config.get("configurable", {})
    
    # Check if we're recreating a sandbox due to an error
    needs_new_sandbox = get_state_field(state, "needs_new_sandbox", False)
    if needs_new_sandbox:
        print("Detected needs_new_sandbox flag, clearing existing sandbox_id")
        # Clear existing sandbox ID to force creation of a new one
        if isinstance(state, dict):
            state["sandbox_id"] = None
            state["sandbox_url"] = None
        else:
            state.sandbox_id = None
            state.sandbox_url = None
        # Reset the flag
        set_state_field(state, "needs_new_sandbox", False)
    
    # Extract user request from the messages array (compatible with planner_style_agent schema)
    user_request = ""
    
    # First, check if we have messages in the state (from GraphInput)
    if hasattr(state, 'messages') and state.messages:
        # Extract the last user message from the messages array
        for message in reversed(state.messages):
            if hasattr(message, 'type') and message.type == 'human':
                user_request = message.content
                break
            elif hasattr(message, 'role') and message.role == 'user':
                user_request = message.content
                break
            elif isinstance(message, dict) and message.get('role') == 'user':
                user_request = message.get('content', '')
                break
    
    # Fallback: check if we have a direct user_request in the state (backward compatibility)
    if not user_request and hasattr(state, 'user_request') and state.user_request:
        user_request = state.user_request
    # Try different ways to get the user request from config (backward compatibility)
    elif not user_request and "user_request" in configurable:
        user_request = configurable["user_request"]
    elif not user_request and "input" in configurable:
        user_request = configurable["input"]
    elif not user_request and "query" in configurable:
        user_request = configurable["query"]
    elif not user_request and "message" in configurable:
        user_request = configurable["message"]
    
    # If no user request found, use a default
    if not user_request:
        user_request = "Open Chrome and search for information"
    
    # Handle both dict and object state
    if isinstance(state, dict):
        state["user_request"] = user_request
        # Also preserve messages if they exist
        if hasattr(state, 'messages'):
            state["messages"] = getattr(state, 'messages', [])
    else:
        state.user_request = user_request
        # Also preserve messages if they exist in the input
        if not hasattr(state, 'messages') or not state.messages:
            # Try to get messages from the original state if available
            if hasattr(state, 'messages'):
                pass  # Already has messages
            else:
                state.messages = []
    
    # Check if we have existing sandbox info in the config or state
    if isinstance(state, dict):
        sandbox_id = state.get('sandbox_id') or cfg.sandbox_id
        sandbox_url = state.get('sandbox_url') or cfg.sandbox_url
    else:
        sandbox_id = getattr(state, 'sandbox_id', None) or cfg.sandbox_id
        sandbox_url = getattr(state, 'sandbox_url', None) or cfg.sandbox_url

    # Check if we can reuse an existing sandbox
    sandbox_to_use = None
    if sandbox_id and sandbox_url:
        # Verify if the sandbox is still active
        try:
            desktop = get_desktop(sandbox_id)
            if desktop:
                print(f"Verified existing sandbox is active: {sandbox_id}")
                sandbox_to_use = {
                    "sandbox_id": sandbox_id,
                    "sandbox_url": sandbox_url
                }
            else:
                print(f"Existing sandbox {sandbox_id} is not active, will create new one")
        except Exception as e:
            print(f"Failed to verify existing sandbox {sandbox_id}: {e}, will create new one")
    
    # Use verified sandbox or create new one
    if sandbox_to_use:
        if isinstance(state, dict):
            state["sandbox_id"] = sandbox_to_use["sandbox_id"]
            state["sandbox_url"] = sandbox_to_use["sandbox_url"]
        else:
            state.sandbox_id = sandbox_to_use["sandbox_id"]
            state.sandbox_url = sandbox_to_use["sandbox_url"]
        print(f"Using existing sandbox: {sandbox_to_use['sandbox_id']}")
    else:
        # Create new sandbox
        print("Creating new sandbox...")
        sandbox_details = start_desktop_sandbox(timeout=cfg.sandbox_timeout)
        if isinstance(state, dict):
            state["sandbox_id"] = sandbox_details.sandbox_id
            state["sandbox_url"] = sandbox_details.url
            state["sandbox_expires_at"] = sandbox_details.expires_at
        else:
            state.sandbox_id = sandbox_details.sandbox_id
            state.sandbox_url = sandbox_details.url
            state.sandbox_expires_at = sandbox_details.expires_at
        print(f"Created new sandbox: {sandbox_details.sandbox_id}")
    
    # Reset state for new execution and set status to in progress
    # BUT preserve previous output and user query for continuity
    if isinstance(state, dict):
        # Preserve previous context
        previous_output = state.get("message_for_user", "")
        previous_user_request = state.get("user_request", "")
        
        state["status"] = TaskStatus.in_progress
        state["_start_time"] = time.time()
        state["interaction_count"] = 0
        state["message_for_user"] = ""  # Reset for new execution
        state["current_instruction"] = None
        state["last_executor_report"] = None
        state["app_launched"] = False
        state["action_history"] = []
        state["brain_instruction_history"] = []
        state["_iteration_limit"] = cfg.iteration_limit  # Set from config
        
        # Store previous context for continuity
        state["previous_output"] = previous_output
        state["previous_user_request"] = previous_user_request
    else:
        # Preserve previous context
        previous_output = getattr(state, "message_for_user", "")
        previous_user_request = getattr(state, "user_request", "")
        
        state.status = TaskStatus.in_progress
        state._start_time = time.time()
        state.interaction_count = 0
        state.message_for_user = ""  # Reset for new execution
        state.current_instruction = None
        state.last_executor_report = None
        state.app_launched = False
        state.action_history = []
        state.brain_instruction_history = []
        state._iteration_limit = cfg.iteration_limit  # Set from config
        
        # Store previous context for continuity
        state.previous_output = previous_output
        state.previous_user_request = previous_user_request
    
    return state

async def strategic_brain(state: ExecutionState, *, config: RunnableConfig) -> ExecutionState:
    """Strategic brain that analyzes screen state and gives semantic instructions."""
    
    try:
        print(f"Strategic brain started - {state.get_interaction_summary()}")
        
        # Get configuration from runnable config
        cfg = AgentConfig.from_runnable_config(config)
        
        # Check if sandbox is available
        desktop = get_desktop(state.sandbox_id)
        if not desktop:
            print("Desktop sandbox not found in brain, setting flag to recreate sandbox")
            state.status = TaskStatus.failed
            state.message_for_user = "Desktop sandbox not found"
            # Set a special flag to indicate we need to recreate the sandbox
            state.needs_new_sandbox = True
            # Clear existing sandbox_id to force creation of a new one
            state.sandbox_id = ""
            return state
            
        brain_provider = cfg.brain_provider
        brain_model = cfg.brain_model
        
        if brain_provider == "anthropic":
            api_key = cfg.anthropic_api_key
            if not api_key:
                print("No Anthropic API key found in config or environment")
                state.status = TaskStatus.failed
                state.message_for_user = "Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable."
                return state
        else:
            api_key = cfg.openai_api_key
            brain_model = cfg.model_planner
            if not api_key:
                print("No OpenAI API key found in config or environment")
                state.status = TaskStatus.failed
                state.message_for_user = "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
                return state
        
        # Build context for strategic brain
        context = build_brain_context(
            user_request=state.user_request,
            interaction_summary=state.get_interaction_summary(),
            recent_actions=state.get_recent_actions_summary(),
            recent_brain_instructions=state.get_recent_brain_instructions_summary(),
            previous_output=state.previous_output,
            previous_user_request=state.previous_user_request,
            last_report=state.last_executor_report
        )
        
        # Create the strategic brain prompt using utility function
        brain_prompt = create_strategic_brain_prompt(context)
        
        # Create LangChain client with structured output using factory
        from agent.llm_factory import get_cached_structured_llm
            
        structured_llm = get_cached_structured_llm(
            provider=brain_provider,
                model=brain_model,
                api_key=api_key,
            schema=SemanticInstructionSchema,
            temperature=1.0
        )
        
        # Prepare messages for the brain
        messages = [{"role": "user", "content": brain_prompt}]
        
        # Add screenshot if available
        if state.last_executor_report and state.last_executor_report.screenshot_b64:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{state.last_executor_report.screenshot_b64}"
                        }
                    }
                ]
            })
        
        # Get instruction from brain
        instruction_response = structured_llm.invoke(messages)
        
        # Check if human intervention is needed BEFORE processing the instruction
        needs_intervention, intervention_reason = should_request_human_intervention(state, instruction_response)
        if needs_intervention:
            print(f"Human intervention needed: {intervention_reason}")
            state.status = TaskStatus.human_intervention_required
            state.human_intervention_reason = intervention_reason
            state.preserve_sandbox = True  # Don't clean up sandbox for human intervention
            state.message_for_user = f"Human intervention required: {intervention_reason}"
            return state
        
        print(f"Brain instruction: {instruction_response.action_type} - {instruction_response.description}")
        
        # Handle repeated actions using utility function
        action_desc = f"{instruction_response.action_type}:{instruction_response.target_element or instruction_response.description}"
        instruction_response = handle_repeated_action(instruction_response, state, action_desc)
        
        # Check for simple application launch tasks and mark complete early if needed
        user_request_lower = state.user_request.lower()
        if (state.interaction_count >= 2 and 
            any(term in user_request_lower for term in ["open", "launch", "start"]) and
            any(app in user_request_lower for app in ["chrome", "firefox", "browser", "google chrome"])):
            if state.app_launched:
                print("Simple browser launch task detected and browser is launched. Marking as complete.")
                instruction_response.is_task_complete = True
                instruction_response.completion_message = f"Successfully launched the browser as requested."
        
        # Check for high interaction count and force different approaches or completion
        if state.interaction_count >= 5 and not instruction_response.is_task_complete:
            # Instead of forcing completion, try different approaches based on context
            if (instruction_response.action_type == "scroll" and 
                state.interaction_count >= 12):
                print(f"High interaction count ({state.interaction_count}) with scroll action. Trying to go back and restart.")
                instruction_response.action_type = "press_key"
                instruction_response.description = "Go back to previous page to try a different approach"
                instruction_response.key_sequence = ["alt+Left"]
                instruction_response.reasoning = "High interaction count with scrolling suggests current approach isn't working"
            elif (instruction_response.action_type == "click_element" and 
                  state.interaction_count >= 15):
                print(f"High interaction count ({state.interaction_count}) with click_element action. Trying to go back and restart.")
                instruction_response.action_type = "press_key"
                instruction_response.description = "Go back to previous page to try a different approach"
                instruction_response.key_sequence = ["alt+Left"]
                instruction_response.reasoning = "High interaction count with clicking suggests current approach isn't working"
            elif (instruction_response.action_type == "type_text" and 
                  state.interaction_count >= 8):
                print(f"High interaction count ({state.interaction_count}) with type_text action. Trying to press Enter.")
                instruction_response.action_type = "press_key"
                instruction_response.description = "Press Enter to execute the search"
                instruction_response.key_sequence = ["enter"]
                instruction_response.reasoning = "High interaction count with typing suggests we should execute the search"
            elif state.interaction_count >= 20:
                # Only mark as incomplete after trying many different approaches
                print(f"Very high interaction count ({state.interaction_count}). Marking as incomplete due to being stuck.")
                instruction_response.is_task_complete = True
                instruction_response.completion_message = "Task could not be completed due to repeated failed attempts. The requested information may not be easily accessible or the approach needs to be reconsidered."
        
        # Convert to SemanticInstruction object
        instruction = SemanticInstruction(
            action_type=ActionType(instruction_response.action_type),
            description=instruction_response.description,
            target_element=safe_get_field(instruction_response, 'target_element', ''),
            text_content=safe_get_field(instruction_response, 'text_content', ''),
            key_sequence=safe_get_field(instruction_response, 'key_sequence', []),
            scroll_direction=safe_get_field(instruction_response, 'scroll_direction', 'down'),
            wait_seconds=safe_get_field(instruction_response, 'wait_seconds', 2),
            url=safe_get_field(instruction_response, 'url', ''),
            reasoning=safe_get_field(instruction_response, 'reasoning', ''),
            is_task_complete=safe_get_field(instruction_response, 'is_task_complete', False),
            completion_message=safe_get_field(instruction_response, 'completion_message', '')
        )
        
        # Check for repeated brain instructions BEFORE storing
        brain_instruction_desc = f"{instruction.action_type.value}:{instruction.description}"
        if state.is_repeating_brain_instruction(brain_instruction_desc):
            print(f"Detected repeated brain instruction: {brain_instruction_desc}")
            print(f"Recent brain instructions: {state.get_recent_brain_instructions_summary()}")
            
            # Force a different approach or completion
            if instruction.action_type == ActionType.click_element:
                # If we keep trying to click the same element, try scrolling or going back
                if state.interaction_count >= 15:
                    print("High interaction count with repeated clicking. Trying to go back and restart approach.")
                    instruction.action_type = ActionType.press_key
                    instruction.description = "Go back to previous page to try a different approach"
                    instruction.target_element = ""
                    instruction.key_sequence = ["alt+Left"]
                    instruction.reasoning = "Going back to try a different approach after repeated clicking failed"
                else:
                    print("Changing approach from repeated clicking to scrolling.")
                    instruction.action_type = ActionType.scroll
                    instruction.description = "Scroll to find different elements or content"
                    instruction.target_element = ""
                    instruction.scroll_direction = "down"
                    instruction.reasoning = "Breaking repetitive clicking by scrolling to find new elements"
            elif instruction.action_type == ActionType.scroll:
                # If we keep scrolling, try going back or taking a different approach
                if state.interaction_count >= 12:
                    print("Repeated scrolling detected. Trying to go back and restart approach.")
                    instruction.action_type = ActionType.press_key
                    instruction.description = "Go back to previous page to try a different approach"
                    instruction.target_element = ""
                    instruction.key_sequence = ["alt+Left"]
                    instruction.reasoning = "Going back to try a different approach after repeated scrolling failed"
                else:
                    print("Changing approach from repeated scrolling to navigation.")
                    instruction.action_type = ActionType.navigate
                    instruction.description = "Navigate to a search engine to try a different approach"
                    instruction.url = "https://www.google.com"
                    instruction.reasoning = "Breaking repetitive scrolling by navigating to a fresh search page"
            elif instruction.action_type == ActionType.type_text:
                # If we keep typing the same thing, press Enter
                print("Changing approach from repeated typing to pressing Enter.")
                instruction.action_type = ActionType.press_key
                instruction.description = "Press Enter to execute the search"
                instruction.text_content = ""
                instruction.key_sequence = ["enter"]
                instruction.reasoning = "Breaking repetitive typing by pressing Enter to execute the search"
            elif instruction.action_type == ActionType.screenshot:
                # If we keep taking screenshots, try going back
                print("Repeated screenshots detected. Trying to go back and restart approach.")
                instruction.action_type = ActionType.press_key
                instruction.description = "Go back to previous page to try a different approach"
                instruction.target_element = ""
                instruction.key_sequence = ["alt+Left"]
                instruction.reasoning = "Going back to try a different approach after repeated screenshots"
            else:
                # For other repeated actions, try going back first, then complete if still stuck
                if state.interaction_count >= 15:
                    print("High interaction count with repeated actions. Marking as incomplete due to being stuck.")
                    instruction.is_task_complete = True
                    instruction.completion_message = "Task could not be completed due to repeated failed attempts. The requested information may not be easily accessible or the approach needs to be reconsidered."
                else:
                    print("Changing approach from repeated action to going back.")
                    instruction.action_type = ActionType.press_key
                    instruction.description = "Go back to previous page to try a different approach"
                    instruction.target_element = ""
                    instruction.key_sequence = ["alt+Left"]
                    instruction.reasoning = "Going back to try a different approach after repeated actions failed"
        
        # Store instruction in state
        state.set_brain_instruction(instruction)
        
        # Check if task is complete
        if instruction.is_task_complete:
            state.status = TaskStatus.completed
            state.message_for_user = instruction.completion_message
            print(f"Task completed: {instruction.completion_message}")
        
        return state
        
    except Exception as e:
        print(f"Failed to get brain instruction: {e}")
        state.status = TaskStatus.failed
        state.message_for_user = f"Brain analysis failed: {str(e)}"
        return state
        
    except Exception as e:
        print(f"Error in strategic brain: {e}")
        state.status = TaskStatus.failed
        state.message_for_user = f"Brain error: {str(e)}"
        return state

async def vision_executor(state: ExecutionState, *, config: RunnableConfig) -> ExecutionState:
    """Vision-capable executor that interprets semantic instructions and performs actions."""
    
    try:
        print(f"Vision executor started - {state.get_interaction_summary()}")
        
        # Increment interaction counter
        state.increment_interaction()
        
        # Get config and check for max interactions
        cfg = AgentConfig.from_runnable_config(config)
        if state.interaction_count >= cfg.iteration_limit:
            print(f"Maximum interactions reached ({cfg.iteration_limit})")
            state.status = TaskStatus.failed
            state.message_for_user = f"Maximum interactions reached ({cfg.iteration_limit}). Task may be too complex or stuck in a loop."
            return state
        
        # Get desktop
        desktop = get_desktop(state.sandbox_id)
        if not desktop:
            print("Desktop sandbox not found, setting flag to recreate sandbox")
            state.status = TaskStatus.failed
            state.message_for_user = "Desktop sandbox not found"
            # Set a special flag to indicate we need to recreate the sandbox
            state.needs_new_sandbox = True
            return state
        
        # Launch application if not already launched
        if not state.app_launched:
            print(f"Launching {state.application_platform}...")
            try:
                if state.application_platform == "google-chrome":
                    desktop.launch("google-chrome")
                elif state.application_platform == "firefox":
                    desktop.launch("firefox")
                elif state.application_platform == "vscode":
                    desktop.launch("code")
                else:
                    desktop.launch("google-chrome")
                
                desktop.wait(5000)  # Wait for application to start
                state.app_launched = True
                print(f"Successfully launched {state.application_platform}")
            except Exception as e:
                print(f"Error launching application: {e}")
                state.status = TaskStatus.failed
                state.message_for_user = f"Failed to launch application: {str(e)}"
                return state
        
        # Execute the semantic instruction
        instruction = state.current_instruction
        action_performed = None
        action_result = None
        error = None
        elements_found = []
        
        if instruction:
            print(f"Executing instruction: {instruction.action_type} - {instruction.description}")
            
            # For click_element actions, use OpenAI Computer Use model for visual analysis
            if instruction.action_type == ActionType.click_element:
                try:
                    # Take screenshot first
                    screenshot_data = desktop.screenshot()
                    if screenshot_data:
                        # Convert screenshot to base64 for OpenAI API
                        import base64
                        screenshot_b64 = base64.b64encode(screenshot_data).decode('utf-8')
                        
                        # Use OpenAI Computer Use model to analyze screenshot and click
                        from agent.cua.client import take_action
                        from agent.cua.actions import execute_click_action_with_validation
                        
                        # Create messages for OpenAI Computer Use model
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Please analyze this screenshot and click on: {instruction.target_element}. Description: {instruction.description}"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{screenshot_b64}"
                                        }
                                    }
                                ]
                            }
                        ]
                        
                        # Call OpenAI Computer Use model
                        response = await take_action(
                            messages=messages,
                            model="gpt-4o",  # Use vision-capable model
                            api_key=cfg.openai_api_key,
                            environment=map_platform_to_environment(state.application_platform or "google-chrome")
                        )
                        
                        # Process the response and execute the action
                        if response and 'choices' in response:
                            choice = response['choices'][0]
                            if 'message' in choice and 'tool_calls' in choice['message']:
                                tool_calls = choice['message']['tool_calls']
                                if tool_calls:
                                    tool_call = tool_calls[0]
                                    if tool_call['function']['name'] == 'computer':
                                        import json
                                        args = json.loads(tool_call['function']['arguments'])
                                        
                                        # Execute the computer action with proper validation
                                        if args.get('action') == 'click' and 'coordinate' in args:
                                            x, y = args['coordinate']
                                            
                                            # Use the new validation function for better click handling
                                            click_result = execute_click_action_with_validation(
                                                desktop, x, y, instruction.target_element
                                            )
                                            
                                            if click_result['success']:
                                                action_performed = f"click_element: {instruction.target_element}"
                                                action_result = f"Successfully clicked on {instruction.target_element} at coordinates ({x}, {y})"
                                                elements_found = [instruction.target_element]
                                                print(f"Successfully clicked on {instruction.target_element} at ({x}, {y})")
                                                
                                                # Wait for the page to respond to the click
                                                import time
                                                time.sleep(2)
                                                print("Waited 2 seconds for page to respond to click")
                                            else:
                                                error = f"Click failed: {click_result['error']}"
                                                action_performed = f"click_element_failed: {instruction.target_element}"
                                                action_result = f"Failed to click on {instruction.target_element} at ({x}, {y}): {click_result['error']}"
                                                print(f"Click failed: {click_result['error']}")
                                        elif args.get('action') == 'screenshot':
                                            # If the model decides to take a screenshot instead of clicking
                                            action_performed = "screenshot"
                                            action_result = "OpenAI model took a screenshot instead of clicking"
                                            print("OpenAI model took a screenshot instead of clicking")
                                        else:
                                            error = f"Unexpected action from model: {args.get('action', 'unknown')}"
                                            print(f"Unexpected action from model: {args.get('action', 'unknown')}")
                                    else:
                                        error = f"Unexpected tool call: {tool_call['function']['name']}"
                                        print(f"Unexpected tool call: {tool_call['function']['name']}")
                                else:
                                    error = "No tool calls in response"
                                    print("No tool calls in response")
                            else:
                                error = "No tool calls in message"
                                print("No tool calls in message")
                        else:
                            error = "Invalid response from OpenAI model"
                            print("Invalid response from OpenAI model")
                            
                    else:
                        error = "Failed to take screenshot for visual analysis"
                        print("Failed to take screenshot for visual analysis")
                        
                except Exception as e:
                    error = f"Visual analysis failed: {str(e)}"
                    print(f"Visual analysis error: {e}")
            
            else:
                # For non-click actions, use the existing utility function
                action_performed, action_result, error, elements_found = execute_semantic_instruction(
                    desktop, instruction
                )
        
        # Always take a screenshot after action to report current state
        try:
            screenshot_data = desktop.screenshot()
            if screenshot_data:
                print(f"Screenshot taken: {len(screenshot_data)} bytes")
                
                # Create executor report
                report = state.create_executor_report(
                    screenshot_data=screenshot_data,
                    action_performed=action_performed,
                    action_result=action_result,
                    error=error,
                    elements_found=elements_found
                )
                
                # Store report in state
                state.set_executor_report(report)
                
                print(f"Executor report created and stored")
                
            else:
                print("Failed to take screenshot")
                state.status = TaskStatus.failed
                state.message_for_user = "Failed to take screenshot"
                return state
                
        except Exception as e:
            print(f"Screenshot error: {e}")
            state.status = TaskStatus.failed
            state.message_for_user = f"Screenshot error: {str(e)}"
            return state
        
        return state
        
    except Exception as e:
        print(f"Error in vision executor: {e}")
        state.status = TaskStatus.failed
        state.message_for_user = f"Executor error: {str(e)}"
        return state

async def cleanup_node(state) -> ExecutionState:
    """Clean up sandbox resources when execution is complete."""
    
    try:
        status = get_state_field(state, "status", "")
        preserve_sandbox = get_state_field(state, "preserve_sandbox", False)
        
        print(f"Cleanup node: status={status}, type={type(status)}, preserve_sandbox={preserve_sandbox}")
        
        # Ensure the final message is preserved and displayed immediately
        final_message = get_state_field(state, "message_for_user", "")
        if final_message:
            print(f"FINAL RESULT: {final_message}")
        
        # Handle both enum and string status values
        if status == TaskStatus.completed or str(status) == "completed":
            # Ensure we have a completion message
            if not final_message:
                set_state_field(state, "message_for_user", "Task completed successfully.")
                final_message = "Task completed successfully."
                print(f"FINAL RESULT: {final_message}")
        
        # Clean up sandbox in background without blocking the final result (only if not preserving)
        if not preserve_sandbox:
            sandbox_id = get_state_field(state, "sandbox_id", "")
            if sandbox_id:
                # Schedule cleanup without waiting for it
                asyncio.create_task(cleanup_sandbox_async(sandbox_id))
                print("Sandbox cleanup scheduled")
        else:
            print("Sandbox preserved for human intervention")
        
        # Return the final state immediately so the user gets the result
        return state
        
    except Exception as e:
        print(f"Cleanup failed: {e}")
        # Even if cleanup fails, preserve any existing result
        existing_message = get_state_field(state, "message_for_user", "")
        if not existing_message:
            set_state_field(state, "message_for_user", f"Task completed but cleanup failed: {e}")
        return state


async def create_final_output(state) -> OutputState:
    """Create the final output state to return to the user."""
    
    try:
        # Calculate execution time if we have a start time
        execution_time = 0.0
        if hasattr(state, '_start_time') and state._start_time is not None:
            execution_time = time.time() - state._start_time
        
        # Check if this is human intervention - if so, preserve sandbox and don't clean up
        status = get_state_field(state, "status", "")
        preserve_sandbox = get_state_field(state, "preserve_sandbox", False)
        
        if status == TaskStatus.human_intervention_required or str(status) == "human_intervention_required":
            print("Human intervention required - preserving sandbox and creating output")
            # Don't schedule any cleanup for human intervention
        else:
            # For completed/failed tasks, schedule cleanup in background
            sandbox_id = get_state_field(state, "sandbox_id", "")
            if sandbox_id and not preserve_sandbox:
                asyncio.create_task(cleanup_sandbox_async(sandbox_id))
                print("Sandbox cleanup scheduled in background")
        
        # Create the output state from the execution state
        output_state = OutputState.from_execution_state(state, execution_time)
        
        print(f"FINAL OUTPUT CREATED:")
        print(f"  Status: {output_state.status}")
        print(f"  Answer: {output_state.answer}")
        print(f"  Interactions: {output_state.total_interactions}")
        print(f"  Execution Time: {output_state.execution_time_seconds:.1f}s")
        print(f"  Requires Human Intervention: {output_state.requires_human_intervention}")
        print(f"  Sandbox Preserved: {output_state.sandbox_preserved}")
        
        return output_state
        
    except Exception as e:
        print(f"Error creating final output: {e}")
        # Return a failure output state
        return OutputState(
            status=TaskStatus.failed,
            answer=f"Failed to create final output: {str(e)}",
            user_request=getattr(state, 'user_request', ''),
            error_message=str(e)
        )


async def cleanup_sandbox_async(sandbox_id: str) -> None:
    """Asynchronously clean up sandbox resources."""
    try:
        print("Starting background sandbox cleanup...")
        await asyncio.sleep(5)  # Brief delay to let the result return first
        cleanup_sandbox(sandbox_id, graceful=True)
        print("Background sandbox cleanup completed")
    except Exception as e:
        print(f"Background cleanup failed: {e}")

# ---------------------------------------------------------------------------
# Conditional logic
# ---------------------------------------------------------------------------

def should_continue_from_brain(state) -> str:
    """Determine the next step after brain analysis."""
    status = get_state_field(state, "status", "")
    needs_new_sandbox = get_state_field(state, "needs_new_sandbox", False)
    
    print(f"Brain conditional check: status={status}")
    
    # Check if we need to recreate the sandbox
    if needs_new_sandbox:
        print("Going back to setup_sandbox to create a new sandbox")
        # Reset the flag so we don't loop
        set_state_field(state, "needs_new_sandbox", False)
        return "setup_sandbox"
    
    # Handle both enum and string status values
    if status == TaskStatus.failed or str(status) == "failed":
        print("Going to cleanup due to failed status")
        return "cleanup"
    elif status == TaskStatus.completed or str(status) == "completed":
        print("Going to cleanup due to completed status")
        return "cleanup"
    elif status == TaskStatus.human_intervention_required or str(status) == "human_intervention_required":
        print("Going to final_output due to human intervention required")
        return "final_output"
    else:
        print("Going to executor")
        return "executor"

def should_continue_from_executor(state) -> str:
    """Determine the next step after executor action."""
    status = get_state_field(state, "status", "")
    needs_new_sandbox = get_state_field(state, "needs_new_sandbox", False)
    
    print(f"Executor conditional check: status={status}")
    
    # Check if we need to recreate the sandbox
    if needs_new_sandbox:
        print("Going back to setup_sandbox to create a new sandbox")
        # Reset the flag so we don't loop
        set_state_field(state, "needs_new_sandbox", False)
        return "setup_sandbox"
    
    # Handle both enum and string status values
    if status == TaskStatus.failed or str(status) == "failed":
        print("Going to cleanup due to failed status")
        return "cleanup"
    elif status == TaskStatus.completed or str(status) == "completed":
        print("Going to cleanup due to completed status")
        return "cleanup"
    elif status == TaskStatus.human_intervention_required or str(status) == "human_intervention_required":
        print("Going to final_output due to human intervention required")
        return "final_output"
    else:
        print("Going back to brain for next decision")
        return "brain"

def create_graph() -> StateGraph:
    """Create the Computer Use Agent graph with semantic brain-executor feedback loop."""
    
    # Create the graph with input schema and config schema
    graph = StateGraph(ExecutionState, input=GraphInput, output=OutputState, config_schema=AgentConfig)
    
    # Add nodes
    graph.add_node("setup_sandbox", setup_sandbox, cache_policy=CachePolicy())
    graph.add_node("brain", strategic_brain, cache_policy=CachePolicy())
    graph.add_node("executor", vision_executor, cache_policy=CachePolicy())
    graph.add_node("cleanup", cleanup_node, cache_policy=CachePolicy())
    graph.add_node("final_output", create_final_output, cache_policy=CachePolicy())
    
    # Add edges
    graph.add_edge(START, "setup_sandbox")
    graph.add_edge("setup_sandbox", "brain")
    
    # Add conditional edges for brain-executor feedback loop
    graph.add_conditional_edges(
        "brain",
        should_continue_from_brain,
        {
            "executor": "executor",  # Go to executor to perform action
            "cleanup": "cleanup",    # Go to cleanup if failed/completed
            "final_output": "final_output",  # Go to final output if human intervention required
            "setup_sandbox": "setup_sandbox",  # Go back to setup_sandbox to create a new sandbox
        }
    )
    
    # Add conditional edges from executor
    graph.add_conditional_edges(
        "executor",
        should_continue_from_executor,
        {
            "brain": "brain",        # Go back to brain for next decision
            "cleanup": "cleanup",    # Go to cleanup if failed/completed
            "final_output": "final_output",  # Go to final output if human intervention required
            "setup_sandbox": "setup_sandbox",  # Go back to setup_sandbox to create a new sandbox
        }
    )
    
    # After cleanup, create final output
    graph.add_edge("cleanup", "final_output")
    graph.add_edge("final_output", END)
    
    return graph

# Legacy functions for backward compatibility
async def executor(state: ExecutionState, *, config: RunnableConfig) -> ExecutionState:
    """Legacy executor function - redirects to new vision_executor."""
    return await vision_executor(state, config=config)

async def planner_node(state: ExecutionState, *, config: RunnableConfig) -> ExecutionState:
    """Legacy planner function - redirects to new strategic_brain."""
    return await strategic_brain(state, config=config)


# Create and compile the graph
workflow_graph = create_graph().compile(cache=InMemoryCache())
workflow_graph.name = "Computer Use Agent"

# Note: To use the recursion limit from config, invoke the graph like this:
# from agent.configuration import get_config
# config = get_config()
# result = workflow_graph.invoke(input_data, config={"recursion_limit": config.recursion_limit})

__all__ = ["workflow_graph"]
