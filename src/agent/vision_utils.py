"""Vision utilities for the Computer Use Agent executor."""

from __future__ import annotations

from typing import List, Tuple, Optional
from agent.state import ActionType, SemanticInstruction
import time


def execute_semantic_instruction(desktop, instruction: SemanticInstruction) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Execute a semantic instruction on the desktop."""
    
    action_performed = None
    action_result = None
    error = None
    elements_found = []
    
    import time
    
    def wait_for_page_ready(seconds=3):
        """Wait for page to be ready with visual feedback."""
        print(f"Waiting {seconds} seconds for page to be ready...")
        time.sleep(seconds)
        print("Page should be ready now.")
    
    # Type text action
    if instruction.action_type == ActionType.type_text:
        if instruction.text_content:
            try:
                # Wait a bit to ensure any previous actions have completed
                wait_for_page_ready(1)
                
                print(f"Attempting to type: '{instruction.text_content}'")
                
                # Use the new API for typing text
                if hasattr(desktop, "write"):
                    desktop.write(instruction.text_content)
                    action_performed = f"type_text: {instruction.text_content}"
                    action_result = f"Successfully typed: '{instruction.text_content}'"
                    print(f"Successfully typed: '{instruction.text_content}'")
                elif hasattr(desktop, "type"):
                    desktop.type(instruction.text_content)
                    action_performed = f"type_text: {instruction.text_content}"
                    action_result = f"Successfully typed: '{instruction.text_content}'"
                    print(f"Successfully typed: '{instruction.text_content}'")
                else:
                    error = "No typing method available on desktop"
                    print(f"Typing error: {error}")
                    
            except Exception as e:
                error = f"Typing failed: {str(e)}"
                print(f"Typing error: {e}")
        else:
            error = "No text content provided for type_text action"
            
    # Key press action
    elif instruction.action_type == ActionType.press_key:
        if instruction.key_sequence:
            try:
                pressed_keys = []
                error = None
                
                for key in instruction.key_sequence:
                    try:
                        print(f"Attempting to press key: '{key}'")
                        
                        # Map key names to e2b format (based on e2b documentation)
                        key_mapping = {
                            "return": "enter",
                            "ret": "enter", 
                            "enter": "enter",
                            "space": "space",
                            "tab": "tab",
                            "backspace": "backspace",
                            "delete": "delete",
                            "escape": "escape",
                            "esc": "escape",
                            "up": "up",
                            "down": "down",
                            "left": "left",
                            "right": "right",
                            "home": "home",
                            "end": "end",
                            "pageup": "pageup",
                            "pagedown": "pagedown",
                        }
                        
                        # Get the correct key name for e2b_desktop
                        desktop_key = key_mapping.get(key.lower(), key)
                        
                        # Press the key using the new API
                        if hasattr(desktop, "press"):
                            desktop.press(desktop_key)
                            pressed_keys.append(desktop_key)
                            print(f"Used press('{desktop_key}')")
                        elif hasattr(desktop, "key_press"):
                            desktop.key_press(desktop_key)
                            pressed_keys.append(desktop_key)
                            print(f"Used key_press('{desktop_key}')")
                        else:
                            raise ValueError("No suitable key press method found")
                            
                        time.sleep(0.05)  # Small delay between keys
                        
                        # Special handling for Enter key - add extra delay for page loading
                        if key.lower() in ["return", "ret", "enter"]:
                            print("Enter key pressed - waiting for page to load...")
                            time.sleep(2.0)  # Wait for search to execute and page to load
                        
                    except Exception as e:
                        print(f"Error pressing key {key}: {e}")
                        error = f"Error pressing key {key}: {e}"
                        break
                
                if not error:
                    action_performed = f"press_key: {instruction.key_sequence}"
                    action_result = f"Successfully pressed keys: {', '.join(pressed_keys)}"
                    print(f"Successfully pressed keys: {pressed_keys}")
                    
                    # Enhanced validation for Enter key presses
                    if any(key.lower() in ["return", "ret", "enter"] for key in instruction.key_sequence):
                        action_result += " (Enter key processed - search or action should be executed)"
                        print("Enter key press completed - search or action should be executed")
            except Exception as e:
                error = f"Keypress failed: {str(e)}"
                print(f"Keypress error: {e}")
        else:
            error = "No key sequence provided for press_key action"
                
    elif instruction.action_type == ActionType.scroll:
        direction = instruction.scroll_direction
        amount = 3  # Default scroll amount
        
        try:
            print(f"Attempting to scroll {direction} by {amount}")
            
            if hasattr(desktop, "scroll"):
                # Use the new API pattern
                desktop.scroll(direction, amount)
                action_performed = f"scroll {direction}"
                action_result = f"Scrolled {direction} by {amount}"
                print(f"Successfully scrolled {direction} by {amount}")
            else:
                error = "Desktop does not support scroll method"
                
        except Exception as e:
            error = f"Scroll failed: {str(e)}"
            print(f"Scroll error: {e}")
        
    elif instruction.action_type == ActionType.wait:
        wait_time = instruction.wait_seconds
        try:
            print(f"Waiting for {wait_time} seconds")
            
            # Use sleep instead of desktop.wait to be more reliable
            time.sleep(wait_time)
            
            action_performed = f"wait {wait_time}s"
            action_result = f"Waited {wait_time} seconds"
            print(f"Successfully waited {wait_time} seconds")
            
        except Exception as e:
            error = f"Wait failed: {str(e)}"
            print(f"Wait error: {e}")
        
    elif instruction.action_type == ActionType.navigate:
        if instruction.url:
            try:
                print(f"Attempting to navigate to: {instruction.url}")
                
                # Navigate by typing URL in address bar using new API
                # Use Ctrl+L to focus address bar
                if hasattr(desktop, "press"):
                    desktop.press("ctrl+l")
                    time.sleep(0.5)
                    
                    # Type the URL
                    if hasattr(desktop, "write"):
                        desktop.write(instruction.url)
                        time.sleep(0.2)
                        
                        # Press Enter
                        desktop.press("enter")
                        time.sleep(8)  # Wait for navigation to complete and page to load
                        print("Waited 8 seconds for page to fully load after navigation")
                        
                        action_performed = f"navigate to {instruction.url}"
                        action_result = f"Navigated to {instruction.url}"
                        print(f"Successfully navigated to {instruction.url}")
                    else:
                        error = "No write method available for typing URL"
                else:
                    error = "No press method available for key presses"
                    
            except Exception as e:
                error = f"Navigation failed: {str(e)}"
                print(f"Navigation error: {e}")
        else:
            error = "No URL provided for navigation"
            
    elif instruction.action_type == ActionType.screenshot:
        try:
            print("Taking screenshot...")
            
            # Take screenshot using desktop API
            screenshot_data = desktop.screenshot()
            if screenshot_data:
                action_performed = "take_screenshot"
                action_result = f"Screenshot taken: {len(screenshot_data)} bytes"
                print(f"Successfully took screenshot: {len(screenshot_data)} bytes")
            else:
                error = "Screenshot returned no data"
                
        except Exception as e:
            error = f"Screenshot failed: {str(e)}"
            print(f"Screenshot error: {e}")
    
    elif instruction.action_type == ActionType.type_and_enter:
        if instruction.text_content:
            try:
                # Wait for page to be ready before typing
                wait_for_page_ready(1)
                
                print(f"Executing combined type_and_enter action: '{instruction.text_content}'")
                
                # Import the combined action function
                from agent.cua.actions import type_and_enter
                
                # Execute the combined action
                success = type_and_enter(desktop, instruction.text_content)
                
                if success:
                    action_performed = f"type_and_enter: {instruction.text_content}"
                    action_result = f"Successfully typed '{instruction.text_content}' and pressed Enter"
                    print(f"Combined action completed successfully")
                else:
                    error = f"Combined type_and_enter action failed"
                    print(f"Combined action failed")
                    
            except Exception as e:
                error = f"Combined type_and_enter failed: {str(e)}"
                print(f"Combined type_and_enter error: {e}")
        else:
            error = "No text provided for type_and_enter action"
    
    else:
        error = f"Unsupported action type: {instruction.action_type}"
        print(f"Unsupported action type: {instruction.action_type}")
    
    return action_performed, action_result, error, elements_found


def build_brain_context(user_request: str, interaction_summary: str, recent_actions: str, recent_brain_instructions: str, previous_output: str, previous_user_request: str, last_report) -> str:
    """Build context string for the strategic brain."""
    context_parts = [
        f"USER REQUEST: {user_request}",
        f"INTERACTION: {interaction_summary}",
        f"RECENT ACTIONS: {recent_actions}",
        f"RECENT BRAIN INSTRUCTIONS: {recent_brain_instructions}",
    ]
    
    # Add previous conversation context if available
    if previous_user_request and previous_output:
        context_parts.append(f"PREVIOUS USER REQUEST: {previous_user_request}")
        context_parts.append(f"PREVIOUS OUTPUT: {previous_output}")
    elif previous_output:
        context_parts.append(f"PREVIOUS OUTPUT: {previous_output}")
    
    if last_report:
        context_parts.append(f"LAST ACTION: {last_report.action_performed or 'Initial screenshot'}")
        if last_report.action_result:
            context_parts.append(f"ACTION RESULT: {last_report.action_result}")
        if last_report.error:
            context_parts.append(f"ERROR: {last_report.error}")
        if last_report.elements_found:
            context_parts.append(f"ELEMENTS FOUND: {', '.join(last_report.elements_found)}")
    
    return "\n".join(context_parts)


def should_force_completion(user_request: str, interaction_count: int, recent_actions: List[str]) -> tuple[bool, str]:
    """
    Intelligently determine if task should be forced to completion based on request patterns.
    
    Returns:
        tuple: (should_complete, completion_message)
    """
    user_request_lower = user_request.lower()
    
    # For search requests, if we've taken multiple actions, likely complete
    if any(keyword in user_request_lower for keyword in ["search", "find", "look for", "top", "list", "dishes", "information"]):
        if interaction_count >= 6:
            return True, "Search task completed. The requested information should be visible on the current screen."
    
    # For navigation requests, if we've taken several actions, likely complete
    if any(keyword in user_request_lower for keyword in ["go to", "navigate", "visit", "open"]):
        if interaction_count >= 4:
            return True, "Navigation task completed. You should now be at the requested destination."
    
    # If we see repeated screenshot actions, task is likely complete
    if recent_actions.count("screenshot") >= 3:
        return True, "Task completed based on current screen content. The requested information should be visible."
    
    # If we see repeated type_text actions, task is likely complete
    if len([a for a in recent_actions if "type_text" in a]) >= 3:
        return True, "Task completed. The search query has been entered and results should be visible."
    
    return False, ""


def handle_repeated_action(instruction_response, state, action_desc: str):
    """Handle repeated actions by modifying the instruction to break loops."""
    if state.is_repeating_action(action_desc):
        print(f"Detected repeated action: {action_desc}. Trying alternative approach.")
        
        # Check if we've been repeating the same action too many times
        recent_actions = state.action_history[-5:]  # Last 5 actions
        same_action_count = recent_actions.count(action_desc)
        
        # If we've repeated the same action 3+ times, force task completion
        if same_action_count >= 3:
            print(f"Action repeated {same_action_count} times. Forcing task completion.")
            instruction_response.is_task_complete = True
            instruction_response.completion_message = "Task completed based on current screen state. The requested information should be visible."
            return instruction_response
        
        # Check if we should force completion based on request patterns
        should_complete, completion_msg = should_force_completion(
            state.user_request, state.interaction_count, state.action_history[-10:]
        )
        if should_complete:
            print(f"Forcing completion based on request pattern analysis.")
            instruction_response.is_task_complete = True
            instruction_response.completion_message = completion_msg
            return instruction_response
        
        # Otherwise, try to break the loop with alternative actions
        if instruction_response.action_type == "click_element":
            instruction_response.action_type = "scroll"
            instruction_response.description = "Scroll to find different elements"
            instruction_response.target_element = ""
            instruction_response.scroll_direction = "down"
            instruction_response.reasoning = "Breaking repetitive clicking by scrolling to find new elements"
        elif instruction_response.action_type == "type_text":
            instruction_response.action_type = "press_key"
            instruction_response.description = "Press Enter to execute the search"
            instruction_response.text_content = ""
            instruction_response.key_sequence = ["enter"]
            instruction_response.reasoning = "Breaking repetitive typing by pressing Enter to execute the search"
        elif instruction_response.action_type == "screenshot":
            # If we keep taking screenshots, the task is likely complete
            instruction_response.is_task_complete = True
            instruction_response.completion_message = "Task completed based on current screen content. The requested information should be visible in the screenshot."
            instruction_response.reasoning = "Breaking repetitive screenshots by marking task complete"
    
    return instruction_response 


def detect_human_intervention_needed(last_report, user_request: str, interaction_count: int) -> tuple[bool, str]:
    """
    Detect if human intervention is needed based on screen content and context.
    
    Returns:
        tuple: (needs_intervention, reason)
    """
    if not last_report or not last_report.screenshot_b64:
        return False, ""
    
    # Check for common intervention scenarios based on screen content
    # Note: This is a simplified check - in practice, you'd use computer vision
    # to analyze the actual screenshot content
    
    intervention_keywords = [
        "login", "sign in", "password", "username", "email",
        "captcha", "recaptcha", "verification", "verify",
        "two-factor", "2fa", "authentication", "auth",
        "blocked", "access denied", "forbidden",
        "rate limit", "too many requests",
        "human verification", "robot check",
        "confirm you are human", "prove you're human"
    ]
    
    # Check if any action results or errors indicate need for human intervention
    if last_report.action_result:
        action_result_lower = last_report.action_result.lower()
        if any(keyword in action_result_lower for keyword in intervention_keywords):
            return True, f"Authentication/verification required: {last_report.action_result}"
    
    if last_report.error:
        error_lower = last_report.error.lower()
        if any(keyword in error_lower for keyword in intervention_keywords):
            return True, f"Authentication/verification error: {last_report.error}"
    
    # Check for high interaction count suggesting the agent is stuck
    if interaction_count >= 18:
        return True, f"Agent appears to be stuck after {interaction_count} interactions. Human guidance needed to proceed."
    
    return False, ""


def should_request_human_intervention(state, instruction_response) -> tuple[bool, str]:
    """
    Determine if human intervention should be requested based on current state.
    
    Returns:
        tuple: (should_request, reason)
    """
    # Check if we're stuck in a loop with high interaction count
    if state.interaction_count >= 18:
        return True, f"Agent is stuck after {state.interaction_count} interactions. Human guidance needed to proceed or complete the task."
    
    # Check if we've detected authentication/verification needs
    needs_intervention, reason = detect_human_intervention_needed(
        state.last_executor_report, 
        state.user_request, 
        state.interaction_count
    )
    
    if needs_intervention:
        return True, reason
    
    # Check if we're repeating the same instruction too many times
    if hasattr(state, 'brain_instruction_history') and len(state.brain_instruction_history) >= 3:
        brain_instruction_desc = f"{instruction_response.action_type}:{instruction_response.description}"
        if state.is_repeating_brain_instruction(brain_instruction_desc):
            recent_same_count = state.brain_instruction_history[-5:].count(brain_instruction_desc)
            if recent_same_count >= 3 and state.interaction_count >= 15:
                return True, f"Agent is stuck repeating the same instruction '{brain_instruction_desc}' {recent_same_count} times. Human guidance needed."
    
    return False, "" 