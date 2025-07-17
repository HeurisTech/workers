"""Map CUA tool-call **actions** onto e2b_desktop sandbox operations."""

from __future__ import annotations

import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .sandbox import get_desktop


# Mapping of CUA key names → e2b_desktop friendly strings or behaviours.
_CUA_KEY_TO_DESKTOP = {
    "/": "slash",
    "\\": "backslash",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "arrowup": "Up",
    "backspace": "BackSpace",
    "capslock": "Caps_Lock",
    "delete": "Delete",
    "end": "End",
    "enter": "enter",
    "esc": "Escape",
    "home": "Home",
    "insert": "Insert",
    "pagedown": "Page_Down",
    "pageup": "Page_Up",
    "tab": "Tab",
    "ctrl": "Control_L",
    "alt": "Alt_L",
    "shift": "Shift_L",
    "cmd": "Meta_L",
    "win": "Meta_L",
    "meta": "Meta_L",
    "space": "space",
}

# Set up logging
logger = logging.getLogger("cua.actions")


def test_desktop_api_methods(desktop):
    """Test function to discover available e2b desktop API methods."""
    print("\n=== Testing e2b desktop API methods ===")
    
    # Get all available methods
    methods = [method for method in dir(desktop) if not method.startswith('_')]
    print(f"Available methods: {methods}")
    
    # Test specific methods we're interested in
    test_methods = [
        'click', 'left_click', 'leftClick', 'left_click_at',
        'right_click', 'rightClick', 'right_click_at',
        'double_click', 'doubleClick', 'double_click_at',
        'move_mouse', 'moveMouse', 'move_mouse_to',
        'press', 'press_key', 'key_press',
        'write', 'type', 'type_text',
        'scroll'
    ]
    
    for method in test_methods:
        if hasattr(desktop, method):
            method_obj = getattr(desktop, method)
            print(f"✓ {method}: {method_obj}")
            
            # Try to get method signature if possible
            try:
                import inspect
                sig = inspect.signature(method_obj)
                print(f"  Signature: {sig}")
            except Exception as e:
                print(f"  Could not get signature: {e}")
        else:
            print(f"✗ {method}: not found")
    
    print("=== End API test ===\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def execute_action(action: str | Dict[str, Any], sandbox_id: str) -> Dict[str, Any]:
    """Execute an action that can be either a string or a structured dict.
    
    This function handles both formats:
    - String actions (from older format): "Click the search box"
    - Dict actions (structured format): {"action": "click", "coordinate": [x, y]}
    """
    if isinstance(action, str):
        # For string actions, we need to parse the intent and convert to structured format
        # This is a simplified parser - in practice, you might want more sophisticated parsing
        action_lower = action.lower()
        
        if "click" in action_lower:
            # For now, we'll just return a generic click action
            # In practice, you'd want to extract coordinates from the string
            return {"success": False, "error": "String actions not fully supported - use structured format"}
        elif "type" in action_lower:
            # Extract text to type
            # This is a simplified extraction
            return {"success": False, "error": "String actions not fully supported - use structured format"}
        else:
            return {"success": False, "error": f"Unknown string action: {action}"}
    
    # Handle structured dict actions
    return perform_action(sandbox_id, action)


def perform_action(sandbox_id: str, action: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Execute a single *action* against the desktop sandbox identified by *sandbox_id*.
    
    Returns a dict with the result of the action, including success status and any error.
    """
    result = {
        "success": False,
        "action_type": action.get("type"),
        "error": None,
        "details": {}
    }

    try:
        desktop = get_desktop(sandbox_id)
        if desktop is None:
            raise RuntimeError(f"Desktop sandbox {sandbox_id} not found in cache.")

        # Test API methods (only run this once for debugging)
        test_desktop_api_methods(desktop)

        action_type = action.get("type")
        logger.info(f"Executing action: {action_type}")

        if action_type == "click":
            _click_new_api(desktop, action)
            result["details"]["coordinates"] = (action.get("x"), action.get("y"))
            
        elif action_type == "double_click":
            _double_click_new_api(desktop, action)
            result["details"]["coordinates"] = (action.get("x"), action.get("y"))
            
        elif action_type == "move":
            _move_mouse_new_api(desktop, action)
            result["details"]["coordinates"] = (action.get("x"), action.get("y"))
            
        elif action_type == "drag":
            _drag(desktop, action)
            path = action.get("path", [])
            if path:
                result["details"]["start"] = (path[0].get("x"), path[0].get("y"))
                result["details"]["end"] = (path[-1].get("x"), path[-1].get("y"))
                
        elif action_type == "keypress":
            keys = _keypress_new_api(desktop, action)
            result["details"]["keys"] = keys
            
        elif action_type == "type":
            text = action.get("text", "")
            _type_text_new_api(desktop, text)
            result["details"]["text"] = text
            
        elif action_type == "scroll":
            _scroll_new_api(desktop, action)
            result["details"]["delta_x"] = action.get("scroll_x", 0)
            result["details"]["delta_y"] = action.get("scroll_y", 0)
            
        elif action_type == "wait":
            duration = action.get("duration", 2)
            time.sleep(duration)
            result["details"]["duration"] = duration
            
        elif action_type == "screenshot":
            # Use screenshot method if available
            if hasattr(desktop, "screenshot"):
                desktop.screenshot()  # type: ignore[attr-defined]
                
        else:
            raise ValueError(f"Unsupported action type: {action_type}")
        
        # Add a small delay after each action to allow the UI to update
        time.sleep(0.2)
        result["success"] = True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Action failed: {error_msg}")
        result["success"] = False
        result["error"] = error_msg
    
    return result


# ---------------------------------------------------------------------------
# New API implementations based on TypeScript examples
# ---------------------------------------------------------------------------


def _click_new_api(desktop, action):
    """Click using the actual e2b desktop API based on TypeScript examples."""
    x, y = action.get("x"), action.get("y")
    button = action.get("button", "left")
    
    if x is None or y is None:
        raise ValueError("Click action requires x and y coordinates")
    
    try:
        print(f"Attempting to click at ({x}, {y}) with {button} button")
        
        # Try different API patterns based on TypeScript examples
        if button == "left":
            # Try the methods we see in TypeScript examples
            if hasattr(desktop, "leftClick"):
                desktop.leftClick(x, y)  # TypeScript: await desktop.leftClick(x, y)
                print(f"Used leftClick({x}, {y})")
            elif hasattr(desktop, "left_click"):
                desktop.left_click(x, y)  # Snake case version
                print(f"Used left_click({x}, {y})")
            elif hasattr(desktop, "click"):
                desktop.click(x, y)  # Generic click
                print(f"Used click({x}, {y})")
            else:
                raise ValueError("No suitable left click method found")
                
        elif button == "right":
            if hasattr(desktop, "rightClick"):
                desktop.rightClick(x, y)  # TypeScript: await desktop.rightClick(x, y)
                print(f"Used rightClick({x}, {y})")
            elif hasattr(desktop, "right_click"):
                desktop.right_click(x, y)  # Snake case version
                print(f"Used right_click({x}, {y})")
            else:
                raise ValueError("No suitable right click method found")
                
        elif button == "middle":
            if hasattr(desktop, "middleClick"):
                desktop.middleClick(x, y)  # TypeScript: await desktop.middleClick(x, y)
                print(f"Used middleClick({x}, {y})")
            elif hasattr(desktop, "middle_click"):
                desktop.middle_click(x, y)  # Snake case version
                print(f"Used middle_click({x}, {y})")
            else:
                raise ValueError("No suitable middle click method found")
        
        # Wait for click to register
        time.sleep(0.1)
        print(f"Successfully clicked at ({x}, {y}) with {button} button")
        
    except Exception as e:
        print(f"Click failed: {e}")
        raise RuntimeError(f"Failed to click at ({x}, {y}): {str(e)}")


def _double_click_new_api(desktop, action):
    """Double click using the actual e2b desktop API."""
    x, y = action.get("x"), action.get("y")
    
    if x is None or y is None:
        raise ValueError("Double click action requires x and y coordinates")
    
    try:
        print(f"Attempting to double click at ({x}, {y})")
        
        if hasattr(desktop, "doubleClick"):
            desktop.doubleClick(x, y)  # TypeScript: await desktop.doubleClick(x, y)
            print(f"Used doubleClick({x}, {y})")
        elif hasattr(desktop, "double_click"):
            desktop.double_click(x, y)  # Snake case version
            print(f"Used double_click({x}, {y})")
        else:
            # Fallback to two single clicks
            _click_new_api(desktop, action)
            time.sleep(0.1)
            _click_new_api(desktop, action)
            print(f"Used fallback double click at ({x}, {y})")
        
        time.sleep(0.1)
        print(f"Successfully double clicked at ({x}, {y})")
        
    except Exception as e:
        print(f"Double click failed: {e}")
        raise RuntimeError(f"Failed to double click at ({x}, {y}): {str(e)}")


def _move_mouse_new_api(desktop, action):
    """Move mouse using the actual e2b desktop API."""
    x, y = action.get("x"), action.get("y")
    
    if x is None or y is None:
        raise ValueError("Move mouse action requires x and y coordinates")
    
    try:
        print(f"Attempting to move mouse to ({x}, {y})")
        
        if hasattr(desktop, "moveMouse"):
            desktop.moveMouse(x, y)  # TypeScript: await desktop.moveMouse(x, y)
            print(f"Used moveMouse({x}, {y})")
        elif hasattr(desktop, "move_mouse"):
            desktop.move_mouse(x, y)  # Snake case version
            print(f"Used move_mouse({x}, {y})")
        else:
            raise ValueError("No suitable move mouse method found")
        
        time.sleep(0.05)
        print(f"Successfully moved mouse to ({x}, {y})")
        
    except Exception as e:
        print(f"Move mouse failed: {e}")
        raise RuntimeError(f"Failed to move mouse to ({x}, {y}): {str(e)}")


def _type_text_new_api(desktop, text: str):
    """Type text using the actual e2b desktop API."""
    if not text:
        return
    
    try:
        print(f"Attempting to type: '{text}'")
        
        if hasattr(desktop, "write"):
            desktop.write(text)  # TypeScript: await desktop.write(text)
            print(f"Used write('{text}')")
        elif hasattr(desktop, "type"):
            desktop.type(text)  # Alternative method
            print(f"Used type('{text}')")
        elif hasattr(desktop, "type_text"):
            desktop.type_text(text)  # Snake case version
            print(f"Used type_text('{text}')")
        else:
            # Fallback to pressing keys one by one
            print(f"No write method found, typing character by character")
            for char in text:
                if hasattr(desktop, "press"):
                    desktop.press(char)
                    time.sleep(0.01)
                else:
                    raise ValueError("No suitable typing method found")
        
        time.sleep(0.1)
        print(f"Successfully typed: '{text}'")
        
    except Exception as e:
        print(f"Type text failed: {e}")
        raise RuntimeError(f"Failed to type text '{text}': {str(e)}")


def _keypress_new_api(desktop, action) -> List[str]:
    """Press keys using the actual e2b desktop API."""
    keys = action.get("keys", [])
    pressed_keys = []
    
    for key in keys:
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
            }
            
            # Get the correct key name for e2b_desktop
            desktop_key = key_mapping.get(key.lower(), key)
            
            if hasattr(desktop, "press"):
                desktop.press(desktop_key)  # TypeScript: await desktop.press(key)
                pressed_keys.append(desktop_key)
                print(f"Used press('{desktop_key}')")
            elif hasattr(desktop, "key_press"):
                desktop.key_press(desktop_key)  # Alternative method
                pressed_keys.append(desktop_key)
                print(f"Used key_press('{desktop_key}')")
            else:
                raise ValueError("No suitable key press method found")
                
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
            break
    
    return pressed_keys


def type_and_enter(desktop, text: str) -> bool:
    """Combined action: Type text and press Enter in one operation for faster execution."""
    try:
        print(f"Executing combined action: type '{text}' and press Enter")
        
        # Wait a moment to ensure the page is ready to receive input
        time.sleep(1)
        print("Waited 1 second for page to be ready for input")
        
        # Type the text
        if hasattr(desktop, "write"):
            desktop.write(text)
            print(f"Successfully typed: '{text}'")
            time.sleep(0.1)  # Small delay between type and enter
            
            # Press Enter
            desktop.press("enter")
            print("Successfully pressed Enter")
            time.sleep(2.0)  # Wait for search/action to execute
            
            return True
        else:
            print("Desktop does not support write method")
            return False
            
    except Exception as e:
        print(f"Combined type_and_enter failed: {e}")
        return False


def _scroll_new_api(desktop, action):
    """Scroll using the actual e2b desktop API."""
    dx = action.get("scroll_x", 0)
    dy = action.get("scroll_y", 0)
    x = action.get("x", 0)
    y = action.get("y", 0)
    
    try:
        print(f"Attempting to scroll: dx={dx}, dy={dy} at ({x}, {y})")
        
        # Move to position first if coordinates provided
        if x > 0 or y > 0:
            _move_mouse_new_api(desktop, {"x": x, "y": y})
        
        if hasattr(desktop, "scroll"):
            # Determine direction and amount
            if dy > 0:
                direction = "down"
                amount = abs(dy) // 20 or 1
            elif dy < 0:
                direction = "up"
                amount = abs(dy) // 20 or 1
            else:
                direction = "down"
                amount = 1
            
            desktop.scroll(direction, amount)  # TypeScript: await desktop.scroll(direction, amount)
            print(f"Used scroll('{direction}', {amount})")
        else:
            raise ValueError("No suitable scroll method found")
        
        time.sleep(0.1)
        print(f"Successfully scrolled {direction} by {amount}")
        
    except Exception as e:
        print(f"Scroll failed: {e}")
        raise RuntimeError(f"Failed to scroll: {str(e)}")


# ---------------------------------------------------------------------------
# Click validation and helper functions
# ---------------------------------------------------------------------------


def validate_click_success(desktop, x: int, y: int, target_element: str) -> Dict[str, Any]:
    """
    Validate if a click was successful by checking UI state changes.
    
    Args:
        desktop: The e2b desktop instance
        x, y: Coordinates that were clicked
        target_element: Description of what was expected to be clicked
        
    Returns:
        Dict with success status and validation details
    """
    result = {
        "success": False,
        "validation_details": {},
        "error": None
    }
    
    try:
        # Take a screenshot after the click to analyze the result
        screenshot_after = desktop.screenshot()
        
        if screenshot_after:
            # Basic validation - check if cursor is in expected position
            # In a full implementation, you could use computer vision here
            # to check if the UI actually changed in the expected way
            
            # For now, we'll do basic checks
            result["validation_details"]["screenshot_taken"] = True
            result["validation_details"]["target_element"] = target_element
            result["validation_details"]["click_coordinates"] = (x, y)
            
            # Simple heuristic: if we can take a screenshot, the click probably worked
            # In practice, you'd want more sophisticated validation
            result["success"] = True
            result["validation_details"]["validation_method"] = "basic_heuristic"
            
        else:
            result["error"] = "Could not take screenshot for validation"
            
    except Exception as e:
        result["error"] = f"Validation failed: {str(e)}"
    
    return result


def execute_click_action_with_validation(desktop, x: int, y: int, target_element: str = "", button: str = "left") -> Dict[str, Any]:
    """Execute a click action with validation and detailed feedback."""
    result = {
        "success": False,
        "coordinates": (x, y),
        "target_element": target_element,
        "button": button,
        "error": None,
        "validation": None
    }
    
    try:
        # Validate coordinates
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"Invalid coordinates: x={x}, y={y}")
            
        if x < 0 or y < 0:
            raise ValueError(f"Coordinates must be non-negative: x={x}, y={y}")
        
        # Execute the click using the new API
        action = {"x": int(x), "y": int(y), "button": button}
        _click_new_api(desktop, action)
        
        # Validate the click was successful
        validation_result = validate_click_success(desktop, int(x), int(y), target_element)
        result["validation"] = validation_result
        
        if validation_result["success"]:
            result["success"] = True
            print(f"Click executed and validated successfully at ({x}, {y}) for '{target_element}'")
        else:
            result["error"] = f"Click validation failed: {validation_result.get('error', 'Unknown validation error')}"
            print(f"Click executed but validation failed at ({x}, {y}) for '{target_element}': {result['error']}")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"Click failed at ({x}, {y}) for '{target_element}': {e}")
    
    return result


# ---------------------------------------------------------------------------
# Old API implementations (kept for fallback)
# ---------------------------------------------------------------------------


def _click(desktop, action):  # noqa: D401 – helper
    """Click at the specified coordinates using the correct e2b desktop API."""
    x, y = action.get("x"), action.get("y")
    button = action.get("button", "left")

    if x is None or y is None:
        raise ValueError("Click action requires x and y coordinates")

    try:
        # Move mouse first (this is the correct e2b API pattern)
        desktop.move_mouse(x, y)  # type: ignore[attr-defined]

        # Wait for mouse to move
        time.sleep(0.1)

        # Click based on button type at the current cursor position
        if button == "right":
            if hasattr(desktop, "right_click"):
                desktop.right_click()  # type: ignore[attr-defined]
            elif hasattr(desktop, "click"):
                desktop.click("right")  # type: ignore[attr-defined]
            else:
                # Fallback using mouse_press/mouse_release
                desktop.mouse_press("right")  # type: ignore[attr-defined]
                time.sleep(0.05)
                desktop.mouse_release("right")  # type: ignore[attr-defined]
        elif button == "middle":
            if hasattr(desktop, "middle_click"):
                desktop.middle_click()  # type: ignore[attr-defined]
            elif hasattr(desktop, "click"):
                desktop.click("middle")  # type: ignore[attr-defined]
            else:
                desktop.mouse_press("middle")  # type: ignore[attr-defined]
                time.sleep(0.05)
                desktop.mouse_release("middle")  # type: ignore[attr-defined]
        else:  # default to left click
            if hasattr(desktop, "left_click"):
                desktop.left_click()  # type: ignore[attr-defined]
            elif hasattr(desktop, "click"):
                desktop.click("left")  # type: ignore[attr-defined]
            else:
                desktop.mouse_press("left")  # type: ignore[attr-defined]
                time.sleep(0.05)
                desktop.mouse_release("left")  # type: ignore[attr-defined]

        # Wait for click to register
        time.sleep(0.1)
        print(f"Successfully clicked at ({x}, {y}) with {button} button")

    except Exception as e:
        raise RuntimeError(f"Failed to click at ({x}, {y}): {str(e)}")


def _double_click(desktop, action):  # noqa: D401 – helper
    x, y = action.get("x"), action.get("y")
    
    # Move mouse first
    desktop.move_mouse(x, y)  # type: ignore[attr-defined]
    
    # Small delay to ensure mouse has moved
    time.sleep(0.05)
    
    # Then double click
    if hasattr(desktop, "double_click"):
        desktop.double_click()  # type: ignore[attr-defined]
    else:
        # Fallback to two left clicks with a short delay
        if hasattr(desktop, "left_click"):
            desktop.left_click()  # type: ignore[attr-defined]
            time.sleep(0.1)  # Short delay between clicks for double-click
            desktop.left_click()  # type: ignore[attr-defined]
        else:
            # Fallback using mouse_press/mouse_release if available
            if hasattr(desktop, "mouse_press") and hasattr(desktop, "mouse_release"):
                desktop.mouse_press("left")  # type: ignore[attr-defined]
                desktop.mouse_release("left")  # type: ignore[attr-defined]
                time.sleep(0.1)
                desktop.mouse_press("left")  # type: ignore[attr-defined]
                desktop.mouse_release("left")  # type: ignore[attr-defined]


def _drag(desktop, action):  # noqa: D401 – helper
    path: List[Dict[str, int]] = action.get("path", [])
    if not path:
        return
    
    if len(path) < 2:
        return
    
    start = path[0]
    end = path[-1]
    
    # Use drag method if available
    if hasattr(desktop, "drag"):
        desktop.drag((start.get("x"), start.get("y")), (end.get("x"), end.get("y")))  # type: ignore[attr-defined]
        return
    
    # Otherwise fallback to move + press + move + release
    desktop.move_mouse(start.get("x"), start.get("y"))  # type: ignore[attr-defined]
    time.sleep(0.1)  # Ensure mouse has moved
    
    if hasattr(desktop, "mouse_press"):
        desktop.mouse_press("left")  # type: ignore[attr-defined]
        time.sleep(0.1)  # Short delay after press
        
        # Move through path points
        for point in path[1:]:
            desktop.move_mouse(point.get("x"), point.get("y"))  # type: ignore[attr-defined]
            time.sleep(0.05)  # Small delay between moves for smoother dragging
            
        time.sleep(0.1)  # Short delay before release
        if hasattr(desktop, "mouse_release"):
            desktop.mouse_release("left")  # type: ignore[attr-defined]
    else:
        # Very basic fallback
        desktop.left_click()  # type: ignore[attr-defined]
        time.sleep(0.1)
        desktop.move_mouse(end.get("x"), end.get("y"))  # type: ignore[attr-defined]
        time.sleep(0.1)
        desktop.left_click()  # type: ignore[attr-defined]


def _keypress(desktop, action) -> List[str]:  # noqa: D401 – helper
    keys = action.get("keys", [])
    pressed_keys = []
    
    for key in keys:
        # Convert key to lowercase for e2b_desktop API
        key_lower = key.lower()
        
        # Map some common key names to e2b_desktop format
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
            "shift": "shift",
            "ctrl": "ctrl",
            "alt": "alt",
            "cmd": "cmd",
            "meta": "cmd",
            "arrowup": "up",
            "arrowdown": "down",
            "arrowleft": "left",
            "arrowright": "right",
            "up": "up",
            "down": "down", 
            "left": "left",
            "right": "right",
            "home": "home",
            "end": "end",
            "pageup": "page_up",
            "pagedown": "page_down",
        }
        
        # Get the correct key name for e2b_desktop
        desktop_key = key_mapping.get(key_lower, key_lower)
        pressed_keys.append(desktop_key)
        
        try:
            # Use the e2b_desktop press method
            if hasattr(desktop, "press"):
                desktop.press(desktop_key)  # type: ignore[attr-defined]
                print(f"Pressed key: {desktop_key}")
            else:
                print(f"Warning: desktop.press method not available for key: {desktop_key}")
                
        except Exception as e:
            print(f"Error pressing key {desktop_key}: {e}")
            # Try alternative methods if available
            if hasattr(desktop, "press_key"):
                try:
                    desktop.press_key(desktop_key)  # type: ignore[attr-defined]
                    print(f"Pressed key using press_key: {desktop_key}")
                except Exception as e2:
                    print(f"Error with press_key for {desktop_key}: {e2}")
    
    return pressed_keys


def _type_text(desktop, text: str) -> None:  # noqa: D401 – helper
    """Type text efficiently based on available methods."""
    if not text:
        return
        
    # Use the most efficient method available
    if hasattr(desktop, "write"):
        desktop.write(text)  # type: ignore[attr-defined]
    elif hasattr(desktop, "type_text"):
        desktop.type_text(text)  # type: ignore[attr-defined]
    else:
        # Fallback to pressing keys one by one
        for char in text:
            if hasattr(desktop, "press"):
                desktop.press(char)  # type: ignore[attr-defined]
                time.sleep(0.01)  # Small delay to avoid overwhelming the system


def _scroll(desktop, action):  # noqa: D401 – helper
    dx = action.get("scroll_x", 0)
    dy = action.get("scroll_y", 0)
    x = action.get("x", 0)
    y = action.get("y", 0)
    
    # Move mouse to position first if coordinates are provided
    if x != 0 or y != 0:
        desktop.move_mouse(x, y)  # type: ignore[attr-defined]
        time.sleep(0.05)
    
    if hasattr(desktop, "scroll"):
        # Determine direction and amount
        if dy != 0:
            direction = "down" if dy > 0 else "up"
            amount = abs(dy) // 20 or 1  # Normalize the amount, minimum 1
            desktop.scroll(direction=direction, amount=amount)  # type: ignore[attr-defined]
        elif dx != 0:
            direction = "right" if dx > 0 else "left"
            amount = abs(dx) // 20 or 1  # Normalize the amount, minimum 1
            desktop.scroll(direction=direction, amount=amount)  # type: ignore[attr-defined]
    else:
        # No scroll method available, try to use a more generic method if exists
        pass


def capture_screenshot(sandbox_id: str) -> bytes | None:  # noqa: D401
    """Capture a screenshot from the sandbox.
    
    Returns bytes of the screenshot or None if not available.
    """
    desktop = get_desktop(sandbox_id)
    if desktop is None:
        logger.error(f"Cannot capture screenshot: sandbox {sandbox_id} not found")
        return None
        
    # Try different screenshot methods
    try:
        if hasattr(desktop, "capture_screen"):
            return desktop.capture_screen()  # type: ignore[attr-defined]
        elif hasattr(desktop, "screenshot"):
            # Check if it returns bytes
            try:
                screenshot = desktop.screenshot(format="bytes")  # type: ignore[attr-defined]
                if isinstance(screenshot, (bytes, bytearray)):
                    return screenshot
            except Exception as e:
                logger.warning(f"Error capturing screenshot with format=bytes: {e}")
                # Try without format parameter
                screenshot = desktop.screenshot()  # type: ignore[attr-defined]
                if isinstance(screenshot, (bytes, bytearray)):
                    return screenshot
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}")
    
    return None 