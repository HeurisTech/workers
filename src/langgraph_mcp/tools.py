"""Tools for the enhanced planner style agent.

This module contains all tools-related functionality including:
- Custom tool framework and abstract base class
- Tool registry for managing custom tools
- Tool merger for combining MCP and custom tools
- Actual tool implementations
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
from langgraph_mcp.supabase.retriever import retrieve_organizational_knowledge


# ============================================================================
# Custom Tool Framework
# ============================================================================

class CustomToolFunction(ABC):
    """Abstract base class for custom tool functions.
    
    Custom tools implement this interface to provide functionality
    that can be used by the planner style agent alongside MCP tools.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """A description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """The parameters schema for the tool (JSON Schema format)."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool with the given parameters.
        
        Args:
            **kwargs: Tool parameters as keyword arguments
            
        Returns:
            str: The result of the tool execution
            
        Raises:
            Exception: If the tool execution fails
        """
        pass
    
    def to_langchain_tool(self) -> Dict[str, Any]:
        """Convert the custom tool to LangChain tool format.
        
        Returns:
            Dict[str, Any]: Tool in LangChain format
        """
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
        }


# ============================================================================
# Tool Registry
# ============================================================================

# Global tools registry
_tools: Dict[str, CustomToolFunction] = {}


def register_tool(tool: CustomToolFunction) -> None:
    """Register a custom tool.
    
    Args:
        tool: The custom tool to register
        
    Raises:
        ValueError: If a tool with the same name already exists
    """
    if tool.name in _tools:
        raise ValueError(f"Tool with name '{tool.name}' already exists")
    _tools[tool.name] = tool


def get_tool(name: str) -> Optional[CustomToolFunction]:
    """Get a custom tool by name.
    
    Args:
        name: The name of the tool to retrieve
        
    Returns:
        Optional[CustomToolFunction]: The tool if found, None otherwise
    """
    return _tools.get(name)


def get_all_tools() -> List[CustomToolFunction]:
    """Get all registered custom tools.
    
    Returns:
        List[CustomToolFunction]: All registered tools
    """
    return list(_tools.values())


# ============================================================================
# Tool Merger Functions
# ============================================================================

async def get_merged_tools(
    server_name: str, 
    server_config: Dict[str, Any], 
    include_custom_tools: bool = True
) -> List[Dict[str, Any]]:
    """Get merged tools from both MCP server and custom tools registry.
    
    Args:
        server_name: Name of the MCP server
        server_config: Configuration for the MCP server
        include_custom_tools: Whether to include custom tools in the result
        
    Returns:
        List[Dict[str, Any]]: Combined list of tools in LangChain format
    """
    tools = []
    
    # Try to get MCP tools if this is a valid MCP server configuration
    # MCP servers require a 'transport' field (stdio, streamable_http, etc.)
    if server_config and 'transport' in server_config:
        try:
            from langgraph_mcp.mcp_wrapper import apply, GetTools
            mcp_tools = await apply(server_name, server_config, GetTools())
            tools.extend(mcp_tools)
            print(f"[MCP] Successfully loaded {len(mcp_tools)} MCP tools from server '{server_name}'")
        except Exception as e:
            print(f"[MCP] Warning: Failed to get MCP tools from {server_name}: {e}")
    else:
        # This expert doesn't have MCP server configuration
        # It might be a conceptual expert that only uses custom tools
        print(f"[EXPERT] Expert '{server_name}' has no MCP transport configuration - will use custom tools only")
    
    # Always include custom tools (they're part of our system)
    if include_custom_tools:
        custom_tools = get_all_tools()
        for tool in custom_tools:
            tools.append(tool.to_langchain_tool())
        print(f"[CUSTOM] Successfully loaded {len(custom_tools)} custom tools: {[t.name for t in custom_tools]}")
    
    print(f"[TOOLS] Total tools available to expert '{server_name}': {len(tools)} tools")
    return tools


async def execute_merged_tool(
    tool_name: str, 
    tool_args: Dict[str, Any], 
    server_name: str = None, 
    server_config: Dict[str, Any] = None
) -> str:
    """Execute a tool (either MCP or custom).
    
    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool
        server_name: Name of the MCP server (for MCP tools)
        server_config: Configuration for the MCP server (for MCP tools)
        
    Returns:
        str: Result of the tool execution
        
    Raises:
        Exception: If tool execution fails
    """
    # First check if it's a custom tool
    custom_tool = get_tool(tool_name)
    if custom_tool:
        print(f"[CUSTOM TOOL] Executing custom tool '{tool_name}' with args: {tool_args}")
        result = await custom_tool.execute(**tool_args)
        print(f"[CUSTOM TOOL] Tool '{tool_name}' completed. Result length: {len(str(result))} chars")
        return result
    
    # If not custom, try MCP tool (only if we have valid MCP config)
    if server_name and server_config and 'transport' in server_config:
        from langgraph_mcp.mcp_wrapper import apply, RunTool
        try:
            print(f"[MCP TOOL] Executing MCP tool '{tool_name}' via server '{server_name}' with args: {tool_args}")
            result = await apply(server_name, server_config, RunTool(tool_name, **tool_args))
            print(f"[MCP TOOL] Tool '{tool_name}' completed successfully via server '{server_name}'")
            return result
        except Exception as e:
            print(f"[MCP TOOL] ERROR: Tool '{tool_name}' failed via server '{server_name}': {str(e)}")
            raise Exception(f"MCP tool execution failed: {str(e)}")
    
    # Tool not found in either registry
    available_custom = [t.name for t in get_all_tools()]
    error_msg = f"Tool '{tool_name}' not found in custom or MCP registries. Available custom tools: {available_custom}"
    print(f"[TOOL ERROR] {error_msg}")
    raise Exception(error_msg)


def is_custom_tool(tool_name: str) -> bool:
    """Check if a tool is a custom tool.
    
    Args:
        tool_name: Name of the tool to check
        
    Returns:
        bool: True if it's a custom tool, False otherwise
    """
    return get_tool(tool_name) is not None


# ============================================================================
# Tool Implementations
# ============================================================================

class GrepSearchTool(CustomToolFunction):
    """Custom tool for searching text patterns in files using regex."""
    
    @property
    def name(self) -> str:
        return "grep_search"
    
    @property
    def description(self) -> str:
        return "Search for text patterns in files using regex. Supports searching in single files or directories."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "file_path": {
                    "type": "string",
                    "description": "The file or directory path to search in"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case sensitive",
                    "default": False
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 50
                },
                "include_line_numbers": {
                    "type": "boolean",
                    "description": "Whether to include line numbers in results",
                    "default": True
                }
            },
            "required": ["pattern", "file_path"]
        }
    
    async def execute(self, **kwargs) -> str:
        pattern = kwargs.get("pattern", "")
        file_path = kwargs.get("file_path", "")
        case_sensitive = kwargs.get("case_sensitive", False)
        max_results = kwargs.get("max_results", 50)
        include_line_numbers = kwargs.get("include_line_numbers", True)
        
        if not pattern:
            return "Error: Pattern is required"
        
        if not file_path:
            return "Error: File path is required"
        
        try:
            # Compile regex pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
            
            results = []
            
            if os.path.isfile(file_path):
                # Search in single file
                results.extend(self._search_in_file(file_path, regex, include_line_numbers))
            elif os.path.isdir(file_path):
                # Search in directory
                for root, dirs, files in os.walk(file_path):
                    for file in files:
                        if file.endswith(('.py', '.txt', '.md', '.json', '.yaml', '.yml', '.js', '.ts')):
                            full_path = os.path.join(root, file)
                            matches = self._search_in_file(full_path, regex, include_line_numbers)
                            results.extend(matches)
            else:
                return f"Error: Path '{file_path}' does not exist"
            
            # Limit results
            if len(results) > max_results:
                results = results[:max_results]
                results.append(f"... (showing first {max_results} results)")
            
            if not results:
                return f"No matches found for pattern '{pattern}' in '{file_path}'"
            
            return "\n".join(results)
            
        except re.error as e:
            return f"Error: Invalid regex pattern - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _search_in_file(self, file_path: str, regex: re.Pattern, include_line_numbers: bool) -> List[str]:
        """Search for pattern in a single file."""
        results = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        line = line.rstrip('\n\r')
                        if include_line_numbers:
                            results.append(f"{file_path}:{line_num}: {line}")
                        else:
                            results.append(f"{file_path}: {line}")
        except Exception as e:
            results.append(f"Error reading {file_path}: {str(e)}")
        
        return results

class KnowledgeRetrievalTool(CustomToolFunction):
    """Custom tool for retrieving organizational knowledge."""

    @property
    def name(self) -> str:
        return "retrieve_organizational_knowledge"

    @property
    def description(self) -> str:
        return (
            "Retrieves documents from the knowledge base for a specific organization. "
            "Use this to find relevant information, documentation, or examples "
            "before trying to write code or answer questions."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's query or question to search for in the knowledge base."
                },
                "organization_id": {
                    "type": "string",
                    "description": "The ID of the organization. If not provided, the system will try to use the ID from the configuration."
                },
                "search_type": {
                    "type": "string",
                    "description": "The type of search to perform. Can be 'similarity', 'text', or 'hybrid'. Defaults to 'text'.",
                    "default": "text"
                }
            },
            "required": ["query"]
        }

    async def execute(self, **kwargs) -> str:
        query = kwargs.get("query")
        organization_id = kwargs.get("organization_id")
        search_type = kwargs.get("search_type", "text")

        print(f"[DEBUG] KnowledgeRetrievalTool executed with: query='{query}', organization_id='{organization_id}', search_type='{search_type}'")
        print(f"[DEBUG] All kwargs: {kwargs}")

        if not query:
            return "Error: 'query' is a required parameter."

        if not organization_id:
            print(f"[DEBUG] organization_id is missing or empty: '{organization_id}'")
            return "Error: 'organization_id' is missing. The system failed to provide it from configuration."

        try:
            print(f"[DEBUG] Calling retrieve_organizational_knowledge with organization_id: '{organization_id}'")
            results = retrieve_organizational_knowledge(
                query=query,
                organization_id=organization_id,
                search_type=search_type
            )
            if not results:
                return "No results found for your query."
            return json.dumps(results, indent=2)
        except Exception as e:
            print(f"[DEBUG] Error in retrieve_organizational_knowledge: {e}")
            return f"An error occurred while calling the retrieval API: {e}"

class FileReadTool(CustomToolFunction):
    """Custom tool for reading file contents."""
    
    @property
    def name(self) -> str:
        return "file_read"
    
    @property
    def description(self) -> str:
        return "Read the contents of a file. Supports text files and provides options for partial reading."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "The line number to start reading from (1-based)",
                    "default": 1
                },
                "end_line": {
                    "type": "integer",
                    "description": "The line number to stop reading at (1-based, inclusive)",
                    "default": -1
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of characters to read",
                    "default": -1
                },
                "encoding": {
                    "type": "string",
                    "description": "The file encoding to use",
                    "default": "utf-8"
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file_path", "")
        start_line = kwargs.get("start_line", 1)
        end_line = kwargs.get("end_line", -1)
        max_chars = kwargs.get("max_chars", -1)
        encoding = kwargs.get("encoding", "utf-8")
        
        if not file_path:
            return "Error: File path is required"
        
        if not os.path.isfile(file_path):
            return f"Error: File '{file_path}' does not exist or is not a file"
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                if start_line == 1 and end_line == -1:
                    # Read entire file
                    content = f.read()
                else:
                    # Read specific lines
                    lines = f.readlines()
                    start_idx = max(0, start_line - 1)
                    end_idx = len(lines) if end_line == -1 else min(len(lines), end_line)
                    content = ''.join(lines[start_idx:end_idx])
                
                # Limit by character count if specified
                if max_chars > 0 and len(content) > max_chars:
                    content = content[:max_chars] + "\n... (truncated)"
                
                if not content:
                    return f"File '{file_path}' is empty or the specified range is empty"
                
                return content
                
        except UnicodeDecodeError:
            return f"Error: Unable to decode file '{file_path}' with encoding '{encoding}'"
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"


class FileListTool(CustomToolFunction):
    """Custom tool for listing files and directories."""
    
    @property
    def name(self) -> str:
        return "file_list"
    
    @property
    def description(self) -> str:
        return "List files and directories in a given path. Supports filtering and detailed information."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "The directory path to list"
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g., '*.py')",
                    "default": "*"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files (starting with .)",
                    "default": False
                },
                "detailed": {
                    "type": "boolean",
                    "description": "Whether to include detailed information (size, modified time)",
                    "default": False
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Maximum number of entries to return",
                    "default": 100
                }
            },
            "required": ["directory_path"]
        }
    
    async def execute(self, **kwargs) -> str:
        directory_path = kwargs.get("directory_path", "")
        pattern = kwargs.get("pattern", "*")
        include_hidden = kwargs.get("include_hidden", False)
        detailed = kwargs.get("detailed", False)
        max_entries = kwargs.get("max_entries", 100)
        
        if not directory_path:
            return "Error: Directory path is required"
        
        if not os.path.isdir(directory_path):
            return f"Error: Directory '{directory_path}' does not exist or is not a directory"
        
        try:
            entries = []
            
            for entry in os.listdir(directory_path):
                if not include_hidden and entry.startswith('.'):
                    continue
                
                full_path = os.path.join(directory_path, entry)
                
                if detailed:
                    stat = os.stat(full_path)
                    size = stat.st_size
                    modified = os.path.getmtime(full_path)
                    import datetime
                    mod_time = datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')
                    
                    if os.path.isdir(full_path):
                        entry_info = f"[DIR]  {entry:30} {size:>10} bytes  {mod_time}"
                    else:
                        entry_info = f"[FILE] {entry:30} {size:>10} bytes  {mod_time}"
                    
                    entries.append(entry_info)
                else:
                    if os.path.isdir(full_path):
                        entries.append(f"[DIR]  {entry}")
                    else:
                        entries.append(f"[FILE] {entry}")
            
            # Sort entries
            entries.sort()
            
            # Limit entries
            if len(entries) > max_entries:
                entries = entries[:max_entries]
                entries.append(f"... (showing first {max_entries} entries)")
            
            if not entries:
                return f"Directory '{directory_path}' is empty"
            
            return "\n".join(entries)
            
        except Exception as e:
            return f"Error listing directory '{directory_path}': {str(e)}"


class ShellCommandTool(CustomToolFunction):
    """Custom tool for executing shell commands locally."""
    
    @property
    def name(self) -> str:
        return "shell_command"
    
    @property
    def description(self) -> str:
        return "Execute shell commands on the local system. Supports various shell operations including file manipulation, system commands, and application execution."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_directory": {
                    "type": "string",
                    "description": "The working directory to execute the command in",
                    "default": "."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for command execution (default: 30)",
                    "default": 30
                },
                "capture_output": {
                    "type": "boolean",
                    "description": "Whether to capture and return the command output",
                    "default": True
                },
                "shell": {
                    "type": "boolean",
                    "description": "Whether to run the command through the shell",
                    "default": True
                }
            },
            "required": ["command"]
        }
    
    async def execute(self, **kwargs) -> str:
        import subprocess
        import asyncio
        
        command = kwargs.get("command", "")
        working_directory = kwargs.get("working_directory", ".")
        timeout = kwargs.get("timeout", 30)
        capture_output = kwargs.get("capture_output", True)
        shell = kwargs.get("shell", True)
        
        if not command:
            return "Error: Command is required"
        
        # Security check - prevent potentially dangerous commands
        dangerous_patterns = [
            'rm -rf /',
            'del /s /q',
            'format c:',
            'dd if=',
            'mkfs',
            'fdisk',
            'shutdown',
            'reboot',
            'halt',
            'poweroff'
        ]
        
        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return f"Error: Potentially dangerous command blocked: {command}"
        
        try:
            print(f"[SHELL] Preparing to execute command: '{command}'")
            print(f"[SHELL] Working directory: '{working_directory}', timeout: {timeout}s, capture_output: {capture_output}")
            
            # Convert working directory to absolute path if it exists
            # Use async-safe path operations to avoid blocking calls
            if working_directory != "." and not os.path.exists(working_directory):
                error_msg = f"Error: Working directory '{working_directory}' does not exist"
                print(f"[SHELL ERROR] {error_msg}")
                return error_msg
            
            # Convert to absolute path without using os.getcwd() (which can block)
            if working_directory == ".":
                # Use the current working directory passed from environment
                working_directory = None  # Let subprocess use its default
                print(f"[SHELL] Using default working directory (current directory)")
            else:
                working_directory = os.path.abspath(working_directory)
                print(f"[SHELL] Using absolute working directory: '{working_directory}'")
            
            # Execute command
            if capture_output:
                print(f"[SHELL] Starting subprocess with output capture...")
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=working_directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=shell
                )
                
                try:
                    print(f"[SHELL] Waiting for command completion (timeout: {timeout}s)...")
                    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
                    output = stdout.decode('utf-8', errors='ignore').strip()
                    return_code = process.returncode
                    
                    print(f"[SHELL] Command completed with return code: {return_code}")
                    print(f"[SHELL] Output length: {len(output)} characters")
                    
                    if return_code == 0:
                        result = output if output else "Command executed successfully (no output)"
                        print(f"[SHELL SUCCESS] {result}")
                        return result
                    else:
                        result = f"Command failed with exit code {return_code}:\n{output}"
                        print(f"[SHELL ERROR] {result}")
                        return result
                        
                except asyncio.TimeoutError:
                    print(f"[SHELL TIMEOUT] Command exceeded timeout of {timeout} seconds, killing process...")
                    process.kill()
                    await process.wait()
                    error_msg = f"Error: Command timed out after {timeout} seconds"
                    print(f"[SHELL ERROR] {error_msg}")
                    return error_msg
                    
            else:
                # Execute without capturing output
                print(f"[SHELL] Starting subprocess without output capture...")
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=working_directory,
                    shell=shell
                )
                
                try:
                    print(f"[SHELL] Waiting for command completion (timeout: {timeout}s)...")
                    return_code = await asyncio.wait_for(process.wait(), timeout=timeout)
                    print(f"[SHELL] Command completed with return code: {return_code}")
                    
                    if return_code == 0:
                        result = "Command executed successfully"
                        print(f"[SHELL SUCCESS] {result}")
                        return result
                    else:
                        result = f"Command failed with exit code {return_code}"
                        print(f"[SHELL ERROR] {result}")
                        return result
                        
                except asyncio.TimeoutError:
                    print(f"[SHELL TIMEOUT] Command exceeded timeout of {timeout} seconds, killing process...")
                    process.kill()
                    await process.wait()
                    error_msg = f"Error: Command timed out after {timeout} seconds"
                    print(f"[SHELL ERROR] {error_msg}")
                    return error_msg
                    
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            print(f"[SHELL EXCEPTION] {error_msg}")
            return error_msg


# ============================================================================
# Tool Registration
# ============================================================================

def register_tools():
    """Register all tools."""
    register_tool(KnowledgeRetrievalTool())
    register_tool(GrepSearchTool())
    register_tool(FileReadTool())
    register_tool(FileListTool())
    register_tool(ShellCommandTool()) 