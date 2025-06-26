import os
import pytest
import subprocess
import time
import requests
from langgraph_sdk import get_sync_client


# Configuration for MCP servers used in the tests
MCP_SERVER_CONFIG = {
    "github": {
        "transport": "streamable_http",
        "url": "https://server.smithery.ai/@smithery-ai/github/mcp",
        "env": {"githubPersonalAccessToken": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")},
        "description": "Access the GitHub API for file operations, repository management, and search.",
    },
    "smithery": {
        "transport": "streamable_http",
        "url": "https://server.smithery.ai/@smithery/toolbox/mcp",
        "env": {"smitheryApiKey": os.getenv("SMITHERY_API_KEY")},
        "description": "Smithery toolbox for dynamic routing to MCPs in the Smithery registry.",
    },
}

@pytest.fixture(scope="module")
def langgraph_server():
    """Starts and stops the langgraph dev server for the test module."""
    process = subprocess.Popen(
        ["langgraph", "dev"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    server_url = "http://127.0.0.1:2024"
    for _ in range(30):  # Wait up to 30 seconds for the server to start
        try:
            if requests.get(f"{server_url}/docs", timeout=1).status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        stdout, stderr = process.communicate()
        pytest.fail(f"Server failed to start.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    yield server_url

    process.terminate()
    process.wait()

def test_github_repository_search(langgraph_server):
    """Tests searching for a GitHub repository."""

    client = get_sync_client(url=langgraph_server)
    
    # Create an assistant with the specified mcp_server_config
    assistant = client.assistants.create(
        graph_id="old_planner_agent",
        config={"configurable": {"mcp_server_config": MCP_SERVER_CONFIG}},
    )
    
    # Create a new thread
    thread = client.threads.create()
    
    # Stream the run and collect events
    events = client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": 'Can you search for the repository: "pdhoolia/langgraph-se-agent"?'}]},
        stream_mode="updates"
    )

    plan_made = False
    tool_selected = None
    tool_output = None

    for event in events:
        if event.event != "updates":
            continue
        
        if planner_data := event.data.get("planner"):
            if planner_result := planner_data.get("planner_result"):
                if plan := planner_result.get("plan"):
                    if len(plan) == 1 and plan[0]["expert"] == "github":
                        plan_made = True
        elif orchestrate_tools := event.data.get("orchestrate_tools"):
            if messages := orchestrate_tools.get("messages"):
                for msg in messages:
                    if msg["type"] == "ai":
                        if tool_calls := msg.get("tool_calls"):
                            tool_selected = tool_calls[0]["name"]
        elif call_tool := event.data.get("call_tool"):
            if messages := call_tool.get("messages"):
                for msg in messages:
                    if msg["type"] == "tool":
                        tool_output = msg.get("content")

    assert plan_made, "The agent did not create the correct plan."
    assert tool_selected == "search_repositories", "The agent did not select the 'search_repositories' tool."
    assert tool_output and 'pdhoolia/langgraph-se-agent' in tool_output
    assert tool_output and 'Software & quality agent built as a set of langgraphs' in tool_output

def test_smithery_server_search(langgraph_server):
    """Tests searching for a server on Smithery."""
    client = get_sync_client(url=langgraph_server)
    
    # Create an assistant with the specified mcp_server_config
    assistant = client.assistants.create(
        graph_id="old_planner_agent",
        config={"configurable": {"mcp_server_config": MCP_SERVER_CONFIG}},
    )
    
    # Create a new thread
    thread = client.threads.create()
    
    # Stream the run and collect events
    events = client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": 'Can you search for server "@smithery-ai/github" on smithery?'}]},
        stream_mode="updates"
    )

    plan_made = False
    tool_selected = None
    tool_output = None

    for event in events:
        if event.event != "updates":
            continue
        
        if planner_data := event.data.get("planner"):
            if planner_result := planner_data.get("planner_result"):
                if plan := planner_result.get("plan"):
                    if len(plan) == 1 and plan[0]["expert"] == "smithery":
                        plan_made = True
        elif orchestrate_tools := event.data.get("orchestrate_tools"):
            if messages := orchestrate_tools.get("messages"):
                for msg in messages:
                    if msg["type"] == "ai":
                        if tool_calls := msg.get("tool_calls"):
                            tool_selected = tool_calls[0]["name"]
        elif call_tool := event.data.get("call_tool"):
            if messages := call_tool.get("messages"):
                for msg in messages:
                    if msg["type"] == "tool":
                        tool_output = msg.get("content")

    assert plan_made, "The agent did not create the correct plan."
    assert tool_selected == "search_servers", "The agent did not select the 'search_servers' tool."
    assert tool_output and '@smithery-ai/github' in tool_output 