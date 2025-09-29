"""Test script to demonstrate intelligent tool filtering with large tool sets."""

import asyncio
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from deepagent_mcp import create_mcp_orchestrator, Configuration

load_dotenv()


def _extract_final_response_text(result: Dict[str, Any]) -> str:
    """Extract final response text from result."""
    msgs = result.get("messages", []) or []
    ai_msgs = [
        m for m in msgs
        if getattr(m, "type", None) in ("ai", "assistant") or getattr(m, "role", None) in ("ai", "assistant")
    ]
    if ai_msgs:
        final = getattr(ai_msgs[-1], "content", "")
        return final if isinstance(final, str) else str(final)
    return ""


async def test_large_tool_set_filtering():
    """Test filtering with a realistic large tool set."""
    config = Configuration(
        model="gpt-4o",
        debug_mode=True,
        enable_advanced_filtering=True,
        max_tools_per_step=10,
        max_tools_before_filtering=20,  # Lower threshold for testing
        mcp_server_config={
            "gmail": {
                "transport": "streamable_http",
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/f88ef1e7-58e0-4358-a5fa-6a1c70e13b88/mcp?connected_account_id=ca_eitcLbt02t3K"
            },
            "youtube": {
                "transport": "streamable_http",
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/f63e2b4f-a989-4633-91b7-a58938f71ba8/mcp?connected_account_id=ca_n4N9uU407JGP"
            },
            "google drive": {
                "transport": "streamable_http", 
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/005d2151-9c7e-4730-b96b-f2a9eebc650b/mcp?connected_account_id=ca_-J7xmtf5qBC_"
            }
        },
        system_prompt={"rule": "Be helpful and efficient", "priority": "Focus on the most relevant tools for each task"}
    )

    orchestrator = await create_mcp_orchestrator(config)

    test_cases = [
        "Send an email to john@example.com about the meeting tomorrow",
        "Find videos about Python programming on YouTube",
        "List my recent Google Drive documents and share one with the team",
        "I need to organize my Gmail inbox and create some labels",
        "Search for specific files in my Google Drive containing 'budget'"
    ]

    for i, request in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {request}")
        print('='*60)

        result = await orchestrator.ainvoke(
            {"messages": [{"role": "user", "content": request}]},
            config={
                "configurable": {
                    "model": config.model,
                    "debug_mode": config.debug_mode,
                    "mcp_server_config": config.mcp_server_config,
                    "enable_advanced_filtering": config.enable_advanced_filtering,
                    "max_tools_per_step": config.max_tools_per_step,
                    "max_tools_before_filtering": config.max_tools_before_filtering,
                    "system_prompt": config.system_prompt,
                }
            },
        )

        final_text = _extract_final_response_text(result)
        print(f"[Response] {final_text[:200]}...")
        
        # Check if filtering stats are in the result
        if hasattr(result, 'filtering_stats'):
            stats = result.filtering_stats
            print(f"[Filtering Stats] {stats['original_count']} -> {stats['final_count']} tools")


async def test_filtering_performance():
    """Test filtering performance with different request types."""
    config = Configuration(
        model="gpt-4o",
        debug_mode=True,
        enable_advanced_filtering=True,
        max_tools_per_step=8,
        mcp_server_config={
            "gmail": {
                "transport": "streamable_http",
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/f88ef1e7-58e0-4358-a5fa-6a1c70e13b88/mcp?connected_account_id=ca_eitcLbt02t3K"
            }
        }
    )

    orchestrator = await create_mcp_orchestrator(config)

    # Test with a very specific request
    specific_request = "Fetch my 5 most recent unread Gmail messages from the inbox"
    
    print(f"\nTesting specific request: {specific_request}")
    
    result = await orchestrator.ainvoke(
        {"messages": [{"role": "user", "content": specific_request}]},
        config={"configurable": {
            "model": config.model,
            "debug_mode": config.debug_mode,
            "mcp_server_config": config.mcp_server_config,
            "enable_advanced_filtering": config.enable_advanced_filtering,
            "max_tools_per_step": config.max_tools_per_step,
            "system_prompt": config.system_prompt,
        }}
    )

    final_text = _extract_final_response_text(result)
    print(f"[Specific Request Result] {final_text}")


if __name__ == "__main__":
    print("Testing Intelligent Tool Filtering...")
    asyncio.run(test_large_tool_set_filtering())
    asyncio.run(test_filtering_performance())
