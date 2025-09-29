"""Test script to verify MCP configuration format with Gmail and YouTube configs."""

import asyncio
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from deepagent_mcp import create_mcp_orchestrator, Configuration

load_dotenv()


def _extract_tool_calls(messages: List[Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for msg in messages or []:
        additional = getattr(msg, "additional_kwargs", {}) or {}
        tc = additional.get("tool_calls")
        if tc and isinstance(tc, list):
            for c in tc:
                fn = (c.get("function") or {}) if isinstance(c, dict) else {}
                name = fn.get("name") or c.get("name") or c.get("tool_name") or "unknown_tool"
                args = fn.get("arguments") or c.get("args") or {}
                calls.append({"name": name, "args": args})
        direct_calls = getattr(msg, "tool_calls", None)
        if direct_calls and isinstance(direct_calls, list):
            for c in direct_calls:
                if isinstance(c, dict):
                    name = c.get("name") or c.get("tool_name") or "unknown_tool"
                    args = c.get("args") or {}
                else:
                    name = getattr(c, "name", "unknown_tool")
                    args = getattr(c, "args", {})
                calls.append({"name": name, "args": args})
        if msg.__class__.__name__ in ("ToolMessage", "ToolResultMessage"):
            name = getattr(msg, "tool", None) or getattr(msg, "name", None) or "tool"
            calls.append({"name": str(name), "result": getattr(msg, "content", "")})
    return calls


def _extract_final_response_text(result: Dict[str, Any]) -> str:
    msgs = result.get("messages", []) or []
    ai_msgs = [
        m for m in msgs
        if getattr(m, "type", None) in ("ai", "assistant") or getattr(m, "role", None) in ("ai", "assistant")
    ]
    if ai_msgs:
        final = getattr(ai_msgs[-1], "content", "")
        return final if isinstance(final, str) else str(final)
    return ""


def _build_test_config() -> Configuration:
    """Build test configuration with Gmail and YouTube MCP servers."""
    return Configuration(
        model="gpt-4o",
        debug_mode=False,  # Enable debug to see what's happening
        enable_advanced_filtering=False,
        max_tools_per_step=50,
        mcp_server_config={
            "gmail": {
                "transport": "streamable_http",
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/f88ef1e7-58e0-4358-a5fa-6a1c70e13b88/mcp?connected_account_id=ca_eitcLbt02t3K"
            },
            "youtube": {
                "transport": "streamable_http",
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/f63e2b4f-a989-4633-91b7-a58938f71ba8/mcp?connected_account_id=ca_n4N9uU407JGP"
            }
        },
        system_prompt={"rule": "Always be helpful and polite", "step-1": "First, understand the user's request", "step-2": "Then execute the appropriate tools"}
    )


async def test_mcp_tool_discovery():
    """Test MCP tool discovery with Gmail and YouTube configs."""
    config = _build_test_config()
    orchestrator = await create_mcp_orchestrator(config)

    request = "List all available tools from Gmail and YouTube. Show me what tools are discovered and their capabilities."

    result = await orchestrator.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={
            "configurable": {
                "model": config.model,
                "debug_mode": config.debug_mode,
                "mcp_server_config": config.mcp_server_config,
                "enable_advanced_filtering": config.enable_advanced_filtering,
                "max_tools_per_step": config.max_tools_per_step,
                "system_prompt": config.system_prompt,
            }
        },
    )

    final_text = _extract_final_response_text(result)
    print(f"[MCP Tool Discovery] {final_text}")
    return final_text


async def test_gmail_functionality():
    """Test Gmail functionality."""
    config = _build_test_config()
    orchestrator = await create_mcp_orchestrator(config)

    request = "Check my Gmail inbox and tell me about any recent emails. What Gmail tools are available?"

    result = await orchestrator.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={
            "configurable": {
                "model": config.model,
                "debug_mode": config.debug_mode,
                "mcp_server_config": config.mcp_server_config,
                "enable_advanced_filtering": config.enable_advanced_filtering,
                "max_tools_per_step": config.max_tools_per_step,
                "system_prompt": config.system_prompt,
            }
        },
    )

    final_text = _extract_final_response_text(result)
    print(f"[Gmail Functionality] {final_text}")
    return final_text


async def test_youtube_functionality():
    """Test YouTube functionality."""
    config = _build_test_config()
    orchestrator = await create_mcp_orchestrator(config)

    request = "What YouTube tools are available? Can you search for videos or channels?"

    result = await orchestrator.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={
            "configurable": {
                "model": config.model,
                "debug_mode": config.debug_mode,
                "mcp_server_config": config.mcp_server_config,
                "enable_advanced_filtering": config.enable_advanced_filtering,
                "max_tools_per_step": config.max_tools_per_step,
                "system_prompt": config.system_prompt,
            }
        },
    )

    final_text = _extract_final_response_text(result)
    print(f"[YouTube Functionality] {final_text}")
    return final_text


async def test_google_drive_key_with_space():
    """Repro: single-server config key contains a space (e.g., 'google drive'). Should still bind tools."""
    from deepagent_mcp import create_mcp_orchestrator, Configuration
    cfg = Configuration(
        model="gpt-4o",
        debug_mode=True,
        enable_advanced_filtering=False,
        max_tools_per_step=50,
        mcp_server_config={
            "google drive": {
                "transport": "streamable_http",
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/005d2151-9c7e-4730-b96b-f2a9eebc650b/mcp?connected_account_id=ca_-J7xmtf5qBC_"
            }
        },
        system_prompt={"rule": "Always be helpful"}
    )
    orch = await create_mcp_orchestrator(cfg)
    req = "List my recent Google Drive files."
    res = await orch.ainvoke(
        {"messages": [{"role": "user", "content": req}]},
        config={"configurable": {
            "model": cfg.model,
            "debug_mode": cfg.debug_mode,
            "mcp_server_config": cfg.mcp_server_config,
            "enable_advanced_filtering": cfg.enable_advanced_filtering,
            "max_tools_per_step": cfg.max_tools_per_step,
            "system_prompt": cfg.system_prompt,
        }}
    )
    final = _extract_final_response_text(res)
    print(f"[Google Drive Key With Space] {final}")
    return final


async def test_google_drive_key_no_space():
    """Repro: single-server config key has no space (e.g., 'googledrive')."""
    from deepagent_mcp import create_mcp_orchestrator, Configuration
    cfg = Configuration(
        model="gpt-4o",
        debug_mode=True,
        enable_advanced_filtering=False,
        max_tools_per_step=50,
        mcp_server_config={
            "googledrive": {
                "transport": "streamable_http",
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/005d2151-9c7e-4730-b96b-f2a9eebc650b/mcp?connected_account_id=ca_-J7xmtf5qBC_"
            }
        },
        system_prompt={"rule": "Always be helpful"}
    )
    orch = await create_mcp_orchestrator(cfg)
    req = "List my recent Google Drive files."
    res = await orch.ainvoke(
        {"messages": [{"role": "user", "content": req}]},
        config={"configurable": {
            "model": cfg.model,
            "debug_mode": cfg.debug_mode,
            "mcp_server_config": cfg.mcp_server_config,
            "enable_advanced_filtering": cfg.enable_advanced_filtering,
            "max_tools_per_step": cfg.max_tools_per_step,
            "system_prompt": cfg.system_prompt,
        }}
    )
    final = _extract_final_response_text(res)
    print(f"[Google Drive Key No Space] {final}")
    return final


async def test_google_drive_key_mixed_case():
    """Repro: single-server config key mixed case (e.g., 'Google Drive')."""
    from deepagent_mcp import create_mcp_orchestrator, Configuration
    cfg = Configuration(
        model="gpt-4o",
        debug_mode=True,
        enable_advanced_filtering=False,
        max_tools_per_step=50,
        mcp_server_config={
            "Google Drive": {
                "transport": "streamable_http",
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/005d2151-9c7e-4730-b96b-f2a9eebc650b/mcp?connected_account_id=ca_-J7xmtf5qBC_"
            }
        },
        system_prompt={"rule": "Always be helpful"}
    )
    orch = await create_mcp_orchestrator(cfg)
    req = "Search my Google Drive for 'proposal'."
    res = await orch.ainvoke(
        {"messages": [{"role": "user", "content": req}]},
        config={"configurable": {
            "model": cfg.model,
            "debug_mode": cfg.debug_mode,
            "mcp_server_config": cfg.mcp_server_config,
            "enable_advanced_filtering": cfg.enable_advanced_filtering,
            "max_tools_per_step": cfg.max_tools_per_step,
            "system_prompt": cfg.system_prompt,
        }}
    )
    final = _extract_final_response_text(res)
    print(f"[Google Drive Key Mixed Case] {final}")
    return final


async def test_google_drive_with_filtering():
    """Repro: enable advanced filtering to ensure tools still selected for Drive."""
    from deepagent_mcp import create_mcp_orchestrator, Configuration
    cfg = Configuration(
        model="gpt-4o",
        debug_mode=True,
        enable_advanced_filtering=True,
        max_tools_per_step=50,
        mcp_server_config={
            "google drive": {
                "transport": "streamable_http",
                "url": "https://apollo-9ej90deaw-composio.vercel.app/v3/mcp/005d2151-9c7e-4730-b96b-f2a9eebc650b/mcp?connected_account_id=ca_-J7xmtf5qBC_"
            }
        },
        system_prompt={"rule": "Always be helpful"}
    )
    orch = await create_mcp_orchestrator(cfg)
    req = "List my 5 most recent Google Drive docs."
    res = await orch.ainvoke(
        {"messages": [{"role": "user", "content": req}]},
        config={"configurable": {
            "model": cfg.model,
            "debug_mode": cfg.debug_mode,
            "mcp_server_config": cfg.mcp_server_config,
            "enable_advanced_filtering": cfg.enable_advanced_filtering,
            "max_tools_per_step": cfg.max_tools_per_step,
            "system_prompt": cfg.system_prompt,
        }}
    )
    final = _extract_final_response_text(res)
    print(f"[Google Drive With Filtering] {final}")
    return final


async def test_config_validation():
    """Test that the configuration is properly loaded."""
    config = _build_test_config()
    # Validate configuration silently
    _ = config.validate_model()
    return config


async def main():
    # Optionally validate configuration silently
    await test_config_validation()

    # Only print final responses for each test
    await test_mcp_tool_discovery()
    await test_gmail_functionality()
    await test_youtube_functionality()
    await test_google_drive_key_with_space()
    await test_google_drive_key_no_space()
    await test_google_drive_key_mixed_case()
    await test_google_drive_with_filtering()


if __name__ == "__main__":
    asyncio.run(main())