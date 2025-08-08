import asyncio
import os
from langchain_core.messages import HumanMessage

from . import compile_with_memory, deep_researcher


async def main():
    db_uri = os.getenv("DATABASE_URI")
    # Build graph with or without persistence
    if db_uri:
        ctx = compile_with_memory(db_uri)
        async with ctx as graph:
            await stream_once(graph)
    else:
        await stream_once(deep_researcher)


async def stream_once(graph):
    # Minimal demo prompt
    user_input = os.getenv("PROMPT", "Research latest LangGraph persistence options and summarize in 5 bullets.")

    # Use thread id to show persistence
    thread_id = os.getenv("THREAD_ID", "demo-thread-1")
    config = {"configurable": {"thread_id": thread_id}}

    # Stream values to stdout
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values",
    ):
        # Print any message tokens as they are produced
        try:
            msg = chunk.get("messages", [])[-1]
            content = getattr(msg, "content", None)
            if content:
                print(str(content))
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())

