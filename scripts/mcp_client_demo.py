"""Minimal MCP client for LabBot.

This demonstrates that LabBot capabilities can be discovered and called
without going through the FastAPI chat interface or LangGraph.

Run from the repository root:

    python -m scripts.mcp_client_demo

Optional arguments:

    python -m scripts.mcp_client_demo --user u1 --query oscilloscope
    python -m scripts.mcp_client_demo --user u2 --query sensor
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and call LabBot MCP tools."
    )

    parser.add_argument(
        "--user",
        default="u1",
        help="Demo LabBot actor ID. Defaults to u1.",
    )

    parser.add_argument(
        "--query",
        default="oscilloscope",
        help="Equipment query for the availability tool.",
    )

    return parser.parse_args()


def result_to_jsonable(result) -> object:
    """Convert an MCP CallToolResult into displayable JSON-like content."""
    structured = getattr(result, "structuredContent", None)

    if structured is not None:
        return structured

    content = getattr(result, "content", [])

    return [
        {
            "type": getattr(part, "type", "unknown"),
            "text": getattr(part, "text", str(part)),
        }
        for part in content
    ]


async def main() -> None:
    args = parse_args()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            discovered = await session.list_tools()

            print("\nDiscovered LabBot MCP tools:\n")

            for tool in discovered.tools:
                print(f"- {tool.name}")
                print(f"  {tool.description}\n")

            availability = await session.call_tool(
                "check_equipment_availability",
                arguments={"query": args.query},
            )

            print("Availability result:\n")
            print(
                json.dumps(
                    result_to_jsonable(availability),
                    indent=2,
                    default=str,
                )
            )

            status = await session.call_tool(
                "get_my_checkout_status",
                arguments={"actor_user_id": args.user},
            )

            print("\nPersonal checkout-status result:\n")
            print(
                json.dumps(
                    result_to_jsonable(status),
                    indent=2,
                    default=str,
                )
            )


if __name__ == "__main__":
    asyncio.run(main())