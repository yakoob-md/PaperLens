"""Launch the local MCP server over stdio and invoke tools."""
import sys
from pathlib import Path
from .config import settings

async def call_mcp_tool(name: str, arguments: dict):
    if not settings.mcp_enabled: return None
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        script = Path(__file__).resolve().parents[1] / "mcp_server.py"
        params = StdioServerParameters(command=sys.executable, args=[str(script)], env=None)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return "\\n".join(x.text for x in (result.content or []) if getattr(x, "text", None))
    except Exception as exc:
        return {"error": f"MCP unavailable: {type(exc).__name__}"}
