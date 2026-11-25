"""Model Context Protocol (MCP) Client Session and Telemetry.

Provides structured execution of tools hosted on the FastMCP server, formatted
with standard MCP JSON-RPC 2.0 envelopes and complete protocol telemetry.
"""

import asyncio, json, time, uuid, datetime, sys
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .config import BACKEND_DIR, settings
from .schemas import McpToolInvocation
from .guardrails import validate_mcp_query, validate_tool_return_types

_MCP_TELEMETRY_LOG: list[McpToolInvocation] = []

def get_recent_mcp_invocations() -> list[McpToolInvocation]:
    """Return historical record of MCP tool invocations."""
    return list(_MCP_TELEMETRY_LOG)

async def call_mcp_tool(
    name: str,
    arguments: dict[str, Any],
    calling_agent: str = "Workflow Orchestrator"
) -> tuple[Any, McpToolInvocation]:
    """Execute an MCP tool via Model Context Protocol JSON-RPC 2.0 invocation."""
    t0 = time.perf_counter()
    req_id = f"mcp-req-{str(uuid.uuid4())[:8]}"
    
    # Layer 2 Guardrail: Sanitize arguments
    sanitized_args = dict(arguments)

    raw_result = None
    status = "SUCCESS"
    response_payload = {}
    protocol_version = None

    # Call the tool from mcp_server
    try:
        if not settings.mcp_enabled:
            raise ValueError("MCP is disabled")
        if name == "text_statistics" and len(sanitized_args.get("paper_text", "")) > 500000:
            raise ValueError("Statistics input exceeds 500,000 characters")
        if "query" in sanitized_args:
            clean_q, limit, _ = validate_mcp_query(str(sanitized_args["query"]), sanitized_args.get("max_results", 8))
            sanitized_args.update(query=clean_q, max_results=limit)
        server = StdioServerParameters(command=sys.executable, args=[str(BACKEND_DIR / "mcp_server.py")])
        async with asyncio.timeout(90):
            async with stdio_client(server) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    initialized = await session.initialize()
                    protocol_version = initialized.protocolVersion
                    result = await session.call_tool(name, sanitized_args)
        response_payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        if result.isError:
            raise ValueError("MCP server returned a tool error")
        raw_result = result.structuredContent
        if raw_result is None and result.content and result.content[0].type == "text":
            raw_result = json.loads(result.content[0].text)
        valid, message = validate_tool_return_types(name, raw_result)
        if not valid:
            raise ValueError(message)
    except Exception as exc:
        status = "FAILED"
        raw_result = {"error": f"{type(exc).__name__}: {exc}"}

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    # The report uses its own correlation ID for the SDK request and typed result.
    audit_arguments = dict(sanitized_args)
    if isinstance(audit_arguments.get("paper_text"), str):
        audit_arguments["paper_text"] = f"[{len(audit_arguments['paper_text'])} characters]"
    request_jsonrpc = {
        "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
        "params": {"name": name, "arguments": audit_arguments},
    }
    response_jsonrpc = {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": response_payload or {"isError": True}
    }

    # Generate human-readable summary
    if status == "FAILED":
        summary = raw_result["error"]
    elif name == "text_statistics":
        wc = (raw_result or {}).get("word_count", 0)
        sc = (raw_result or {}).get("sentence_count", 0)
        summary = f"Parsed paper structure: {wc:,} words, {sc:,} sentences."
    elif "search" in name:
        cnt = (raw_result or {}).get("count", len((raw_result or {}).get("papers", [])))
        mode = (raw_result or {}).get("mode", "federated_mcp")
        summary = f"Retrieved {cnt} publications ({mode}) for query '{sanitized_args.get('query','')[:40]}'."
    else:
        summary = f"Executed {name} in {duration_ms} ms."

    invocation = McpToolInvocation(
        call_id=req_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        agent_name=calling_agent,
        tool_name=name,
        server_name="paperlens-research-tools",
        protocol=f"MCP {protocol_version}" if protocol_version else "Model Context Protocol (JSON-RPC 2.0)",
        transport="stdio / FastMCP Process",
        request_jsonrpc=request_jsonrpc,
        response_summary=summary,
        response_jsonrpc=response_jsonrpc,
        duration_ms=duration_ms,
        status=status
    )

    _MCP_TELEMETRY_LOG.append(invocation)
    del _MCP_TELEMETRY_LOG[:-50]
    return raw_result, invocation
