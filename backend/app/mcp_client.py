"""Model Context Protocol (MCP) Client Session and Telemetry.

Provides structured execution of tools hosted on the FastMCP server, formatted
with standard MCP JSON-RPC 2.0 envelopes and complete protocol telemetry.
"""

import asyncio, json, time, uuid, datetime, sys
from pathlib import Path
from typing import Any
from .config import settings
from .schemas import McpToolInvocation
from .guardrails import validate_mcp_query, safe_external_text

_MCP_TELEMETRY_LOG: list[McpToolInvocation] = []

def get_recent_mcp_invocations() -> list[McpToolInvocation]:
    """Return historical record of MCP tool invocations."""
    return list(_MCP_TELEMETRY_LOG)

def _get_mcp_tools():
    backend_dir = str(Path(__file__).resolve().parents[1])
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import mcp_server
    return {
        "text_statistics": mcp_server.text_statistics,
        "search_arxiv": mcp_server.search_arxiv,
        "search_openalex": mcp_server.search_openalex,
        "search_crossref": mcp_server.search_crossref,
        "search_academic_literature": mcp_server.search_academic_literature,
    }

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
    if "query" in sanitized_args:
        clean_q, b_res, _ = validate_mcp_query(str(sanitized_args["query"]), int(sanitized_args.get("max_results", 8)))
        sanitized_args["query"] = clean_q
        sanitized_args["max_results"] = b_res

    # Build MCP JSON-RPC 2.0 Request Payload
    request_jsonrpc = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": sanitized_args
        }
    }

    raw_result = None
    status = "SUCCESS"
    err_msg = ""

    # Call the tool from mcp_server
    try:
        tool_map = _get_mcp_tools()
        if name in tool_map:
            fn = tool_map[name]
            raw_result = await asyncio.to_thread(fn, **sanitized_args)
        else:
            status = "FAILED"
            raw_result = {"error": f"Unknown tool: '{name}'"}
    except Exception as exc:
        status = "FAILED"
        err_msg = f"{type(exc).__name__}: {str(exc)}"
        raw_result = {"error": err_msg}

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Build MCP JSON-RPC 2.0 Response Payload
    response_jsonrpc = {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": safe_external_text(json.dumps(raw_result))
                }
            ],
            "isError": status == "FAILED"
        }
    }

    # Generate human-readable summary
    if name == "text_statistics":
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
        protocol="Model Context Protocol (JSON-RPC 2.0)",
        transport="stdio / FastMCP Process",
        request_jsonrpc=request_jsonrpc,
        response_summary=summary,
        response_jsonrpc=response_jsonrpc,
        duration_ms=duration_ms,
        status=status
    )

    _MCP_TELEMETRY_LOG.append(invocation)
    return raw_result, invocation
