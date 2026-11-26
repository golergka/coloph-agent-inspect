"""Convert observed Codex JSON events into Inspect transcript events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import cast

from inspect_ai.event import InfoEvent, ToolEvent
from inspect_ai.model import ContentImage, ContentText
from inspect_ai.tool import ToolCallError
from pydantic import JsonValue


def _mcp_result(encoded: JsonValue) -> list[ContentText | ContentImage]:
    content = encoded.get("content") if isinstance(encoded, dict) else None
    parts: list[ContentText | ContentImage] = []
    if not isinstance(content, list):
        return parts
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(ContentText(text=cast(str, part["text"])))
        if part.get("type") == "image" and isinstance(part.get("data"), str) and isinstance(part.get("mimeType"), str):
            parts.append(ContentImage(image=f"data:{part['mimeType']};base64,{part['data']}"))
    return parts


def codex_tool_event(
    event: Mapping[str, JsonValue],
    timestamp: datetime,
    *,
    working_start: float = 0,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ToolEvent | None:
    """Render a completed tool item. Unknown or incomplete items have no projection."""
    item = event.get("item")
    if event.get("type") != "item.completed" or not isinstance(item, dict):
        return None
    identifier = item.get("id")
    if not isinstance(identifier, str):
        return None

    details: dict[str, JsonValue] = dict(metadata or {})
    result: str | list[ContentText | ContentImage]
    error = item.get("error")
    kind = item.get("type")
    status = item.get("status")
    function: str
    arguments: dict[str, JsonValue]
    if kind == "mcp_tool_call":
        tool = item.get("tool")
        recorded_arguments = item.get("arguments")
        if not isinstance(tool, str) or not isinstance(recorded_arguments, dict):
            return None
        function, arguments = tool, recorded_arguments
        result = _mcp_result(item.get("result"))
        details["server"] = item.get("server")
    elif kind == "command_execution":
        command = item.get("command")
        if not isinstance(command, str):
            return None
        function, arguments = "command_execution", {"command": command}
        output, exit_code = item.get("aggregated_output"), item.get("exit_code")
        result = output if isinstance(output, str) else ""
        if isinstance(exit_code, int):
            details["exit_code"] = exit_code
            if exit_code and not error:
                error = f"Command exited with code {exit_code}."
        if status == "failed" and not error:
            error = "Command execution failed."
    elif kind == "file_change":
        changes = item.get("changes")
        if not isinstance(changes, list):
            return None
        function, arguments = "file_change", {"changes": changes}
        result = ""
        if status == "failed" and not error:
            error = "File change failed."
    else:
        return None

    tool_error = ToolCallError("unknown", error if isinstance(error, str) else json.dumps(error)) if error else None
    return ToolEvent(
        timestamp=timestamp,
        working_start=working_start,
        id=identifier,
        function=function,
        arguments=arguments,
        result=result if isinstance(result, str) else [part for part in result],
        error=tool_error,
        metadata=details,
    )


def codex_event(
    event: Mapping[str, JsonValue],
    timestamp: datetime,
    *,
    working_start: float = 0,
    metadata: Mapping[str, JsonValue] | None = None,
) -> InfoEvent | ToolEvent | None:
    """Render observed Codex messages, terminal errors, and completed tools."""
    kind = event.get("type")
    item = event.get("item")
    if kind == "item.completed" and isinstance(item, dict) and item.get("type") == "agent_message":
        message = item.get("text")
        if isinstance(message, str):
            return InfoEvent(
                source="Codex message", timestamp=timestamp, working_start=working_start, data=message,
                metadata=dict(metadata or {}),
            )
    if kind in ("error", "turn.failed"):
        error = event.get("error", event.get("message"))
        if error is not None:
            return InfoEvent(
                source="Codex error", timestamp=timestamp, working_start=working_start, data=error,
                metadata=dict(metadata or {}),
            )
    return codex_tool_event(event, timestamp, working_start=working_start, metadata=metadata)
