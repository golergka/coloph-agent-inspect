"""Publish externally executed tools to an active Inspect transcript."""

from collections.abc import Awaitable, Callable, Collection
from datetime import datetime, timezone
from typing import TypeVar
import json

from inspect_ai.event import Event, ModelEvent, ToolEvent
from inspect_ai.log import EvalSample, transcript
from inspect_ai.model import ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallError
from pydantic import JsonValue

T = TypeVar("T")


def record_event(event: Event) -> None:
    """Publish a normalized event using Inspect's transcript event API."""
    transcript()._event(event)


async def observe_tool(
    operation: Callable[[], Awaitable[T]], *,
    call_id: str, name: str, arguments: dict[str, JsonValue], source: str,
) -> T:
    """Record a tool execution while preserving its return value or exception."""
    event = ToolEvent(id=call_id, function=name, arguments=arguments, pending=True,
                      metadata={"source": source})
    record_event(event)
    try:
        result = await operation()
        event.result = result if isinstance(result, str) else json.dumps(result, default=str)
        return result
    except BaseException as error:
        event.error = ToolCallError("unknown", f"{type(error).__name__}: {error}")
        event.failed = True
        raise
    finally:
        event.pending = False
        event.completed = datetime.now(timezone.utc)
        event.working_time = (event.completed - event.timestamp).total_seconds()
        transcript()._event_updated(event)


def restore_tool_history(
    sample: EvalSample, *,
    separate_spans: Collection[str] = (),
    excluded_tools: Collection[str] = (),
) -> list[tuple[ModelEvent, ToolCall]]:
    """Recover observed tool results without inventing execution or duration."""
    model_events = [
        event for event in sample.events if isinstance(event, ModelEvent) and event.role != "judge"
    ]
    parent_events = [event for event in model_events if event.span_id not in separate_spans]
    longest = max((event.input for event in parent_events), key=len, default=[])
    results = {
        message.tool_call_id: message
        for messages in [*(event.input for event in model_events), sample.messages]
        for message in messages if isinstance(message, ChatMessageTool)
    }
    if len(longest) > len(sample.messages):
        sample.messages = list(longest)
        last_answer = next(
            (event.output.choices[0].message for event in reversed(parent_events) if event.output.choices), None
        )
        if last_answer is not None:
            sample.messages.append(last_answer)

    existing = {event.id for event in sample.events if isinstance(event, ToolEvent)}
    issued: list[tuple[ModelEvent, ToolCall]] = []
    additions: dict[int, list[ToolEvent]] = {}
    for event in model_events:
        calls = event.output.choices[0].message.tool_calls if event.output.choices else None
        for call in calls or ():
            if call.function in excluded_tools:
                continue
            issued.append((event, call))
            if call.id in existing:
                continue
            result = results.get(call.id)
            additions.setdefault(id(event), []).append(ToolEvent(
                id=call.id, function=call.function, arguments=call.arguments,
                timestamp=event.timestamp, working_start=event.working_start,
                result=result.text if result is not None else "Execution outcome unavailable in source log.",
                error=result.error if result is not None else None,
                metadata={"source": "recovered_model_history", "execution_observed": result is not None,
                          "timing": "request timestamp only; execution duration unknown"},
            ))
            existing.add(call.id)
    sample.events = [item for event in sample.events for item in (event, *additions.get(id(event), ()))]
    return issued
