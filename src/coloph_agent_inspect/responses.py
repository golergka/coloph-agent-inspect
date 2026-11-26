"""Normalize recorded OpenAI Responses exchanges without making model calls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Literal, cast

from inspect_ai.event import ModelEvent
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelCall,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall, ToolInfo
from pydantic import JsonValue, TypeAdapter

_JSON = TypeAdapter(JsonValue)


def _json_object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError('Recorded event must be a JSON object.')
    return value


def _content(value: JsonValue) -> list[ContentText | ContentImage]:
    if isinstance(value, str):
        return [ContentText(text=value)]
    parts: list[ContentText | ContentImage] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                parts.append(ContentText(text=json.dumps(item)))
            elif isinstance(item.get('text'), str):
                parts.append(ContentText(text=cast(str, item['text'])))
            elif item.get('type') == 'input_image' and isinstance(item.get('image_url'), str):
                parts.append(ContentImage(image=cast(str, item['image_url'])))
            else:
                parts.append(ContentText(text=json.dumps(item, ensure_ascii=False)))
    return parts


def _tool_call(value: dict[str, JsonValue]) -> ToolCall:
    identifier = str(value.get('call_id', value.get('id', '')))
    name = str(value.get('name', ''))
    if value.get('type') == 'custom_tool_call':
        return ToolCall(id=identifier, function=name, arguments={'input': value.get('input')}, type='custom')
    arguments = value.get('arguments', '{}')
    parsed = _JSON.validate_json(arguments) if isinstance(arguments, str) else arguments
    return ToolCall(id=identifier, function=name, arguments=_json_object(parsed))


def _input_messages(request: dict[str, JsonValue]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    instructions = request.get('instructions')
    if isinstance(instructions, str) and instructions:
        messages.append(ChatMessageSystem(content=instructions, metadata={'source_field': 'instructions'}))
    inputs = request.get('input', [])
    if isinstance(inputs, str):
        return [*messages, ChatMessageUser(content=inputs)]
    if not isinstance(inputs, list):
        raise ValueError('Recorded Responses input is not a string or array.')
    for item in inputs:
        if not isinstance(item, dict):
            raise ValueError('Recorded Responses input item is not an object.')
        role, kind = item.get('role'), item.get('type')
        parts = _content(item.get('content'))
        if role in ('system', 'developer'):
            messages.append(ChatMessageSystem(content=[part for part in parts], metadata={'source_role': role}))
        elif role == 'user':
            messages.append(ChatMessageUser(content=[part for part in parts], metadata={}))
        elif role == 'assistant':
            messages.append(ChatMessageAssistant(content=[part for part in parts], metadata={}))
        elif kind in ('function_call', 'custom_tool_call'):
            messages.append(ChatMessageAssistant(content='', tool_calls=[_tool_call(item)], metadata={}))
        elif kind in ('function_call_output', 'custom_tool_call_output'):
            output = item.get('output')
            if isinstance(output, list):
                messages.append(ChatMessageTool(
                    content=[part for part in _content(output)], tool_call_id=str(item.get('call_id', '')), metadata={},
                ))
            else:
                messages.append(ChatMessageTool(
                    content=output if isinstance(output, str) else json.dumps(output),
                    tool_call_id=str(item.get('call_id', '')), metadata={},
                ))
        elif kind == 'reasoning':
            summary = item.get('summary', [])
            text = '\n'.join(part.text for part in _content(summary) if isinstance(part, ContentText))
            messages.append(ChatMessageAssistant(
                content=[ContentReasoning(reasoning=text, redacted=not bool(text))], metadata={},
            ))
        else:
            # Unsupported items remain explicitly identifiable in exact raw
            # evidence. Do not claim this normalized ModelEvent is complete.
            raise ValueError(f'Unsupported captured Responses input item: {kind}')
    return messages


def completed_response(text: str) -> dict[str, JsonValue] | None:
    """Return JSON or the terminal Responses payload from a complete SSE frame."""
    if text.lstrip().startswith('{'):
        value = _JSON.validate_json(text)
        return _json_object(value)
    for block in text.replace('\r\n', '\n').split('\n\n')[:-1]:
        payload = '\n'.join(line[5:].lstrip() for line in block.splitlines() if line.startswith('data:'))
        if not payload or payload == '[DONE]':
            continue
        value = _JSON.validate_json(payload)
        if isinstance(value, dict) and value.get('type') == 'response.completed':
            return _json_object(value.get('response'))
    return None


def _usage(raw: dict[str, JsonValue]) -> ModelUsage:
    input_details = raw.get('input_tokens_details')
    output_details = raw.get('output_tokens_details')
    usage = dict(raw)
    if isinstance(input_details, dict):
        cached = input_details.get('cached_tokens')
        if type(cached) is int:
            usage.setdefault('input_tokens_cache_read', cached)
    if isinstance(output_details, dict):
        reasoning = output_details.get('reasoning_tokens')
        if type(reasoning) is int:
            usage.setdefault('reasoning_tokens', reasoning)
    return ModelUsage.model_validate(usage)


def response_event(
    request: dict[str, JsonValue],
    response: dict[str, JsonValue],
    timestamp: datetime,
    *,
    working_start: float = 0,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ModelEvent:
    """Convert an observed Responses exchange into an Inspect model event."""
    parts: list[ContentText | ContentImage | ContentReasoning] = []
    calls: list[ToolCall] = []
    output = response.get('output', [])
    if not isinstance(output, list):
        raise ValueError('Recorded Responses output is not an array.')
    for item in output:
        if not isinstance(item, dict):
            raise ValueError('Recorded Responses output item is not an object.')
        if item.get('type') == 'message':
            parts.extend(_content(item.get('content')))
        elif item.get('type') in ('function_call', 'custom_tool_call'):
            calls.append(_tool_call(item))
        elif item.get('type') == 'reasoning':
            summary = '\n'.join(part.text for part in _content(item.get('summary')) if isinstance(part, ContentText))
            parts.append(ContentReasoning(reasoning=summary, redacted=not bool(summary)))
        else:
            raise ValueError(f"Unsupported captured Responses output item: {item.get('type')}")
    tools: list[ToolInfo] = []
    raw_tools = request.get('tools', [])
    if isinstance(raw_tools, list):
        for tool in raw_tools:
            if isinstance(tool, dict) and tool.get('type') == 'function':
                tools.append(ToolInfo.model_validate({
                    'name': tool.get('name'), 'description': tool.get('description', ''),
                    'parameters': tool.get('parameters', {}),
                }))
            elif isinstance(tool, dict) and tool.get('type') == 'custom':
                tools.append(ToolInfo(
                    name=str(tool.get('name', '')), description=str(tool.get('description', '')),
                    options={'type': 'custom', 'format': tool.get('format')},
                ))
    raw_usage = response.get('usage')
    usage = _usage(raw_usage) if isinstance(raw_usage, dict) else None
    model = str(response.get('model', request.get('model', 'recorded-runtime')))
    assistant = ChatMessageAssistant(content=[part for part in parts], tool_calls=calls or None, model=model)
    completion = '\n'.join(part.text for part in parts if isinstance(part, ContentText))
    config = GenerateConfig()
    if isinstance(request.get('temperature'), (int, float)):
        config.temperature = float(cast(float, request['temperature']))
    if isinstance(request.get('max_output_tokens'), int):
        config.max_tokens = cast(int, request['max_output_tokens'])
    raw_choice = request.get('tool_choice', 'auto')
    choice: Literal['auto', 'any', 'none'] = 'any' if raw_choice == 'required' else 'none' if raw_choice == 'none' else 'auto'
    return ModelEvent(
        timestamp=timestamp, working_start=working_start, model=model, input=_input_messages(request),
        tools=tools, tool_choice=choice, config=config,
        output=ModelOutput(
            model=model,
            choices=[ChatCompletionChoice(message=assistant, stop_reason='tool_calls' if calls else 'stop')],
            completion=completion,
            usage=usage,
        ),
        call=ModelCall(request=request, response=response),
        metadata={**(metadata or {}), 'normalization': 'Responses API; developer role retained in message metadata'},
    )
