from datetime import datetime, timezone
import json
import unittest

from inspect_ai.event import InfoEvent
from pydantic import JsonValue

from coloph_agent_inspect.native import codex_event, codex_tool_event
from coloph_agent_inspect.responses import completed_response, response_event


class EventConversionTests(unittest.TestCase):
    def test_responses_event_retains_reported_usage_and_wire_payload(self) -> None:
        request: dict[str, JsonValue] = {'model': 'test-model', 'input': 'Task'}
        response: dict[str, JsonValue] = {
            'model': 'test-model',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Done'}]}],
            'usage': {
                'input_tokens': 31, 'output_tokens': 8, 'total_tokens': 39,
                'input_tokens_details': {'cached_tokens': 17},
                'output_tokens_details': {'reasoning_tokens': 5},
            },
        }
        body = 'data: ' + json.dumps({'type': 'response.completed', 'response': response})
        self.assertIsNone(completed_response(body))
        completed = completed_response(body + '\r\n\r\n')
        self.assertEqual(completed, response)
        assert completed is not None

        event = response_event(request, completed, datetime.now(timezone.utc), working_start=2)
        self.assertEqual(event.working_start, 2)
        self.assertEqual(event.output.completion, 'Done')
        assert event.output.usage is not None and event.call is not None
        self.assertEqual(event.output.usage.total_tokens, 39)
        self.assertEqual(event.output.usage.input_tokens_cache_read, 17)
        self.assertEqual(event.output.usage.reasoning_tokens, 5)
        self.assertEqual(event.call.request, request)
        self.assertEqual(event.call.response, response)

    def test_native_events_render_commands_changes_and_messages(self) -> None:
        timestamp = datetime.now(timezone.utc)
        command: dict[str, JsonValue] = {
            'id': 'cmd', 'type': 'command_execution', 'command': 'check sample',
            'aggregated_output': 'missing argument', 'exit_code': 2, 'status': 'failed',
        }
        self.assertIsNone(codex_tool_event({'type': 'item.started', 'item': command}, timestamp))
        tool = codex_tool_event({'type': 'item.completed', 'item': command}, timestamp)
        assert tool is not None and tool.error is not None and tool.metadata is not None
        self.assertEqual(tool.arguments, {'command': 'check sample'})
        self.assertEqual(tool.metadata['exit_code'], 2)
        self.assertEqual(tool.result, 'missing argument')
        self.assertIn('code 2', tool.error.message)

        changes: list[JsonValue] = [{'path': 'sample.txt', 'kind': 'update'}]
        edit = codex_tool_event({'type': 'item.completed', 'item': {
            'type': 'file_change', 'id': 'edit', 'changes': changes,
        }}, timestamp)
        assert edit is not None
        self.assertEqual(edit.arguments, {'changes': changes})

        message = codex_event({'type': 'item.completed', 'item': {
            'type': 'agent_message', 'text': 'Updated sample.',
        }}, timestamp)
        self.assertIsInstance(message, InfoEvent)
        assert isinstance(message, InfoEvent)
        self.assertEqual(message.data, 'Updated sample.')
