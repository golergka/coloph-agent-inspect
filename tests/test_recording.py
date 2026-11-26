from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from coloph_agent_runtime import CodexClient, CodexResult, CodexRunOptions
from coloph_agent_runtime.codex import CodexInvocationError
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import Score

from coloph_agent_inspect.codex import export_codex
from coloph_agent_inspect.recording import CodexRecord, read_events, record_codex


class RecordingTest(unittest.TestCase):
    def test_record_and_reexport_preserve_partial_and_zero_scores(self):
        def invoke(prompt, *, cwd, options, on_event):
            on_event({"type": "thread.started", "thread_id": "session"})
            return CodexResult(
                session_id="session", model="model", response_text="answer",
                input_tokens=12, cache_read_tokens=3, output_tokens=2,
            )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            client = Mock(spec=CodexClient)
            client.run.side_effect = invoke
            record = record_codex(client, "question", cwd=root, directory=root / "record",
                                  options=CodexRunOptions())
            self.assertEqual(record.session_id, "session")
            self.assertGreaterEqual(record.elapsed_seconds, 0.0)
            self.assertEqual(list(read_events(root / "record")),
                             [{"type": "thread.started", "thread_id": "session"}])
            destination = root / "trial.eval"
            for scores in (None, {"reward": Score(value=0), "criteria": Score(value={"covered": 1})}):
                export_codex(root / "record", destination, task="external", trial_id="trial", scores=scores)
                log = read_eval_log(destination)
                sample, = log.samples
                self.assertEqual(sample.scores, scores)
                self.assertEqual(sample.output.completion, "answer")
                self.assertEqual(log.stats.model_usage["model"].input_tokens_cache_read, 3)

    def test_failure_preserves_stderr_and_reports_inspect_error(self):
        def invoke(prompt, *, cwd, options, on_event):
            on_event({"type": "thread.started", "thread_id": "session"})
            raise CodexInvocationError("failed", stderr="detail", returncode=2)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            client = Mock(spec=CodexClient)
            client.run.side_effect = invoke
            with self.assertRaisesRegex(CodexInvocationError, "failed"):
                record_codex(client, "question", cwd=root, directory=root / "record",
                             options=CodexRunOptions())
            record = CodexRecord.read(root / "record")
            self.assertEqual(record.session_id, "session")
            self.assertEqual(record.error.returncode, 2)
            self.assertEqual((root / "record" / record.artifacts["stderr"]).read_text(), "detail")
            path = export_codex(root / "record", root / "trial.eval", task="external", trial_id="trial")
            log = read_eval_log(path)
            self.assertEqual(log.status, "error")
            self.assertEqual(log.error.message, "failed")
            self.assertIsNone(log.samples[0].scores)
