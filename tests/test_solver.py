import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from coloph_agent_runtime import CodexClient
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import Score, accuracy, scorer

from coloph_agent_inspect.codex import codex_solver
from coloph_agent_inspect.recording import CodexRecord


class SolverTest(TestCase):
    def test_inspect_scores_recorded_cli_output_and_counts_it_once(self):
        @scorer(metrics=[accuracy()])
        def matches():
            async def score(state, target):
                return Score(value="C" if state.output.completion == target.text else "I")
            return score

        with TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "codex"
            events = [
                {"type": "thread.started", "thread_id": "session"},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
                {"type": "turn.completed", "usage": {
                    "input_tokens": 8, "cached_input_tokens": 4, "output_tokens": 2,
                }},
            ]
            fake.write_text(f"#!{sys.executable}\n" + "".join(f"print({json.dumps(event)!r})\n" for event in events))
            fake.chmod(0o700)
            records = root / "records"
            task = Task(
                dataset=[Sample(input="question", target="answer")],
                solver=codex_solver(cwd=root, records=records, client=CodexClient(executable=str(fake))),
                scorer=matches(),
            )
            log, = eval(task, model="mockllm/model", display="none", log_dir=str(root / "logs"))
            log = read_eval_log(log.location)
            sample, = log.samples
            score, = sample.scores.values()
            self.assertEqual(score.value, "C")
            self.assertEqual(sample.metadata["session_id"], "session")
            usage, = log.stats.model_usage.values()
            self.assertEqual((usage.input_tokens, usage.input_tokens_cache_read, usage.output_tokens), (8, 4, 2))
            path, = records.iterdir()
            self.assertEqual(CodexRecord.read(path).prompt, "question")
