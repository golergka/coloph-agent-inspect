"""Open an existing Coloph run in Inspect without starting an evaluation."""

import argparse
import json
from pathlib import Path

from inspect_ai.scorer import Score

from .codex import export_codex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="Directory containing run.json and codex.jsonl")
    parser.add_argument("destination", type=Path, help="Output .eval path")
    parser.add_argument("--task", required=True, help="Benchmark name")
    parser.add_argument("--trial-id", help="Sample ID. Defaults to the record directory name.")
    parser.add_argument("--scores", type=Path, help="JSON object with benchmark score names and values")
    args = parser.parse_args()
    scores = None
    if args.scores is not None:
        payload = json.loads(args.scores.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            parser.error("--scores must contain a JSON object")
        scores = {name: Score(value=value) for name, value in payload.items()}
    print(export_codex(
        args.record, args.destination, task=args.task,
        trial_id=args.trial_id or args.record.name, scores=scores,
    ))
