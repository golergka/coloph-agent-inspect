"""Write completed or in-progress external trials as standard Inspect logs."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from inspect_ai.event import Event
from inspect_ai.log import (
    EvalConfig, EvalDataset, EvalError, EvalLog, EvalMetric, EvalResults, EvalSample,
    EvalScore, EvalSpec, EvalStats, write_eval_log,
)
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.scorer import Score
from pydantic import JsonValue


def write_log(log: EvalLog, destination: Path) -> Path:
    """Publish a complete .eval file atomically, including when replacing a log."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".pending", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write_eval_log(log, temporary, format="eval")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def external_log(
    spec: EvalSpec, sample: EvalSample, *,
    status: Literal["started", "success", "error"] = "success",
    invalidated: bool = False,
    error: EvalError | None = None,
    completed: bool = True,
    score_metric: str = "score",
    metadata: Mapping[str, JsonValue] | None = None,
) -> EvalLog:
    """Build an Inspect log for one trial that was run by another system."""
    aggregate = [
        EvalScore(name=name, scorer=name, scored_samples=1,
                  metrics={score_metric: EvalMetric(name=score_metric, value=float(score.value))})
        for name, score in (sample.scores or {}).items() if isinstance(score.value, (int, float))
    ]
    return EvalLog(
        status=status, invalidated=invalidated, error=error, eval=spec,
        samples=[sample], metadata=dict(metadata or {}),
        stats=EvalStats(started_at=sample.started_at or spec.created,
                        completed_at=sample.completed_at or "", model_usage=sample.model_usage),
        results=EvalResults(total_samples=1, completed_samples=int(completed), scores=aggregate),
    )


def export_trial(
    destination: Path,
    *,
    task: str,
    trial_id: str,
    prompt: str,
    output: ModelOutput,
    events: Sequence[Event] = (),
    scores: Mapping[str, Score] | None = None,
    error: EvalError | None = None,
    usage: Mapping[str, ModelUsage] | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> Path:
    """Export one externally executed trial. Score values belong to the caller."""
    started = (started_at or datetime.now(timezone.utc)).isoformat()
    completed = completed_at.isoformat() if completed_at is not None else None
    model_usage = dict(usage) if usage is not None else (
        {output.model: output.usage} if output.usage is not None else {}
    )
    sample = EvalSample(
        id=trial_id, epoch=1, input=prompt, target="", output=output, events=list(events),
        scores=dict(scores) if scores is not None and error is None else None, error=error,
        model_usage=model_usage, metadata=dict(metadata or {}),
        started_at=started, completed_at=completed,
    )
    spec = EvalSpec(
        eval_id=trial_id, run_id=trial_id, created=started, task=task,
        dataset=EvalDataset(name=task, samples=1), model=output.model, config=EvalConfig(),
    )
    return write_log(external_log(
        spec, sample, status="error" if error is not None else "success",
        error=error, completed=error is None,
    ), destination)
