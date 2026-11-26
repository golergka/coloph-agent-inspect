"""Converters for Codex run records and an Inspect solver."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from os.path import relpath
from pathlib import Path
from tempfile import mkdtemp
from threading import Event as CancelEvent
from typing import TYPE_CHECKING

from inspect_ai.event import Event, InfoEvent
from inspect_ai.log import EvalError
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.solver import Generate, Solver, TaskState, solver
from pydantic import JsonValue

from .export import export_trial
from .native import codex_event
from .recording import CodexRecord, read_events, record_codex
from .tool_events import record_event
from .usage import record_sample_usage

if TYPE_CHECKING:
    from coloph_agent_runtime import CodexClient, CodexRunOptions
    from coloph_agent_runtime._remote_codex import RemoteCodexClient
    from inspect_ai.scorer import Score


def recorded_output(record: CodexRecord) -> ModelOutput:
    model = record.model or "codex"
    usage = None
    if any(value is not None for value in (
        record.input_tokens, record.cached_input_tokens, record.output_tokens,
    )):
        usage = ModelUsage(
            input_tokens=record.input_tokens or 0, output_tokens=record.output_tokens or 0,
            input_tokens_cache_read=record.cached_input_tokens,
            total_tokens=(record.input_tokens or 0) + (record.output_tokens or 0),
        )
    return ModelOutput(model=model, completion=record.output or "", usage=usage,
                       time=record.elapsed_seconds)


def recorded_metadata(record: CodexRecord) -> dict[str, JsonValue]:
    return record.model_dump(mode="json", exclude={"prompt", "output"})


def recorded_events(directory: Path, record: CodexRecord) -> list[Event]:
    events: list[Event] = []
    for line, source in enumerate(read_events(directory), start=1):
        event = codex_event(source, record.started_at, metadata={
            "source_file": "codex.jsonl", "source_line": line,
            "timestamp_source": "run start; per-event wall clock unavailable",
        })
        if event is not None:
            events.append(event)
    if record.error is not None:
        events.append(InfoEvent(source="Coloph", timestamp=record.completed_at, data=record.error.message))
    return events


def export_codex(
    directory: Path, destination: Path, *, task: str, trial_id: str,
    scores: dict[str, Score] | None = None,
    infrastructure_error: EvalError | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> Path:
    record = CodexRecord.read(directory)
    error = infrastructure_error
    if error is None and record.error is not None and scores is None:
        message = record.error.message
        error = EvalError(message=message, traceback=message, traceback_ansi=message)
    metadata = recorded_metadata(record)
    metadata["record_directory"] = relpath(directory, destination.parent)
    if artifacts is not None:
        metadata["benchmark_artifacts"] = dict(artifacts)
    return export_trial(
        destination, task=task, trial_id=trial_id, prompt=record.prompt,
        output=recorded_output(record), events=recorded_events(directory, record),
        scores=scores, error=error, metadata=metadata,
        started_at=record.started_at, completed_at=record.completed_at,
    )


@solver
def codex_solver(
    *,
    cwd: Path | Callable[[TaskState], Path],
    records: Path,
    options: CodexRunOptions | None = None,
    client: CodexClient | RemoteCodexClient | None = None,
) -> Solver:
    """Run Coloph inside an Inspect task, leaving scoring to that task."""
    from coloph_agent_runtime import CodexClient, CodexRunOptions

    runtime = client if client is not None else CodexClient()
    run_options = options if options is not None else CodexRunOptions(skip_git_repo_check=True, capture_events=False)
    records.mkdir(mode=0o700, parents=True, exist_ok=True)
    workspace_lock = asyncio.Lock()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        directory = Path(mkdtemp(prefix="trial-", dir=records))
        prompt = "\n\n".join(message.text for message in state.messages)
        workspace = cwd(state) if callable(cwd) else cwd

        async def run() -> None:
            cancel = CancelEvent()
            attempt = asyncio.create_task(asyncio.to_thread(
                record_codex, runtime, prompt, cwd=workspace, directory=directory,
                options=run_options, cancel=cancel,
            ))
            try:
                await asyncio.shield(attempt)
            except asyncio.CancelledError:
                cancel.set()
                await asyncio.gather(attempt, return_exceptions=True)
                raise

        try:
            if callable(cwd):
                await run()
            else:
                async with workspace_lock:
                    await run()
        finally:
            if (directory / "run.json").is_file():
                record = CodexRecord.read(directory)
                state.output = recorded_output(record)
                state.metadata.update(recorded_metadata(record))
                state.metadata["record_directory"] = directory.name
                state.metadata["session_id"] = record.session_id
                if state.output.usage is not None:
                    record_sample_usage(state.output.model, state.output.usage)
                for event in recorded_events(directory, record):
                    record_event(event)
        return state

    return solve
