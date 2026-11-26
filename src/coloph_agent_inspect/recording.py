"""Versioned on-disk records for a Coloph Codex invocation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from shutil import copytree
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, JsonValue, TypeAdapter

if TYPE_CHECKING:
    from pathlib import Path

    from coloph_agent_runtime import CodexClient, CodexResult, CodexRunOptions
    from coloph_agent_runtime._remote_codex import RemoteCodexClient


_JSON = TypeAdapter(JsonValue)


class RunError(BaseModel):
    kind: str
    message: str
    returncode: int | None = None


class CodexRecord(BaseModel):
    schema_version: Literal[1] = 1
    prompt: str
    model: str | None
    effort: str | None
    sandbox: str
    timeout_seconds: float
    started_at: datetime
    completed_at: datetime
    elapsed_seconds: float
    session_id: str | None = None
    output: str | None = None
    structured_output: JsonValue | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    error: RunError | None = None
    artifacts: dict[str, str]

    @classmethod
    def read(cls, directory: Path) -> CodexRecord:
        return cls.model_validate_json((directory / "run.json").read_bytes())


def read_events(directory: Path) -> Iterator[dict[str, JsonValue]]:
    with (directory / "codex.jsonl").open(encoding="utf-8") as events:
        for line in events:
            yield json.loads(line)


def record_codex(
    client: CodexClient | RemoteCodexClient,
    prompt: str,
    *,
    cwd: Path,
    directory: Path,
    options: CodexRunOptions,
    session_id: str | None = None,
    resume: bool = True,
    session_directory: Path | None = None,
    cancel: Event | None = None,
) -> CodexRecord:
    """Run Codex and preserve artifacts, including when the invocation fails.

    Use a separate output directory for each invocation. A supplied session
    directory must contain session files for this invocation only.
    """
    from coloph_agent_runtime import JsonlRecorder

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    clock = monotonic()
    artifacts = {"events": "codex.jsonl"}
    result: CodexResult | None = None
    failure: RunError | None = None
    seen_session = session_id
    stderr = ""

    with (directory / artifacts["events"]).open("x", encoding="utf-8") as stream:
        write_event = JsonlRecorder(stream)

        def on_event(event: dict[str, JsonValue]) -> None:
            nonlocal seen_session
            if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                seen_session = str(event["thread_id"])
            write_event(event)

        try:
            if session_id is None and cancel is None:
                result = client.run(prompt, cwd=cwd, options=options, on_event=on_event)
            else:
                result = client.run(
                    prompt, cwd=cwd, options=options, on_event=on_event,
                    session_id=session_id, resume=resume, cancel=cancel,
                )
            stderr = result.stderr
        except Exception as error:
            stderr = getattr(error, "stderr", "")
            failure = RunError(
                kind=type(error).__name__, message=str(error),
                returncode=getattr(error, "returncode", None),
            )
            raise
        finally:
            ended = datetime.now(timezone.utc)
            elapsed = monotonic() - clock
            if stderr:
                artifacts["stderr"] = "codex.stderr.txt"
                (directory / artifacts["stderr"]).write_text(stderr, encoding="utf-8")
            if session_directory is not None and session_directory.is_dir():
                artifacts["sessions"] = "sessions"
                copytree(session_directory, directory / artifacts["sessions"], dirs_exist_ok=True)
            record = CodexRecord(
                prompt=prompt, model=(result.model if result is not None else None) or options.model,
                effort=options.effort, sandbox=options.sandbox, timeout_seconds=options.timeout_seconds,
                started_at=started, completed_at=ended, elapsed_seconds=elapsed,
                session_id=result.session_id if result is not None else seen_session,
                output=result.response_text if result is not None else None,
                structured_output=_JSON.validate_python(result.structured_output) if result is not None else None,
                input_tokens=result.input_tokens if result is not None else None,
                cached_input_tokens=result.cache_read_tokens if result is not None else None,
                output_tokens=result.output_tokens if result is not None else None,
                error=failure, artifacts=artifacts,
            )
            pending = directory / ".run.pending.json"
            pending.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
            pending.replace(directory / "run.json")
            if failure is None:
                (directory / "usage.json").write_text(
                    json.dumps(record.model_dump(include={
                        "model", "effort", "session_id", "input_tokens", "cached_input_tokens", "output_tokens",
                    }), indent=2) + "\n", encoding="utf-8",
                )
    return record
