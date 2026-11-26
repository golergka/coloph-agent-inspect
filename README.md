# Coloph Agent Inspect

Use this library to understand how a coding agent got its benchmark score.
It gives you the same way to inspect runs across benchmarks.

The library connects Coloph Agent Runtime to Inspect AI. It records the agent
configuration, session, output, events, errors, timing and token usage.
Your benchmark supplies the tasks and scores.

## Install

Install the packages from GitHub Releases. The packages require Python 3.11
or later. If you already installed Coloph Agent Runtime, install only the
second wheel.

```sh
uv pip install \
  https://github.com/golergka/coloph-agent-runtime/releases/download/v0.3.0/coloph_agent_runtime-0.3.0-py3-none-any.whl \
  https://github.com/golergka/coloph-agent-inspect/releases/download/v0.1.0/coloph_agent_inspect-0.1.0-py3-none-any.whl
```

## Inspect runs the benchmark

Pass `codex_solver` the workspace and the directory for trial records. Use your
benchmark's dataset and scorer. With multiple samples, provide a workspace
function that returns a separate checkout or container path for each sample.

```python
from pathlib import Path

from inspect_ai import Task
from coloph_agent_inspect.codex import codex_solver

task = Task(
    dataset=dataset,
    solver=codex_solver(cwd=Path("/workspace"), records=Path("agent-runs")),
    scorer=benchmark_scorer,
)
```

The default client runs Codex locally. To use a container that Coloph
already created, pass the container's Codex client with `client=`.

## Another tool runs the benchmark

Call `record_codex` where your existing runner invokes Coloph.

```python
from pathlib import Path

from coloph_agent_inspect.recording import record_codex

record = record_codex(
    client, prompt, cwd=Path("/workspace"),
    directory=Path("agent-runs/trial-1"), options=options,
)
```

After the benchmark produces scores, add them to the Inspect log.

```sh
coloph-agent-inspect agent-runs/trial-1 logs/trial-1.eval --task my-benchmark --scores scores.json
inspect view --log-dir logs
```

The scores file maps metric names to values, for example `{"passed": 1}`.
Omit that file to inspect an ungraded run. Use a separate directory for each
agent invocation. Pass `session_directory` to preserve the files from an
isolated session directory.

The logs contain benchmark prompts and agent output. Store them with the same
access controls as the source data. Full model requests and responses require a
capture that supplies them. The Codex event stream supplies agent messages and
recorded tool events.
