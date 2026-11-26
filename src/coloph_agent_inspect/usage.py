"""Accumulate observed usage across samples and runtime calls."""

from collections.abc import Iterable, Mapping

from inspect_ai.log import EvalLog
from inspect_ai.model import ModelUsage
from inspect_ai.model._model import model_usage, sample_model_usage


def merge_model_usage(groups: Iterable[Mapping[str, ModelUsage]]) -> dict[str, ModelUsage]:
    """Sum per-model counters without mutating any input."""
    totals: dict[str, ModelUsage] = {}
    for group in groups:
        for model, usage in group.items():
            totals[model] = totals[model] + usage if model in totals else usage.model_copy()
    return totals


def refresh_log_usage(log: EvalLog) -> None:
    """Recompute aggregate usage from the samples in a log."""
    log.stats.model_usage = merge_model_usage(sample.model_usage for sample in log.samples or [])


def record_sample_usage(model: str, usage: ModelUsage) -> None:
    """Add external usage to Inspect's active sample and run accumulators."""
    for totals in (sample_model_usage(), model_usage()):
        totals.update(merge_model_usage((totals, {model: usage})))
