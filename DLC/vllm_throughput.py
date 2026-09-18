"""Parse and rank vLLM throughput measurements used by DeepSeek tuning."""

from __future__ import annotations

import math
import re
from typing import Iterable, Mapping


_THROUGHPUT_RE = re.compile(
    r"Throughput:\s*[-+0-9.eE]+\s*requests/s,\s*"
    r"(?P<total>[-+0-9.eE]+)\s*total tokens/s,\s*"
    r"(?P<output>[-+0-9.eE]+)\s*output tokens/s"
)


def parse_throughput(output: str) -> dict[str, float]:
    """Return finite positive vLLM throughput metrics from command output.

    The total-token rate is the optimization objective.  The output-token rate
    is retained as a companion metric for reports and regression checks.
    """
    match = _THROUGHPUT_RE.search(output)
    if match is None:
        raise ValueError("vLLM throughput line not found")

    total = float(match.group("total"))
    output_tokens = float(match.group("output"))
    if (not math.isfinite(total) or not math.isfinite(output_tokens)
            or total <= 0 or output_tokens <= 0):
        raise ValueError("vLLM throughput metrics must be finite and positive")

    return {
        "total_tokens_per_s": total,
        "output_tokens_per_s": output_tokens,
        # OpenTuner minimizes Result.time, so maximize total throughput by
        # minimizing its negation.
        "score": -total,
    }


def select_best_total(records: Iterable[Mapping[str, float]]) -> Mapping[str, float]:
    """Select the candidate with the highest total-token throughput."""
    candidates = list(records)
    if not candidates:
        raise ValueError("at least one throughput record is required")
    for record in candidates:
        total = float(record["total_tokens_per_s"])
        if not math.isfinite(total) or total <= 0:
            raise ValueError("throughput records must contain finite positive totals")
    return max(candidates, key=lambda record: float(record["total_tokens_per_s"]))
