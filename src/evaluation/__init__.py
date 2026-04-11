"""Evaluation package.

Contains experiment metrics and result summarization.
"""

from evaluation.metrics import EvaluationResult, compute_attack_success, compute_contamination_rate, compute_faithfulness, summarize_run
from evaluation.framework import (
    BenchmarkCase,
    BenchmarkMetrics,
    BenchmarkReport,
    CallableRunner,
    RunOutcome,
    SecureRAGEvaluationFramework,
)

__all__ = [
    "EvaluationResult",
    "compute_contamination_rate",
    "compute_attack_success",
    "compute_faithfulness",
    "summarize_run",
    "BenchmarkCase",
    "RunOutcome",
    "BenchmarkMetrics",
    "BenchmarkReport",
    "CallableRunner",
    "SecureRAGEvaluationFramework",
]
