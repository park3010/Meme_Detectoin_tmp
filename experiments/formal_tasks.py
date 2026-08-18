"""Canonical paper-facing task and metric provenance policy."""

FORMAL_TASKS = (
    "harmfulness",
    "target_presence",
    "target_granularity",
    "intent_primary",
    "tactic_rhetorical",
    "tactic_multimodal_relation",
)


def is_formal_task(task: str) -> bool:
    return task in FORMAL_TASKS


__all__ = ["FORMAL_TASKS", "is_formal_task"]
