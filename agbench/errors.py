"""Exceptions shared by the harness and the workloads."""


class SkipScenario(Exception):
    """Raised by a workload (or the harness) when a scenario does not apply here.

    ``reason`` is a short machine-readable word, e.g. ``"workers_exceed_cores"`` or
    ``"model_missing"``; it ends up in the record's ``status_detail``.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason if not detail else f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class ScenarioError(Exception):
    """A workload found a problem that makes its numbers meaningless (status ``error``)."""
