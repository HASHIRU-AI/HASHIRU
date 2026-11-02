"""A market that answers from the outcome table instead of calling models (as RouterBench evaluates routers)."""
from collections.abc import Sequence

from purse.jobs import Job
from purse.market import Option, Outcome
from purse.table import OutcomeTable

__all__ = ["ReplayMarket"]


class ReplayMarket:
    """Attempt k of a (job, option) pair returns its recorded attempt number (draw + k) modulo the records, so seeds
    see different recorded outcomes where there are several. An outside agent costs its current price."""

    def __init__(self, table: OutcomeTable, market: Sequence[Option] = (), draw: int = 0) -> None:
        self.table, self.draw = table, draw
        self.prices = {o.name: o.per_message_usd for o in market if o.kind == "agent"}

    async def attempt(self, job: Job, option: str) -> Outcome:
        records = self.table.records(job.uid, option)
        if not records:
            return Outcome("", False, 0.0, 0.0, refused="no record")
        r = records[self.draw % len(records)]
        return Outcome(r.answer, r.solved, self.prices.get(option, r.cost_usd), r.seconds, r.refused)
