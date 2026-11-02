"""Builders for PURSE's tests."""
from purse.jobs import Job


def job(uid: str, family: str = "arithmetic", level: int = 1, hidden: str = "low", value: int = 1, answer: str = "1",
        id: str = "") -> Job:
    return Job(id=id, uid=uid, family=family, level=level, hidden=hidden, value=value, prompt=f"task {uid}",
               answer=answer, features=())
