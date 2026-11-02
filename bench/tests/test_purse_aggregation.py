"""F3: counting the records in a long document that match a condition."""
import random
import re

import pytest

from purse.families.aggregation import AGGREGATION, RECORDS

LINE = re.compile(r"^#\d+ \| [\d-]+ \| customer [\w ]+ \| region ([\w-]+) \| items (\d+) \| total \$([\d.]+) \| "
                  r"status ([\w-]+)$")


def recount(task: str) -> int:
    """An independent reading: parse every record and the question, and count."""
    question = task.rsplit("\n\n", 1)[1]
    region = re.search(r"from region ([\w-]+)", question).group(1)
    status = m.group(1) if (m := re.search(r"status ([\w-]+)", question)) else None
    above = float(m.group(1)) if (m := re.search(r"total above \$([\d.]+)", question)) else None
    count = 0
    for line in task.splitlines():
        if r := LINE.match(line):
            ok = r.group(1) == region and (status is None or r.group(4) == status)
            count += ok and (above is None or float(r.group(3)) > above)
    return count


@pytest.mark.parametrize(("level", "hidden"), [(1, "low"), (2, "mid"), (3, "high"), (4, "high")])
def test_answer_counts_matches(level, hidden):
    for seed in range(5):
        task, answer, features = AGGREGATION.generate(level, hidden, random.Random(seed))
        assert sum(bool(LINE.match(x)) for x in task.splitlines()) == RECORDS[level] == features["records"]
        assert recount(task) == int(answer)
        assert 3 <= int(answer) <= max(3, RECORDS[level] // 5)


def test_high_bin_contains_near_misses():
    task, _, _ = AGGREGATION.generate(3, "high", random.Random(1))
    question = task.rsplit("\n\n", 1)[1]
    region, threshold = re.search(r"from region (\w+)", question).group(1), re.search(r"above \$([\d.]+)", question)
    records = [LINE.match(x) for x in task.splitlines() if LINE.match(x)]
    assert any(r.group(1) != region and r.group(1).startswith(region) for r in records)
    assert any(r.group(4) == "shipped-late" for r in records)
    assert any(r.group(3) == threshold.group(1) and r.group(1) == region for r in records)
    low_task, _, _ = AGGREGATION.generate(3, "low", random.Random(1))
    assert "-late" not in low_task and "North-East" not in low_task


def test_level_5_is_long():
    task, _, _ = AGGREGATION.generate(5, "mid", random.Random(0))
    assert len(task) > 150_000


def test_correct_ignores_formatting():
    assert AGGREGATION.correct("17", "<answer> 17 </answer>") and AGGREGATION.correct("17", "17.")
    assert not AGGREGATION.correct("17", "<answer>18</answer>") and not AGGREGATION.correct("17", "seventeen")
