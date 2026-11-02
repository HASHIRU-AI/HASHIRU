"""Wider hidden bins (`purse pool --wide`): within a level, the hidden bin moves difficulty further, so that knowing it
is worth more (V5); without the switch, generation is what the first round's pool was built from."""
import hashlib
import json
import random
from dataclasses import asdict

from purse.families.aggregation import AGGREGATION
from purse.families.arithmetic import ARITHMETIC
from purse.families.strings import STRINGS
from purse.jobs import WIDE
from purse.projects import build_pool


def wide(generate, *args):
    token = WIDE.set(True)
    try:
        return generate(*args)
    finally:
        WIDE.reset(token)


def test_without_the_switch_the_pool_is_the_first_rounds():
    jobs, skipped = build_pool(1, 1)
    digest = hashlib.sha256("".join(json.dumps(asdict(j)) for j in jobs).encode()).hexdigest()[:16]
    assert (len(jobs), skipped, digest) == (60, [], "71c22d99e5650d77")


def test_wide_arithmetic_bins_change_digits_and_multiplications():
    def digits(hidden, seed):
        return dict(wide(ARITHMETIC.generate, 3, hidden, random.Random(seed))[2])["digits"]
    assert digits("low", 1) < dict(ARITHMETIC.generate(3, "low", random.Random(1))[2])["digits"] < digits("high", 1)
    times = [wide(ARITHMETIC.generate, 3, h, random.Random(s))[0].count("*")
             for h in ("low", "high") for s in range(20)]
    assert sum(times[20:]) > 2 * sum(times[:20])


def test_wide_aggregation_high_bin_has_more_near_misses():
    import re

    def at_threshold(text):  # orders right in region and status, at exactly the threshold: made only as near misses
        threshold = re.search(r"total above \$([\d.]+)", text).group(1)
        return text.count(f"total ${threshold} |")
    narrow = at_threshold(AGGREGATION.generate(3, "high", random.Random(5))[0])
    assert at_threshold(wide(AGGREGATION.generate, 3, "high", random.Random(5))[0]) >= 3 * narrow > 0
    mid = wide(AGGREGATION.generate, 3, "mid", random.Random(5))[0]
    asked = re.search(r"from region (\w+)", mid).group(1)
    assert any(f"region {r} |" in mid for r in ("North-East", "Northwest", "South-East", "Southwest")
               if r.startswith(asked[:5]))  # mid gets near-miss regions too


def test_wide_strings_high_bin_is_mostly_positional():
    def positional(text):  # "Rotate the string left by ..." and "Delete every ..." steps
        return text.count("Rotate the string left by") + text.count("Delete every")
    k = 7  # operations at level 4
    assert all(positional(wide(STRINGS.generate, 4, "high", random.Random(s))[0]) >= k - 1 for s in range(10))
    assert sum(positional(STRINGS.generate(4, "high", random.Random(s))[0]) for s in range(10)) < 10 * (k - 1)
    low = [wide(STRINGS.generate, 4, "low", random.Random(s))[0] for s in range(10)]
    assert not any("Shift every letter" in t for t in low)  # the low bin drops the Caesar shift


def test_wide_answers_are_still_exact():
    from purse.jobs import make_job
    token = WIDE.set(True)
    try:
        for fam in ("arithmetic", "aggregation", "strings"):
            for hidden in ("low", "mid", "high"):
                job = make_job(fam, 3, hidden, seed=11, uid="x")
                assert job.answer and job.prompt
    finally:
        WIDE.reset(token)


def test_the_pool_command_takes_the_switch(tmp_path):
    from purse import cli
    narrow, wider = tmp_path / "n.jsonl", tmp_path / "w.jsonl"
    assert cli.main(["pool", "--seed", "3", "--per-cell", "1", "--out", str(narrow)]) == 0
    assert cli.main(["pool", "--seed", "3", "--per-cell", "1", "--wide", "--out", str(wider)]) == 0
    assert narrow.read_text() != wider.read_text() and len(cli.load_pool(wider)) == 60


def test_wide_aggregation_bins_set_how_many_orders_match():
    def answer(hidden, seed):
        return int(wide(AGGREGATION.generate, 3, hidden, random.Random(seed))[1])
    assert all(answer("low", s) <= 4 for s in range(10))  # 400 orders, a handful to find
    assert all(answer("high", s) >= 400 // 10 for s in range(10))


def test_wide_strings_bins_set_the_start_length():
    from purse.families.strings import LENGTH

    def start(hidden):
        text = wide(STRINGS.generate, 3, hidden, random.Random(1))[0]
        return len(text.split("`")[1])
    assert start("low") == LENGTH[3] // 2 and start("mid") == LENGTH[3] and start("high") == 2 * LENGTH[3]
