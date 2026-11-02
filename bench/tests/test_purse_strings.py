"""F4: a pipeline of string operations."""
import random
import re

import pytest

from purse.families import strings
from purse.families.strings import LENGTH, OPERATIONS, STRINGS


def test_each_operation():
    assert strings.reverse("abc") == "cba"
    assert strings.rotate_left("abcde", 2) == "cdeab" and strings.rotate_left("abc", 4) == "bca"
    assert strings.caesar("Az-y", 2) == "Cb-a"
    assert strings.swapcase("aB1") == "Ab1"
    assert strings.drop_every("abcdefg", 3) == "abdeg"
    assert strings.double_vowels("bAe1") == "bAAee1"


STEPS = [(re.compile(r"Reverse the string\."), lambda s, g: strings.reverse(s)),
         (re.compile(r"Rotate the string left by (\d+)"), lambda s, g: strings.rotate_left(s, int(g[0]))),
         (re.compile(r"Shift every letter forward (\d+)"), lambda s, g: strings.caesar(s, int(g[0]))),
         (re.compile(r"Swap the case"), lambda s, g: strings.swapcase(s)),
         (re.compile(r"Delete every (\d+)"), lambda s, g: strings.drop_every(s, int(g[0]))),
         (re.compile(r"Double every vowel"), lambda s, g: strings.double_vowels(s))]


def run(task: str) -> tuple[str, int]:
    """Apply the task's numbered operations to its start string, read independently from the text."""
    s = re.search(r"Start with the string `(.*?)` and apply", task).group(1)
    ops = re.findall(r"^\d+\. (.+)$", task, re.M)
    for op in ops:
        for pattern, f in STEPS:
            if m := pattern.match(op):
                s = f(s, m.groups())
                break
        else:
            raise AssertionError(op)
    return s, len(ops)


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("hidden", ["low", "mid", "high"])
def test_answer_applies_the_listed_operations_in_order(level, hidden):
    for seed in range(5):
        task, answer, features = STRINGS.generate(level, hidden, random.Random(seed))
        final, count = run(task)
        assert final == answer and count == OPERATIONS[level] == features["operations"]
        assert features["length"] == LENGTH[level] and answer == answer.strip() and answer


def test_bins_set_the_alphabet_and_the_operations():
    low = [STRINGS.generate(3, "low", random.Random(s))[0] for s in range(20)]
    high = [STRINGS.generate(3, "high", random.Random(s))[0] for s in range(20)]
    assert all(re.search(r"`[A-Za-z]+`", t) for t in low)
    assert not any(re.search(r"Rotate|Delete", t) for t in low)
    assert all(len(re.findall(r"Rotate|Delete", t)) * 2 >= OPERATIONS[3] for t in high)
    assert all(int(k) > 13 for t in high for k in re.findall(r"forward (\d+)", t))


def test_correct_strips_tags_and_backticks_but_not_inner_spaces():
    assert STRINGS.correct("a b", "<answer>`a b`</answer>")
    assert not STRINGS.correct("a b", "<answer>ab</answer>")
    assert not STRINGS.correct("Ab", "<answer>ab</answer>")
