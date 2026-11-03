"""Jobs, the answer extractor and the family registry."""
import dataclasses

import pytest

from purse.jobs import ANSWER_NOTE, extract_answer, family, make_job


def test_extract_answer_takes_the_last_tagged_answer():
    assert extract_answer("x <answer>1</answer> y <answer> 42 </answer>") == "42"
    assert extract_answer("```\n42\n```") == "42" and extract_answer("`abc`") == "abc"
    assert extract_answer("no tags") == "no tags"
    assert extract_answer('<answer>"quoted"</answer>') == "quoted"


def test_registry_rejects_unknown_families_by_name():
    with pytest.raises(KeyError, match="nope"):
        family("nope")


def test_a_job_is_immutable_and_its_prompt_asks_for_tags(fake_family):
    job = make_job("fake", 1, "low", seed=1, uid="fake-l1-low-000")
    assert job.prompt.endswith(ANSWER_NOTE) and job.answer == "1" and job.id == ""
    assert dict(job.features) == {"size": 1}
    with pytest.raises(dataclasses.FrozenInstanceError):
        job.value = 4  # type: ignore[misc]


def test_make_job_is_deterministic_per_seed(fake_family):
    assert make_job("fake", 3, "mid", seed=5, uid="u") == make_job("fake", 3, "mid", seed=5, uid="u")


def test_same_integer_reads_the_value_after_an_equals_sign():
    from purse.jobs import same_integer
    assert same_integer("1234", "(1 + 2) * 3 = 1234") and same_integer("-5", "x = -5.")
    assert not same_integer("1234", "1234 = 1235") and not same_integer("7", "= ")


def test_same_integer_handles_a_runaway_answer():
    """A model that loops at temperature 0 can write tens of thousands of digits; that is a wrong answer, not a crash
    (Python refuses to convert a string of more than 4300 digits to an int)."""
    from purse.jobs import same_integer
    assert not same_integer("1234", "<answer>" + "9" * 30_000 + "</answer>")
    assert same_integer("7", "<answer>007</answer>") and same_integer("0", "-0") and not same_integer("-7", "7")
