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
