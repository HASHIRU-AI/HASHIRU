"""F2: logic grids with one solution, the solver's split count, and the text format the peers parse."""
import json
import random

import pytest

from purse.families import grids
from purse.families.grids import GRIDS, SIZE, Puzzle, parse, render, satisfies, solve
from purse.jobs import InfeasibleCell


def puzzle_of(task: str) -> Puzzle:
    return parse(task)


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_render_parse_round_trip(level):
    for seed in range(20):
        p = grids.draw(level, random.Random(seed))
        assert parse(render(p)) == p


@pytest.mark.parametrize(("level", "hidden"), [(1, "low"), (3, "mid"), (5, "high")])
def test_generated_puzzles_have_one_solution_and_the_bin_they_claim(level, hidden):
    task, answer, features = GRIDS.generate(level, hidden, random.Random(level))
    p = puzzle_of(task)
    solutions, splits = solve(p)
    assert len(solutions) == 1 and grids.bin_of(splits) == hidden
    assert solutions[0] == json.loads(answer)
    assert (p.n, len(p.attributes)) == SIZE[level]
    assert features == {"positions": p.n, "attributes": len(p.attributes), "clues": len(p.clues)}


def test_satisfies_accepts_only_the_solution():
    task, answer, _ = GRIDS.generate(3, "low", random.Random(1))
    p, solution = puzzle_of(task), json.loads(answer)
    assert satisfies(p, solution)
    attr = p.attributes[0][0]
    solution["1"][attr], solution["2"][attr] = solution["2"][attr], solution["1"][attr]
    assert not satisfies(p, solution)
    assert not satisfies(p, {"1": {}})


def test_infeasible_cell_raises(monkeypatch):
    monkeypatch.setattr(grids, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(grids, "bin_of", lambda splits: "low")
    with pytest.raises(InfeasibleCell):
        GRIDS.generate(2, "high", random.Random(0))


def test_correct_reads_json_in_tags_or_fences_case_insensitively():
    _, answer, _ = GRIDS.generate(2, "low", random.Random(3))
    shouted = json.dumps({k: {a: v.upper() for a, v in row.items()} for k, row in json.loads(answer).items()})
    assert GRIDS.correct(answer, f"<answer>{answer}</answer>")
    assert GRIDS.correct(answer, f"Here it is:\n```json\n{shouted}\n```")
    wrong = json.loads(answer)
    wrong["1"] = wrong["2"]
    assert not GRIDS.correct(answer, json.dumps(wrong))
    assert not GRIDS.correct(answer, "<answer>not json</answer>")
