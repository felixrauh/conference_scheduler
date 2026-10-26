"""
Tests for Phase 2: partitioning Phase 1 tuples into blocks and ordering them.

Tuples come from the greedy Phase 1 of the heuristic pipeline, so these tests
run without a MILP solver.
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_from_csv_dir
from src.instance import build_instance
from src.phase2 import (
    Block,
    FixedSequence,
    Phase2Input,
    check_block_feasibility,
    compute_block_hopping,
    optimize_block_ordering,
    repair_block_feasibility,
    solve_phase2,
)
from src.pipelines.heuristic import _greedy_phase1

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def orbel2017():
    """ORBEL 2017 (80 talks; 5 blocks of 4 rooms x 4 talks) with greedy tuples."""
    data = load_from_csv_dir(EXAMPLES / "orbel2017")
    data.validate()
    instance = build_instance(data)
    tuples_by_n = _greedy_phase1(
        instance.talks, instance.preferences, [(4, 20)],
        talk_presenter=instance.talk_presenter)
    return instance, tuples_by_n


@pytest.mark.parametrize("strategy", ["greedy", "random", "matching"])
def test_partition_places_every_tuple_once(orbel2017, strategy):
    instance, tuples_by_n = orbel2017
    phase2_input = Phase2Input(
        tuples_by_n=tuples_by_n,
        block_specs=[(4, 4, 5, "4R4T")],
        preferences=instance.preferences,
    )

    result = solve_phase2(phase2_input, partition_strategy=strategy,
                          use_local_search=False, verbose=False)

    assert len(result.blocks) == 5
    assert all(b.block_type == "4R4T" and len(b.tuples) == 4 for b in result.blocks)
    placed = [t for b in result.blocks for t in b.tuples]
    assert sorted(placed) == sorted(tuples_by_n[4])
    assert result.total_hopping == sum(
        compute_block_hopping(b.tuples, instance.preferences) for b in result.blocks)


def test_enumerated_ordering_beats_input_and_greedy_order(orbel2017):
    instance, tuples_by_n = orbel2017
    tuples = tuples_by_n[4][:4]

    ordered, cost = optimize_block_ordering(tuples, instance.preferences, "enumerate")
    _, greedy_cost = optimize_block_ordering(tuples, instance.preferences, "greedy")

    assert sorted(ordered) == sorted(tuples)
    assert cost == compute_block_hopping(ordered, instance.preferences)
    assert cost <= compute_block_hopping(tuples, instance.preferences)
    assert cost <= greedy_cost


def test_fixed_sequence_keeps_its_talk_order(orbel2017):
    """An attached fixed sequence stays in one room, in the configured order."""
    instance, _ = orbel2017
    # Use popular talks so that re-ordering would change the hopping cost
    popularity = Counter(t for prefs in instance.preferences.values() for t in prefs)
    fixed_talks = tuple(t for t, _ in popularity.most_common(4))
    other_talks = [t for t in instance.talks if t not in fixed_talks]
    tuples_by_n = _greedy_phase1(other_talks, instance.preferences, [(4, 16), (3, 4)],
                                 talk_presenter=instance.talk_presenter)
    fixed = FixedSequence(name="Award", talks=fixed_talks,
                          target_block_type="3R4T", result_block_type="4R4T")

    for strategy in ["greedy", "matching"]:
        result = solve_phase2(
            Phase2Input(tuples_by_n, [(4, 4, 4, "4R4T"), (3, 4, 1, "3R4T")],
                        instance.preferences, fixed_sequences=[fixed]),
            partition_strategy=strategy, use_local_search=False, verbose=False)

        hosts = [b for b in result.blocks
                 if any(fixed_talks[0] in t for t in b.tuples)]
        assert len(hosts) == 1
        block = hosts[0]
        assert block.block_type == "4R4T"
        assert [t[-1] for t in block.tuples] == list(fixed_talks)


def test_repair_makes_infeasible_block_schedulable():
    """Two presenters who are only available in different timeslots get split up."""
    preferences = {"p1": {"T1", "T5"}, "p2": {"T2", "T6"}}
    talk_presenter = {f"T{i}": f"P{i}" for i in range(1, 9)}
    unavailability = {"P1": {"TB"}, "P2": {"TA"}}
    blocks = [
        Block("B01", "2R2T", [("T1", "T3"), ("T2", "T4")], 0),  # P1 + P2: infeasible
        Block("B02", "2R2T", [("T5", "T7"), ("T6", "T8")], 0),
    ]
    timeslots = {"TA", "TB"}
    by_type = {"2R2T": ["TA", "TB"]}

    repaired, ok = repair_block_feasibility(
        blocks, preferences, talk_presenter, unavailability, timeslots,
        timeslots_by_type=by_type)

    assert ok
    for block in repaired:
        assert check_block_feasibility(block, talk_presenter, unavailability,
                                       timeslots, by_type)[0]
    all_talks = sorted(t for b in repaired for tup in b.tuples for t in tup)
    assert all_talks == sorted(talk_presenter)
