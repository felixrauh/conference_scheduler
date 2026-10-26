"""
Tests for the end-to-end workflow: configuration, data loading, fixed
sessions, placeholders, evaluation and itineraries.

Everything here runs without Gurobi, except where marked.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_schedule  # noqa: E402
from run_schedule import (  # noqa: E402
    SchedulerConfig,
    SchedulerError,
    apply_fixed_block_sessions,
    build_config,
    parse_args,
    run_scheduler,
    strip_jsonc,
)
from src.data_loader import load_from_csv_dir, parse_keywords  # noqa: E402
from src.phase2 import Block  # noqa: E402
from src.phase3 import Phase3Result, RoomAssignment  # noqa: E402
from src.phase4 import FinalSession, ScheduleResult  # noqa: E402
from src.schedule_evaluator import (  # noqa: E402
    ScheduleEvaluator,
    load_availability_from_csv,
    load_keywords_from_csv,
    load_talk_presenters_from_csv,
)
from src.swap_optimization import ViolationType, detect_dummy_violations  # noqa: E402
from src.utils import is_placeholder  # noqa: E402

EXAMPLES = ROOT / "examples"


def write_dataset(directory: Path, talks: str, preferences: str, sessions: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "talks.csv").write_text(talks)
    (directory / "preferences.csv").write_text(preferences)
    (directory / "sessions.csv").write_text(sessions)
    return directory


# =============================================================================
# Configuration
# =============================================================================

def test_strip_jsonc_removes_comments_but_not_string_contents():
    text = """{
        // line comment
        "url": "https://example.org//path",
        "pattern": "/* not a comment */",
        /* block
           comment */
        "list": [1, 2,],
    }"""
    assert json.loads(strip_jsonc(text)) == {
        "url": "https://example.org//path",
        "pattern": "/* not a comment */",
        "list": [1, 2],
    }


def test_shipped_config_file_is_valid():
    config = SchedulerConfig.from_dict(
        run_schedule.load_jsonc(ROOT / "config" / "scheduling_config.jsonc"))
    assert config.validate() == []


def test_invalid_choice_is_rejected():
    assert SchedulerConfig(phase1_method="greedy").validate()
    args = parse_args([])
    args.phase3_method = "simplex"  # bypasses argparse choices, like a config file would
    with pytest.raises(SchedulerError, match="phase3_method"):
        build_config(args)


def test_broken_or_missing_config_file_is_an_error(tmp_path):
    broken = tmp_path / "broken.jsonc"
    broken.write_text('{"pipeline": }')
    with pytest.raises(SchedulerError, match="Could not parse"):
        build_config(parse_args(["--config", str(broken)]))
    with pytest.raises(SchedulerError, match="not found"):
        build_config(parse_args(["--config", str(tmp_path / "missing.jsonc")]))


def test_keyword_options_can_be_set_in_config():
    config = SchedulerConfig.from_dict(
        {"matching_pipeline": {"require_common_keyword": True,
                               "max_keyword_violations": 3}})
    assert config.require_common_keyword is True
    assert config.max_keyword_violations == 3


# =============================================================================
# Data loading
# =============================================================================

def test_loader_keeps_keywords():
    data = load_from_csv_dir(EXAMPLES / "orbel2026")
    assert "keywords" in data.talks.columns
    assert data.talk_keywords()["T001"] == {"Optimization", "Analytics&AI", "Applications"}


def test_keywords_and_titles_are_optional(tmp_path):
    d = write_dataset(
        tmp_path / "data",
        talks="talk_id,presenter_id,title\n1,1,First talk\n2,2,\n",
        preferences="participant_id,talk_id\n1,1\n",
        sessions="session_id,n_rooms,n_talks_per_room\nTA,2,1\n",
    )
    data = load_from_csv_dir(d)
    assert data.talk_keywords() == {}
    assert list(data.talks["title"]) == ["First talk", "T002"]
    assert data.validate() == []


def test_parse_keywords():
    assert parse_keywords("A; B ;;C") == {"A", "B", "C"}
    assert parse_keywords(float("nan")) == set()
    assert parse_keywords("") == set()


def test_block_types_override_expands_count():
    data = load_from_csv_dir(EXAMPLES / "synthetic",
                             block_types={"2R4T": {"n": 2, "k": 4, "count": 2}})
    assert [ts["id"] for ts in data.timeslots] == ["2R4T_1", "2R4T_2"]
    assert data.block_types == {"2R4T": {"n": 2, "k": 4, "count": 2}}
    assert data.total_talk_slots() == 16


def test_more_talks_than_slots_is_an_error(tmp_path):
    d = write_dataset(
        tmp_path / "data",
        talks="talk_id,presenter_id\n1,1\n2,2\n3,3\n",
        preferences="participant_id,talk_id\n1,1\n",
        sessions="session_id,n_rooms,n_talks_per_room\nTA,2,1\n",
    )
    data = load_from_csv_dir(d)
    assert any("Too many talks" in e for e in data.validate())

    config = SchedulerConfig(pipeline="heuristic", data_dir=str(d),
                             output_file=str(tmp_path / "out.csv"))
    with pytest.raises(SchedulerError, match="only 2 slots"):
        run_scheduler(config)


def test_more_slots_than_talks_is_not_an_error():
    data = load_from_csv_dir(EXAMPLES / "orbel2026")  # 123 slots, 118 talks
    assert data.validate() == []


# =============================================================================
# Fixed block sessions
# =============================================================================

def test_fixed_block_sessions_reserve_rooms():
    data = load_from_csv_dir(EXAMPLES / "orbel2026")
    data.validate()
    slots_before = data.total_talk_slots()

    fixed = apply_fixed_block_sessions(data, [
        {"name": "A", "block": "FA", "talks": ["T001", "T002", "T003", "T004"]},
        {"name": "B", "block": "FA", "talks": ["T005", "T006", "T007", "T008"]},
    ])

    assert fixed == {f"T00{i}" for i in range(1, 9)}
    fa = next(ts for ts in data.timeslots if ts["id"] == "FA")
    assert fa["type_id"] == "3R4T" and len(fa["rooms"]) == 3
    assert data.total_talk_slots() == slots_before - 8
    # Block type counts still match the timeslots
    for type_id, bt in data.block_types.items():
        assert bt["count"] == sum(ts["type_id"] == type_id for ts in data.timeslots)


@pytest.mark.parametrize("session, message", [
    ({"name": "X", "block": "ZZ", "talks": ["T001"]}, "not a session_id"),
    ({"name": "X", "block": "TA", "talks": ["T999"]}, "unknown talk"),
    ({"name": "X", "block": "TC", "talks": ["T001", "T002", "T003", "T004"]},
     "only has 3 talks per room"),
])
def test_invalid_fixed_block_sessions_are_reported(session, message):
    data = load_from_csv_dir(EXAMPLES / "orbel2026")
    with pytest.raises(SchedulerError, match=message):
        apply_fixed_block_sessions(data, [session])


# =============================================================================
# End to end (heuristic pipeline, no Gurobi needed)
# =============================================================================

def test_heuristic_end_to_end_with_fixed_session(tmp_path):
    output = tmp_path / "schedule.csv"
    config = SchedulerConfig(
        pipeline="heuristic",
        data_dir=str(EXAMPLES / "orbel2026"),
        output_file=str(output),
        phase4_verbose=False,
        fixed_block_sessions=[
            {"name": "Award", "block": "FA", "talks": ["T001", "T002", "T003", "T004"]}],
    )

    results = run_scheduler(config)

    df = pd.read_csv(output)
    real = df[~df["Talk_ID"].map(is_placeholder)]
    assert sorted(real["Talk_ID"]) == [f"T{i:03d}" for i in range(1, 119)]
    assert df["Talk_ID"].is_unique
    # Blocks follow sessions.csv order
    assert list(dict.fromkeys(df["Block_ID"])) == ["TA", "TB", "TC", "TD", "FA", "FB", "FC"]
    # The fixed session is in FA, in one room, in order
    fixed = df[df["Is_Fixed"]]
    assert set(fixed["Block_ID"]) == {"FA"} and fixed["Room_ID"].nunique() == 1
    assert list(fixed.sort_values("Slot")["Talk_ID"]) == ["T001", "T002", "T003", "T004"]
    # No placeholder in a short (3-talk) block, at most one per room-session
    placeholders = df[df["Talk_ID"].map(is_placeholder)]
    assert not placeholders["Block_ID"].isin(["TC", "TD"]).any()
    assert placeholders.groupby(["Block_ID", "Room_ID"]).size().max() <= 1
    # Keywords reach the evaluator
    metrics = results["evaluation"]
    assert metrics is not None and metrics.total_talks == 118
    assert output.with_suffix(".md").exists()


# =============================================================================
# Reproducibility
# =============================================================================

def test_greedy_phase1_does_not_depend_on_set_order():
    """The greedy construction gives the same tuples under different hash seeds."""
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from src.data_loader import load_from_csv_dir\n"
        "from src.instance import build_instance\n"
        "from src.pipelines.heuristic import _greedy_phase1\n"
        "d = load_from_csv_dir('examples/orbel2026'); d.validate(); i = build_instance(d)\n"
        "print(sorted(_greedy_phase1(i.talks, i.preferences, [(5, 16), (4, 11)]).items()))\n"
    )
    outputs = set()
    for hash_seed in ["1", "2", "3"]:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                capture_output=True, text=True, check=True)
        outputs.add(result.stdout.strip().splitlines()[-1])
    assert len(outputs) == 1


def test_same_seed_gives_same_schedule(tmp_path):
    schedules = []
    for run in range(2):
        output = tmp_path / f"schedule_{run}.csv"
        run_scheduler(SchedulerConfig(
            pipeline="heuristic", data_dir=str(EXAMPLES / "orbel2017"),
            output_file=str(output), phase4_verbose=False, seed=7))
        schedules.append(output.read_text())
    assert schedules[0] == schedules[1]


def test_seed_must_be_an_integer():
    assert SchedulerConfig(seed="abc").validate()
    assert SchedulerConfig.from_dict({"seed": 3}).seed == 3


# =============================================================================
# Placeholders, evaluation and itineraries
# =============================================================================

def test_infeasible_pairs_with_few_timeslots():
    """With 2 timeslots, presenters unavailable in TA and TB can't share a tuple."""
    from src.instance import build_instance, compute_infeasible_talk_pairs

    data = load_from_csv_dir(EXAMPLES / "synthetic")  # P001: not TB, P002: not TA
    data.validate()
    assert compute_infeasible_talk_pairs(build_instance(data)) == {("T001", "T002")}


def test_swap_optimization_recognizes_pipeline_placeholders():
    block = Block("B01", "2R3T", [("T001", "PLACEHOLDER_1"), ("T002", "T003"),
                                  ("T004", "T005")], 0)
    phase3 = Phase3Result(
        assignments=[RoomAssignment(block, {"id": "TA"}, {0: "R1", 1: "R2"}, 0)],
        total_violations=0, total_capacity_gap=0)

    violations = detect_dummy_violations(phase3, short_block_threshold=3)

    assert [v.talk_id for v in violations] == ["PLACEHOLDER_1"]
    assert violations[0].violation_type == ViolationType.DUMMY_IN_SHORT_BLOCK


def test_coherence_ignores_placeholders():
    schedule = pd.DataFrame([
        {"Session_ID": "TA", "Block_ID": "TA", "Slot": s, "Room": "R1", "Talk_ID": t}
        for s, t in [(1, "T001"), (2, "T002"), (3, "PLACEHOLDER_1")]
    ])
    keywords = {"T001": {"A", "B"}, "T002": {"B"}}
    evaluator = ScheduleEvaluator(schedule, preferences={}, talk_keywords=keywords)
    assert evaluator.compute_incoherent_sessions()[0] == 0


def test_evaluator_loaders_read_standard_input_format(tmp_path):
    keywords, _ = load_keywords_from_csv(str(EXAMPLES / "orbel2026" / "talks.csv"))
    assert keywords["T002"] == {"Analytics&AI", "Applications"}

    presenters = load_talk_presenters_from_csv(str(EXAMPLES / "orbel2026" / "talks.csv"))
    assert presenters["T010"] == "P010"

    availability = tmp_path / "availability.csv"
    availability.write_text("presenter_id,unavailable_timeslot\n7,TA\nP008,TB\n")
    assert load_availability_from_csv(str(availability)) == {"P007": {"TA"}, "P008": {"TB"}}


def _session(block, room, slot, talk):
    return FinalSession(block_id=block, room_id=room, room_name=room, slot=slot,
                        talk_id=talk, paper_id=0, title=talk,
                        primary_contact_author="", author_names="",
                        total_likes=0, unique_attendees=0, is_fixed=False)


def test_itinerary_counts_room_switches_within_blocks_only():
    result = ScheduleResult(
        sessions=[
            _session("TA", "R1", 1, "T1"), _session("TA", "R2", 1, "T2"),
            _session("TA", "R1", 2, "T3"), _session("TA", "R2", 2, "T4"),
            _session("FA", "R1", 1, "T5"), _session("FA", "R2", 1, "T6"),
        ],
        session_metrics=[], fixed_sessions_added=[])

    # T2 (R2) -> T3 (R1) is a switch within TA; moving on to FA is not
    itinerary = result.generate_personal_itinerary("p", {"p": {"T2", "T3", "T5"}})

    assert "**Room switches**: 1" in itinerary
    assert itinerary.index("Block TA") < itinerary.index("Block FA")


# =============================================================================
# Gurobi-specific
# =============================================================================

def test_column_generation_with_zero_time_limit():
    gp = pytest.importorskip("gurobipy")
    from src.columngeneration_phase1 import Phase1ColumnGenerationEnhanced
    from src.instance import build_instance

    data = load_from_csv_dir(EXAMPLES / "synthetic")
    data.validate()
    instance = build_instance(data)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        solver = Phase1ColumnGenerationEnhanced(
            env, instance.talks, instance.participants, instance.preferences,
            [(2, 8)], verbose=False)
        result = solver.solve(time_limit=0)

    assert result["status"] == gp.GRB.OPTIMAL
    assert result["stats"]["converged"] is False


def test_column_generation_is_never_worse_than_its_seed():
    gp = pytest.importorskip("gurobipy")
    from src.columngeneration_phase1 import Phase1ColumnGenerationEnhanced, compute_tuple_cost
    from src.instance import build_instance

    data = load_from_csv_dir(EXAMPLES / "synthetic")
    data.validate()
    instance = build_instance(data)
    talks = instance.talks
    seed = [tuple(talks[i:i + 2]) for i in range(0, len(talks), 2)]
    seed_cost = sum(compute_tuple_cost(t, instance.preferences) for t in seed)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with pytest.raises(ValueError, match="Invalid initial column"):
            Phase1ColumnGenerationEnhanced(
                env, talks, instance.participants, instance.preferences,
                [(2, 8)], initial_columns=[("T001", "T002", "T003")])

        solver = Phase1ColumnGenerationEnhanced(
            env, talks, instance.participants, instance.preferences,
            [(2, 8)], verbose=False, initial_columns=seed)
        result = solver.solve(time_limit=0)

    assert result["objective"] <= seed_cost
