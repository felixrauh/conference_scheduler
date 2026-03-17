"""
Test solver abstraction with Phase 1-like problems.

This test creates a small set partitioning problem similar to Phase 1
and verifies that different solvers produce equivalent solutions.

The test problem:
- 6 talks: T1, T2, T3, T4, T5, T6
- Need to select 2 tuples of size 3 (each tuple has 3 talks running in parallel)
- Preferences create different costs for different tuple combinations
- Expected: unique optimal solution with known objective value
"""

from itertools import combinations
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_gurobi_available() -> bool:
    """Check if Gurobi is installed and licensed."""
    try:
        import gurobipy as gp
        # Try creating a model to verify license
        m = gp.Model()
        m.dispose()
        return True
    except:
        return False


def check_ortools_available() -> bool:
    """Check if OR-Tools is installed."""
    try:
        from ortools.sat.python import cp_model
        return True
    except:
        return False


# Known test problem
# 6 talks need to be partitioned into 2 groups of 3
# Preferences: P1 wants T1,T2; P2 wants T3,T4; P3 wants T5,T6; P4 wants T1,T3,T5
TALKS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
PREFERENCES = {
    'P1': {'T1', 'T2'},
    'P2': {'T3', 'T4'},
    'P3': {'T5', 'T6'},
    'P4': {'T1', 'T3', 'T5'},
}

# All possible 3-tuples
ALL_TUPLES = list(combinations(TALKS, 3))


def compute_tuple_cost(tup: tuple) -> int:
    """Compute missed attendance cost for a tuple."""
    cost = 0
    for participant, prefs in PREFERENCES.items():
        # Count how many preferred talks in this tuple
        overlap = len(set(tup) & prefs)
        # Missed attendance = max(0, overlap - 1)
        if overlap > 1:
            cost += overlap - 1
    return cost


# Pre-compute costs
TUPLE_COSTS = {tup: compute_tuple_cost(tup) for tup in ALL_TUPLES}


def solve_with_gurobi():
    """Solve the test problem using Gurobi directly."""
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("test_phase1")
    model.Params.OutputFlag = 0

    # Binary variable for each tuple
    x = {}
    for tup in ALL_TUPLES:
        x[tup] = model.addVar(vtype=GRB.BINARY, name=f"x_{tup}")

    # Each talk in exactly one selected tuple
    for talk in TALKS:
        tuples_with_talk = [t for t in ALL_TUPLES if talk in t]
        model.addConstr(
            gp.quicksum(x[t] for t in tuples_with_talk) == 1,
            name=f"cover_{talk}"
        )

    # Select exactly 2 tuples
    model.addConstr(
        gp.quicksum(x[t] for t in ALL_TUPLES) == 2,
        name="select_2"
    )

    # Minimize cost
    model.setObjective(
        gp.quicksum(TUPLE_COSTS[t] * x[t] for t in ALL_TUPLES),
        GRB.MINIMIZE
    )

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        selected = [t for t in ALL_TUPLES if x[t].X > 0.5]
        return model.ObjVal, selected
    return None, None


def solve_with_ortools():
    """Solve the test problem using OR-Tools CP-SAT."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()

    # Binary variable for each tuple
    x = {}
    for tup in ALL_TUPLES:
        x[tup] = model.NewBoolVar(f"x_{tup}")

    # Each talk in exactly one selected tuple
    for talk in TALKS:
        tuples_with_talk = [t for t in ALL_TUPLES if talk in t]
        model.Add(sum(x[t] for t in tuples_with_talk) == 1)

    # Select exactly 2 tuples
    model.Add(sum(x[t] for t in ALL_TUPLES) == 2)

    # Minimize cost
    model.Minimize(sum(TUPLE_COSTS[t] * x[t] for t in ALL_TUPLES))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        selected = [t for t in ALL_TUPLES if solver.Value(x[t]) == 1]
        return solver.ObjectiveValue(), selected
    return None, None


class TestSetPartitioning:
    """Test set partitioning (Phase 1-like) problem with different solvers."""

    def test_tuple_costs(self):
        """Verify cost computation is sensible."""
        # T1,T2,T3 should have cost from P1 (has T1,T2) + P4 (has T1,T3)
        cost = TUPLE_COSTS[('T1', 'T2', 'T3')]
        # P1 has 2 preferred in tuple -> +1
        # P4 has 2 preferred in tuple -> +1
        assert cost == 2

    @pytest.mark.skipif(not check_gurobi_available(), reason="Gurobi not available")
    def test_gurobi_solves(self):
        """Verify Gurobi finds a solution."""
        obj, selected = solve_with_gurobi()
        assert obj is not None
        assert len(selected) == 2
        # All talks covered
        all_in_solution = set()
        for tup in selected:
            all_in_solution.update(tup)
        assert all_in_solution == set(TALKS)
        print(f"Gurobi solution: {selected}, objective: {obj}")

    @pytest.mark.skipif(not check_ortools_available(), reason="OR-Tools not available")
    def test_ortools_solves(self):
        """Verify OR-Tools finds a solution."""
        obj, selected = solve_with_ortools()
        assert obj is not None
        assert len(selected) == 2
        # All talks covered
        all_in_solution = set()
        for tup in selected:
            all_in_solution.update(tup)
        assert all_in_solution == set(TALKS)
        print(f"OR-Tools solution: {selected}, objective: {obj}")

    @pytest.mark.skipif(
        not (check_gurobi_available() and check_ortools_available()),
        reason="Both solvers required for equivalence test"
    )
    def test_solver_equivalence(self):
        """Verify both solvers find the same optimal objective value."""
        obj_gurobi, selected_gurobi = solve_with_gurobi()
        obj_ortools, selected_ortools = solve_with_ortools()

        print(f"Gurobi: obj={obj_gurobi}, selected={selected_gurobi}")
        print(f"OR-Tools: obj={obj_ortools}, selected={selected_ortools}")

        # Objectives should be identical for this problem
        assert obj_gurobi == pytest.approx(obj_ortools, abs=0.01)

        # Both should find optimal (but not necessarily same solution due to ties)
        assert obj_gurobi is not None
        assert obj_ortools is not None


# Baseline: expected optimal value established by running Gurobi
# This is the gold-standard result that all solvers should achieve
EXPECTED_OPTIMAL_OBJECTIVE = 1.0  # Established 2026-02-11


class TestBaseline:
    """Establish baseline expected values."""

    @pytest.mark.skipif(not check_gurobi_available(), reason="Gurobi not available")
    def test_gurobi_matches_baseline(self):
        """Verify Gurobi still finds the expected optimal value."""
        obj, selected = solve_with_gurobi()

        assert obj is not None
        assert obj == pytest.approx(EXPECTED_OPTIMAL_OBJECTIVE, abs=0.01)
        print(f"Gurobi matches baseline: {obj}")

    @pytest.mark.skipif(not check_ortools_available(), reason="OR-Tools not available")
    def test_ortools_matches_baseline(self):
        """Verify OR-Tools finds the expected optimal value."""
        obj, selected = solve_with_ortools()

        assert obj is not None
        assert obj == pytest.approx(EXPECTED_OPTIMAL_OBJECTIVE, abs=0.01)
        print(f"OR-Tools matches baseline: {obj}")


if __name__ == "__main__":
    # Run baseline test
    print("Testing set partitioning problem...")
    print(f"Talks: {TALKS}")
    print(f"Preferences: {PREFERENCES}")
    print(f"All possible tuples: {len(ALL_TUPLES)}")
    print(f"Tuple costs: {TUPLE_COSTS}")

    if check_gurobi_available():
        obj, selected = solve_with_gurobi()
        print(f"\nGurobi solution: objective={obj}, selected={selected}")

    if check_ortools_available():
        obj, selected = solve_with_ortools()
        print(f"OR-Tools solution: objective={obj}, selected={selected}")
