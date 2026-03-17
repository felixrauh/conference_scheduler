"""
Phase 1 Conference Scheduling with Column Generation

This module implements a column generation approach to efficiently solve
the Phase 1 scheduling problem, avoiding explicit enumeration of all n-tuples.
"""

import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from typing import Dict, List, Tuple, Set, Union
from dataclasses import dataclass
import time
import numpy as np


# Type definitions
NTuple = Tuple[str, ...]


def compute_tuple_cost(ntuple: NTuple, preferences: Dict[str, Set[str]]) -> int:
    """Compute missed attendance cost for an n-tuple."""
    cost = 0
    for p_id, prefs in preferences.items():
        preferred_count = sum(1 for talk_id in ntuple if talk_id in prefs)
        if preferred_count > 1:
            cost += preferred_count - 1
    return cost


class Phase1ColumnGeneration:
    """
    Column generation solver for Phase 1 conference scheduling.
    
    Efficiently solves large instances by generating only promising n-tuples
    instead of enumerating all possibilities.
    """
    
    def __init__(self, env: gp.Env, talks: List[str], participants: List[str],
                 preferences: Dict[str, Set[str]], tuple_types: List[Tuple[int, int]],
                 verbose: bool = True):
        """
        Initialize column generation solver.
        
        Args:
            env: Gurobi environment
            talks: List of talk IDs
            participants: List of participant IDs
            preferences: Dict mapping participant_id to set of preferred talk_ids
            tuple_types: List of (n_tau, p_tau) tuples
            verbose: Print detailed progress information
        """
        self.env = env
        self.talks = talks
        self.participants = participants
        self.preferences = preferences
        self.tuple_types = tuple_types
        self.verbose = verbose
        
        # Validate feasibility
        self._validate_feasibility()
        
        # Column storage
        self.columns = []  # List of n-tuples (columns)
        self.column_costs = {}  # Dict: n-tuple -> cost
        self.column_sizes = {}  # Dict: n-tuple -> size
        
        # Models
        self.master_model = None
        self.x_vars = {}  # Master problem variables
        
        # Solution tracking
        self.objective_history = []
        self.iteration_count = 0
        self.total_columns_generated = 0
        
        # Statistics
        self.stats = {
            'initialization_time': 0,
            'master_solve_time': 0,
            'pricing_solve_time': 0,
            'total_iterations': 0,
            'columns_added': 0,
            'final_columns': 0
        }
        
    def _validate_feasibility(self):
        """Validate that the problem is feasible."""
        total_slots = sum(n_tau * p_tau for n_tau, p_tau in self.tuple_types)
        if total_slots != len(self.talks):
            raise ValueError(
                f"Infeasible: Total slots ({total_slots}) != talks ({len(self.talks)})"
            )
    
    def solve(self, max_iterations: int = 100, optimality_gap: float = 1e-6,
              time_limit: float = 600) -> Dict:
        """
        Solve Phase 1 using column generation.
        
        Args:
            max_iterations: Maximum number of CG iterations
            optimality_gap: Stop if no column with reduced cost < -gap is found
            time_limit: Maximum time in seconds
            
        Returns:
            Dictionary with solution information
        """
        start_time = time.time()
        
        if self.verbose:
            print("="*70)
            print("PHASE 1 COLUMN GENERATION")
            print("="*70)
            print(f"Problem size: {len(self.talks)} talks, {len(self.participants)} participants")
            print(f"Tuple types: {self.tuple_types}")
        
        # Step 1: Initialize with feasible columns
        init_start = time.time()
        self._initialize_columns()
        self.stats['initialization_time'] = time.time() - init_start
        
        if self.verbose:
            print(f"\nInitialization: {len(self.columns)} initial columns")
            print(f"Time: {self.stats['initialization_time']:.2f}s\n")
        
        # Step 2: Column generation loop
        iteration = 0
        while iteration < max_iterations and time.time() - start_time < time_limit:
            iteration += 1
            self.iteration_count = iteration
            
            if self.verbose:
                print(f"Iteration {iteration}")
                print("-" * 40)
            
            # Solve master problem (LP relaxation)
            master_start = time.time()
            master_obj, duals = self._solve_master_lp()
            self.stats['master_solve_time'] += time.time() - master_start
            self.objective_history.append(master_obj)
            
            if self.verbose:
                print(f"  Master LP objective: {master_obj:.4f}")
            
            # Solve pricing problems
            pricing_start = time.time()
            new_columns = self._solve_pricing_problems(duals, optimality_gap)
            self.stats['pricing_solve_time'] += time.time() - pricing_start
            
            if not new_columns:
                if self.verbose:
                    print(f"  No columns with negative reduced cost found.")
                    print(f"  LP relaxation optimal!\n")
                break
            
            # Add new columns to master
            for col in new_columns:
                self._add_column(col)
                self.total_columns_generated += 1
            
            if self.verbose:
                print(f"  Added {len(new_columns)} new columns")
                print(f"  Total columns: {len(self.columns)}\n")
        
        self.stats['total_iterations'] = iteration
        self.stats['columns_added'] = self.total_columns_generated
        self.stats['final_columns'] = len(self.columns)
        
        # Step 3: Solve final MIP with integer variables
        if self.verbose:
            print("="*70)
            print("SOLVING FINAL INTEGER PROBLEM")
            print("="*70)
        
        final_start = time.time()
        status = self._solve_master_mip()
        final_time = time.time() - final_start
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("COLUMN GENERATION SUMMARY")
            print("="*70)
            print(f"Total time: {total_time:.2f}s")
            print(f"  Initialization: {self.stats['initialization_time']:.2f}s")
            print(f"  Master LP solves: {self.stats['master_solve_time']:.2f}s")
            print(f"  Pricing solves: {self.stats['pricing_solve_time']:.2f}s")
            print(f"  Final MIP: {final_time:.2f}s")
            print(f"\nIterations: {self.stats['total_iterations']}")
            print(f"Columns generated: {self.stats['columns_added']}")
            print(f"Final columns: {self.stats['final_columns']}")
            
            if status == GRB.OPTIMAL:
                print(f"\nOptimal objective: {self.master_model.objVal:.0f}")
        
        return {
            'status': status,
            'objective': self.master_model.objVal if status == GRB.OPTIMAL else None,
            'stats': self.stats,
            'selected_tuples': self._get_selected_tuples() if status == GRB.OPTIMAL else None
        }
    
    def _initialize_columns(self):
        """
        Initialize with a feasible set of columns.
        
        Strategy: Use greedy heuristic to create initial feasible solution.
        """
        remaining_talks = set(self.talks)
        
        for n_tau, p_tau in self.tuple_types:
            for _ in range(p_tau):
                if len(remaining_talks) < n_tau:
                    # Fallback: just take any remaining talks
                    if remaining_talks:
                        ntuple = tuple(sorted(remaining_talks)[:n_tau])
                    else:
                        break
                else:
                    # Greedy: select n_tau talks with minimum cost
                    best_cost = float('inf')
                    best_tuple = None
                    
                    # Try several random samples or use heuristic
                    candidates = list(combinations(remaining_talks, n_tau))
                    
                    # If too many candidates, sample
                    if len(candidates) > 1000:
                        import random
                        candidates = random.sample(candidates, 1000)
                    
                    for candidate in candidates:
                        cost = compute_tuple_cost(candidate, self.preferences)
                        if cost < best_cost:
                            best_cost = cost
                            best_tuple = candidate
                    
                    ntuple = best_tuple
                
                # Add this column
                self._add_column(ntuple)
                
                # Remove talks from remaining
                for talk in ntuple:
                    remaining_talks.discard(talk)
    
    def _add_column(self, ntuple: NTuple):
        """Add a column to the master problem."""
        if ntuple in self.column_costs:
            return  # Already exists
        
        # Compute cost
        cost = compute_tuple_cost(ntuple, self.preferences)
        
        # Store column
        self.columns.append(ntuple)
        self.column_costs[ntuple] = cost
        self.column_sizes[ntuple] = len(ntuple)
        
        # Add variable to master problem if it exists
        if self.master_model is not None:
            var = self.master_model.addVar(
                vtype=GRB.CONTINUOUS,  # Relaxed initially
                lb=0, ub=1,
                obj=cost,
                name=f"x_{'_'.join(ntuple)}"
            )
            self.x_vars[ntuple] = var
            
            # Add to coverage constraints
            for talk in ntuple:
                constr_name = f"coverage_{talk}"
                constr = self.master_model.getConstrByName(constr_name)
                if constr is not None:
                    self.master_model.chgCoeff(constr, var, 1.0)
            
            # Add to tuple count constraint
            size = len(ntuple)
            constr_name = f"tuple_count_n{size}"
            constr = self.master_model.getConstrByName(constr_name)
            if constr is not None:
                self.master_model.chgCoeff(constr, var, 1.0)
            
            self.master_model.update()
    
    def _build_master_problem(self):
        """Build the master problem with current columns."""
        self.master_model = gp.Model("Master_LP", env=self.env)
        self.master_model.setParam('OutputFlag', 0)
        
        # Create variables
        self.x_vars = {}
        for ntuple in self.columns:
            self.x_vars[ntuple] = self.master_model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0, ub=1,
                obj=self.column_costs[ntuple],
                name=f"x_{'_'.join(ntuple)}"
            )
        
        # Coverage constraints
        for talk in self.talks:
            tuples_with_talk = [nt for nt in self.columns if talk in nt]
            self.master_model.addConstr(
                gp.quicksum(self.x_vars[nt] for nt in tuples_with_talk) == 1,
                name=f"coverage_{talk}"
            )
        
        # Tuple count constraints
        for n_tau, p_tau in self.tuple_types:
            tuples_of_size = [nt for nt in self.columns if len(nt) == n_tau]
            self.master_model.addConstr(
                gp.quicksum(self.x_vars[nt] for nt in tuples_of_size) == p_tau,
                name=f"tuple_count_n{n_tau}"
            )
        
        self.master_model.update()
    
    def _solve_master_lp(self) -> Tuple[float, Dict]:
        """
        Solve LP relaxation of master problem.
        
        Returns:
            Tuple of (objective_value, dual_values)
        """
        if self.master_model is None:
            self._build_master_problem()
        
        # Ensure variables are continuous
        for var in self.x_vars.values():
            var.vtype = GRB.CONTINUOUS
        
        self.master_model.optimize()
        
        if self.master_model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Master LP not optimal: status {self.master_model.status}")
        
        # Extract dual values
        duals = {
            'coverage': {},
            'tuple_count': {}
        }
        
        for talk in self.talks:
            constr = self.master_model.getConstrByName(f"coverage_{talk}")
            duals['coverage'][talk] = constr.Pi
        
        for n_tau, p_tau in self.tuple_types:
            constr = self.master_model.getConstrByName(f"tuple_count_n{n_tau}")
            duals['tuple_count'][n_tau] = constr.Pi
        
        return self.master_model.objVal, duals
    
    def _solve_pricing_problems(self, duals: Dict, optimality_gap: float) -> List[NTuple]:
        """
        Solve pricing problems for each tuple type.
        
        Args:
            duals: Dual values from master LP
            optimality_gap: Only add columns with reduced cost < -gap
            
        Returns:
            List of new columns with negative reduced cost
        """
        new_columns = []
        
        for n_tau, p_tau in self.tuple_types:
            # Solve pricing problem for this tuple size
            best_tuple, reduced_cost = self._solve_pricing_for_size(
                n_tau, duals
            )
            
            if reduced_cost < -optimality_gap:
                new_columns.append(best_tuple)
                
                if self.verbose:
                    print(f"    Size {n_tau}: reduced cost = {reduced_cost:.4f}")
        
        return new_columns
    
    def _solve_pricing_for_size(self, n_tau: int, duals: Dict) -> Tuple[NTuple, float]:
        """
        Solve pricing problem for specific tuple size.
        
        Find n-tuple of size n_tau with minimum reduced cost:
        reduced_cost = cost - sum(pi_i for i in tuple) - lambda_tau
        
        Args:
            n_tau: Tuple size
            duals: Dual values
            
        Returns:
            Tuple of (best_ntuple, reduced_cost)
        """
        # Compute adjusted costs for each talk
        talk_weights = {}
        for talk in self.talks:
            talk_weights[talk] = -duals['coverage'][talk]
        
        lambda_tau = duals['tuple_count'][n_tau]
        
        # Enumerate and find best tuple
        # For small n_tau (≤ 5), full enumeration is feasible
        # For larger n_tau, could use heuristics or beam search
        
        best_reduced_cost = float('inf')
        best_tuple = None
        
        # Limit enumeration for very large problems
        max_candidates = 100000
        all_combinations = combinations(self.talks, n_tau)
        
        count = 0
        for candidate in all_combinations:
            count += 1
            if count > max_candidates:
                break
            
            # Skip if already in columns
            if candidate in self.column_costs:
                continue
            
            # Compute cost
            cost = compute_tuple_cost(candidate, self.preferences)
            
            # Compute reduced cost
            reduced_cost = cost + sum(talk_weights[t] for t in candidate) - lambda_tau
            
            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_tuple = candidate
        
        if best_tuple is None:
            # No new tuple found, return dummy with positive reduced cost
            best_tuple = tuple(self.talks[:n_tau])
            best_reduced_cost = float('inf')
        
        return best_tuple, best_reduced_cost
    
    def _solve_master_mip(self) -> int:
        """
        Solve master problem as MIP (with integer variables).
        
        Returns:
            Gurobi status code
        """
        # Convert to integer variables
        for var in self.x_vars.values():
            var.vtype = GRB.BINARY
        
        self.master_model.update()
        self.master_model.setParam('OutputFlag', 1 if self.verbose else 0)
        self.master_model.optimize()
        
        return self.master_model.status
    
    def _get_selected_tuples(self) -> List[NTuple]:
        """Extract selected tuples from solution."""
        if self.master_model.status != GRB.OPTIMAL:
            return None
        
        selected = []
        for ntuple, var in self.x_vars.items():
            if var.X > 0.5:
                selected.append(ntuple)
        
        return selected
    
    def get_result(self) -> List[NTuple]:
        """Get Phase1Result format."""
        return self._get_selected_tuples()
    
    def get_result_by_type(self) -> Dict[int, List[NTuple]]:
        """Get Phase1ResultByType format."""
        selected = self._get_selected_tuples()
        if selected is None:
            return None
        
        result_by_type = {}
        for ntuple in selected:
            size = len(ntuple)
            if size not in result_by_type:
                result_by_type[size] = []
            result_by_type[size].append(ntuple)
        
        return result_by_type
    
    def display_results(self):
        """Display optimization results."""
        print("\n" + "="*70)
        print("COLUMN GENERATION RESULTS")
        print("="*70)
        
        if self.master_model.status == GRB.OPTIMAL:
            print(f"Status: OPTIMAL")
            print(f"Total Missed Attendance: {self.master_model.objVal:.0f}")
            
            selected = self._get_selected_tuples()
            print(f"\nScheduled {len(selected)} parallel sessions:")
            print("-"*70)
            
            result_by_type = self.get_result_by_type()
            for n_tau in sorted(result_by_type.keys()):
                tuples = result_by_type[n_tau]
                print(f"\nTuple Size {n_tau} ({len(tuples)} sessions):")
                
                for idx, ntuple in enumerate(tuples, 1):
                    cost = self.column_costs[ntuple]
                    print(f"  Session {idx}: {', '.join(ntuple)} - Missed: {cost}")
        else:
            print(f"Status: {self.master_model.status}")


# ============================================================================
# COMPARISON UTILITY
# ============================================================================

def compare_approaches(talks: List[str], participants: List[str],
                       preferences: Dict[str, Set[str]], 
                       tuple_types: List[Tuple[int, int]],
                       env: gp.Env):
    """
    Compare explicit enumeration vs column generation.
    
    Args:
        talks, participants, preferences, tuple_types: Problem data
        env: Gurobi environment
    """
    print("="*70)
    print("COMPARING ENUMERATION VS COLUMN GENERATION")
    print("="*70)
    
    # Calculate theoretical number of variables for full enumeration
    total_vars_enum = 0
    for n_tau, p_tau in tuple_types:
        from math import comb
        vars_for_type = comb(len(talks), n_tau)
        total_vars_enum += vars_for_type
    
    print(f"\nProblem size:")
    print(f"  Talks: {len(talks)}")
    print(f"  Participants: {len(participants)}")
    print(f"  Tuple types: {tuple_types}")
    print(f"\nExplicit enumeration would generate: {total_vars_enum:,} variables")
    
    # Column generation approach
    print(f"\n{'='*70}")
    print("COLUMN GENERATION APPROACH")
    print("="*70)
    
    cg_start = time.time()
    cg_solver = Phase1ColumnGeneration(
        env, talks, participants, preferences, tuple_types, verbose=True
    )
    cg_result = cg_solver.solve(max_iterations=100, time_limit=300)
    cg_time = time.time() - cg_start
    
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nColumn Generation:")
    print(f"  Variables generated: {cg_result['stats']['final_columns']:,}")
    print(f"  Percentage of full enumeration: {100*cg_result['stats']['final_columns']/total_vars_enum:.2f}%")
    print(f"  Solution time: {cg_time:.2f}s")
    print(f"  Objective: {cg_result['objective']:.0f}")
    
    return cg_result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Create sample problem
    talks = [f'T{str(i).zfill(3)}' for i in range(1, 21)]  # 20 talks
    participants = [f'P{str(i).zfill(3)}' for i in range(1, 16)]  # 15 participants
    
    # Generate preferences
    import random
    random.seed(42)
    preferences = {}
    for p_id in participants:
        num_prefs = random.randint(3, 7)
        preferences[p_id] = set(random.sample(talks, num_prefs))
    
    # Tuple types
    tuple_types = [
        (4, 3),  # 3 sessions with 4 parallel talks = 12 talks
        (2, 4),  # 4 sessions with 2 parallel talks = 8 talks
    ]
    # Total: 20 talks
    
    # Solve with column generation
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        
        compare_approaches(talks, participants, preferences, tuple_types, env)
