"""
Advanced Column Generation for Phase 1 - Enhanced Pricing Strategies

This module provides improved pricing problem solvers with multiple strategies:
1. Full enumeration (for small instances)
2. Greedy construction heuristic
3. Local search improvement
4. Beam search for large instances
"""

import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from typing import Dict, List, Tuple, Set
import random
import heapq


def compute_tuple_cost(ntuple: Tuple[str, ...], preferences: Dict[str, Set[str]]) -> int:
    """Compute missed attendance cost for an n-tuple."""
    cost = 0
    for p_id, prefs in preferences.items():
        preferred_count = sum(1 for talk_id in ntuple if talk_id in prefs)
        if preferred_count > 1:
            cost += preferred_count - 1
    return cost


class PricingProblemSolver:
    """
    Sophisticated pricing problem solver with multiple strategies.
    """

    def __init__(self, talks: List[str], preferences: Dict[str, Set[str]],
                 existing_columns: Set[Tuple[str, ...]]):
        """
        Initialize pricing solver.

        Args:
            talks: List of all talk IDs
            preferences: Participant preferences
            existing_columns: Set of already generated columns (to avoid duplicates)
        """
        self.talks = talks
        self.preferences = preferences
        self.existing_columns = existing_columns

        # Precompute talk-level information
        self._precompute_talk_metrics()

    def _precompute_talk_metrics(self):
        """Precompute useful metrics for each talk."""
        self.talk_popularity = {}  # Number of participants interested
        self.talk_conflicts = {}  # Which other talks conflict with this one

        for talk in self.talks:
            self.talk_popularity[talk] = sum(
                1 for prefs in self.preferences.values() if talk in prefs
            )

            self.talk_conflicts[talk] = set()
            for p_id, prefs in self.preferences.items():
                if talk in prefs:
                    self.talk_conflicts[talk].update(prefs - {talk})

    def solve(self, n_tau: int, talk_weights: Dict[str, float],
              lambda_tau: float, strategy: str = 'auto') -> Tuple[Tuple[str, ...], float]:
        """
        Solve pricing problem for given tuple size.

        Args:
            n_tau: Tuple size
            talk_weights: Modified costs for each talk (negative of dual prices)
            lambda_tau: Dual price for tuple count constraint
            strategy: 'enumeration', 'greedy', 'local_search', 'beam_search', 'auto'

        Returns:
            Tuple of (best_ntuple, reduced_cost)
        """
        if strategy == 'auto':
            # Choose strategy based on problem size
            from math import comb
            if comb(len(self.talks), n_tau) <= 10000:
                strategy = 'enumeration'
            elif n_tau <= 5:
                strategy = 'local_search'
            else:
                strategy = 'beam_search'

        if strategy == 'enumeration':
            return self._solve_by_enumeration(n_tau, talk_weights, lambda_tau)
        elif strategy == 'greedy':
            return self._solve_by_greedy(n_tau, talk_weights, lambda_tau)
        elif strategy == 'local_search':
            return self._solve_by_local_search(n_tau, talk_weights, lambda_tau)
        elif strategy == 'beam_search':
            return self._solve_by_beam_search(n_tau, talk_weights, lambda_tau)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _solve_by_enumeration(self, n_tau: int, talk_weights: Dict[str, float],
                              lambda_tau: float) -> Tuple[Tuple[str, ...], float]:
        """Full enumeration of all possible n-tuples."""
        best_reduced_cost = float('inf')
        best_tuple = None

        for candidate in combinations(self.talks, n_tau):
            if candidate in self.existing_columns:
                continue

            cost = compute_tuple_cost(candidate, self.preferences)
            reduced_cost = cost + \
                sum(talk_weights[t] for t in candidate) - lambda_tau

            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_tuple = candidate

        if best_tuple is None:
            best_tuple = tuple(self.talks[:n_tau])
            best_reduced_cost = float('inf')

        return best_tuple, best_reduced_cost

    def _solve_by_greedy(self, n_tau: int, talk_weights: Dict[str, float],
                         lambda_tau: float) -> Tuple[Tuple[str, ...], float]:
        """
        Greedy construction heuristic.

        Iteratively add talks with best marginal reduced cost.
        """
        selected = []
        remaining = set(self.talks)

        # Precompute interaction costs between talks
        def marginal_cost(talk: str, current_selection: List[str]) -> float:
            """Cost of adding 'talk' to current selection."""
            candidate = tuple(current_selection + [talk])
            cost = compute_tuple_cost(candidate, self.preferences)

            # Subtract cost of current selection
            if current_selection:
                current_cost = compute_tuple_cost(
                    tuple(current_selection), self.preferences)
                marginal = cost - current_cost
            else:
                marginal = cost

            return marginal + talk_weights[talk]

        # Greedy selection
        for _ in range(n_tau):
            best_marginal = float('inf')
            best_talk = None

            for talk in remaining:
                marginal = marginal_cost(talk, selected)
                if marginal < best_marginal:
                    best_marginal = marginal
                    best_talk = talk

            selected.append(best_talk)
            remaining.remove(best_talk)

        best_tuple = tuple(sorted(selected))
        cost = compute_tuple_cost(best_tuple, self.preferences)
        reduced_cost = cost + sum(talk_weights[t]
                                  for t in best_tuple) - lambda_tau

        return best_tuple, reduced_cost

    def _solve_by_local_search(self, n_tau: int, talk_weights: Dict[str, float],
                               lambda_tau: float, max_iterations: int = 100) -> Tuple[Tuple[str, ...], float]:
        """
        Local search improvement starting from greedy solution.

        Uses swap and replace neighborhoods.
        """
        # Start with greedy solution
        current_tuple, current_reduced_cost = self._solve_by_greedy(
            n_tau, talk_weights, lambda_tau
        )
        current_set = set(current_tuple)

        def evaluate(ntuple: Tuple[str, ...]) -> float:
            """Evaluate reduced cost of tuple."""
            cost = compute_tuple_cost(ntuple, self.preferences)
            return cost + sum(talk_weights[t] for t in ntuple) - lambda_tau

        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try replacing each talk
            for talk_in in current_set:
                for talk_out in self.talks:
                    if talk_out in current_set:
                        continue

                    # Create neighbor by swapping
                    neighbor_set = current_set - {talk_in} | {talk_out}
                    neighbor_tuple = tuple(sorted(neighbor_set))

                    if neighbor_tuple in self.existing_columns:
                        continue

                    neighbor_reduced_cost = evaluate(neighbor_tuple)

                    if neighbor_reduced_cost < current_reduced_cost:
                        current_tuple = neighbor_tuple
                        current_set = neighbor_set
                        current_reduced_cost = neighbor_reduced_cost
                        improved = True
                        break

                if improved:
                    break

        return current_tuple, current_reduced_cost

    def _solve_by_beam_search(self, n_tau: int, talk_weights: Dict[str, float],
                              lambda_tau: float, beam_width: int = 10) -> Tuple[Tuple[str, ...], float]:
        """
        Beam search for large instances.

        Maintains beam_width best partial solutions at each level.
        """
        def evaluate_partial(partial: Tuple[str, ...]) -> float:
            """Evaluate partial tuple (lower bound on reduced cost)."""
            if len(partial) == n_tau:
                cost = compute_tuple_cost(partial, self.preferences)
                return cost + sum(talk_weights[t] for t in partial) - lambda_tau
            else:
                # Lower bound: assume remaining talks have zero marginal cost
                cost = compute_tuple_cost(partial, self.preferences)
                return cost + sum(talk_weights[t] for t in partial)

        # Initialize beam with empty tuple
        beam = [(0, tuple())]  # (score, partial_tuple)

        # Extend beam level by level
        for level in range(n_tau):
            candidates = []

            for score, partial in beam:
                used_talks = set(partial)

                # Try adding each unused talk
                for talk in self.talks:
                    if talk in used_talks:
                        continue

                    extended = partial + (talk,)
                    extended_score = evaluate_partial(extended)
                    candidates.append((extended_score, extended))

            # Keep best beam_width candidates
            candidates.sort()
            beam = candidates[:beam_width]

        # Best complete solution
        best_tuple = beam[0][1]
        best_tuple = tuple(sorted(best_tuple))

        cost = compute_tuple_cost(best_tuple, self.preferences)
        reduced_cost = cost + sum(talk_weights[t]
                                  for t in best_tuple) - lambda_tau

        return best_tuple, reduced_cost


class Phase1ColumnGenerationEnhanced:
    """
    Enhanced column generation with improved pricing strategies.
    """

    def __init__(self, env: gp.Env, talks: List[str], participants: List[str],
                 preferences: Dict[str, Set[str]], tuple_types: List[Tuple[int, int]],
                 pricing_strategy: str = 'auto', verbose: bool = True):
        """
        Initialize enhanced column generation solver.

        Args:
            env: Gurobi environment
            talks: List of talk IDs
            participants: List of participant IDs
            preferences: Dict mapping participant_id to set of preferred talk_ids
            tuple_types: List of (n_tau, p_tau) tuples
            pricing_strategy: 'auto', 'enumeration', 'greedy', 'local_search', 'beam_search'
            verbose: Print detailed progress information
        """
        self.env = env
        self.talks = talks
        self.participants = participants
        self.preferences = preferences
        self.tuple_types = tuple_types
        self.pricing_strategy = pricing_strategy
        self.verbose = verbose

        # Validate feasibility
        total_slots = sum(n_tau * p_tau for n_tau, p_tau in tuple_types)
        if total_slots != len(talks):
            raise ValueError(
                f"Infeasible: Total slots ({total_slots}) != talks ({len(talks)})"
            )

        # Column storage
        self.columns = []
        self.column_costs = {}
        self.column_sizes = {}
        self.column_set = set()  # For fast lookup

        # Models
        self.master_model = None
        self.x_vars = {}

        # Pricing solvers (one per tuple size)
        self.pricing_solvers = {}

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
            'final_columns': 0,
            'pricing_strategy': pricing_strategy
        }

    def solve(self, max_iterations: int = 100, optimality_gap: float = 1e-6,
              time_limit: float = 600) -> Dict:
        """Solve Phase 1 using enhanced column generation."""
        import time
        start_time = time.time()

        if self.verbose:
            print("="*70)
            print("ENHANCED COLUMN GENERATION")
            print("="*70)
            print(
                f"Problem: {len(self.talks)} talks, {len(self.participants)} participants")
            print(f"Pricing strategy: {self.pricing_strategy}")

        # Initialize
        init_start = time.time()
        self._initialize_columns()
        self.stats['initialization_time'] = time.time() - init_start

        # Initialize pricing solvers
        for n_tau, _ in self.tuple_types:
            self.pricing_solvers[n_tau] = PricingProblemSolver(
                self.talks, self.preferences, self.column_set
            )

        if self.verbose:
            print(
                f"\nInitialization: {len(self.columns)} columns in {self.stats['initialization_time']:.2f}s\n")

        # Column generation loop
        iteration = 0
        while iteration < max_iterations and time.time() - start_time < time_limit:
            iteration += 1
            self.iteration_count = iteration

            if self.verbose:
                print(f"Iteration {iteration}")
                print("-" * 40)

            # Solve master LP
            master_start = time.time()
            master_obj, duals = self._solve_master_lp()
            self.stats['master_solve_time'] += time.time() - master_start
            self.objective_history.append(master_obj)

            if self.verbose:
                print(f"  Master objective: {master_obj:.4f}")

            # Solve pricing problems
            pricing_start = time.time()
            new_columns = self._solve_pricing_problems_enhanced(
                duals, optimality_gap)
            self.stats['pricing_solve_time'] += time.time() - pricing_start

            if not new_columns:
                if self.verbose:
                    print(f"  No improving columns found. Optimal!\n")
                break

            # Add columns
            for col in new_columns:
                self._add_column(col)
                self.total_columns_generated += 1

            if self.verbose:
                print(
                    f"  Added {len(new_columns)} columns (Total: {len(self.columns)})\n")

        converged = not new_columns if iteration <= max_iterations else False
        stopped_reason = None
        if not converged:
            if time.time() - start_time >= time_limit:
                stopped_reason = "time limit"
            else:
                stopped_reason = "iteration limit"

        self.stats['total_iterations'] = iteration
        self.stats['columns_added'] = self.total_columns_generated
        self.stats['final_columns'] = len(self.columns)
        self.stats['converged'] = converged

        # Record final LP value from restricted master.
        # IMPORTANT: This is NOT a valid lower bound on the true IP optimum
        # unless CG converged with exact pricing. The restricted master LP
        # has fewer columns, so for minimization z_LP(restricted) >= z_LP(full).
        # Only at convergence (no negative RC) does z_LP(restricted) = z_LP(full).
        final_lp_value = self.objective_history[-1] if self.objective_history else None

        # Solve final MIP
        if self.verbose:
            print("="*70)
            print("SOLVING FINAL MIP")
            print("="*70)

        final_start = time.time()
        status = self._solve_master_mip()
        final_time = time.time() - final_start

        total_time = time.time() - start_time

        ip_objective = self.master_model.objVal if status == GRB.OPTIMAL else None

        # Only with exact pricing + convergence do we have a proven bound
        exact_pricing = self.pricing_strategy == 'enumeration'
        has_proven_bound = converged and exact_pricing
        lp_ip_gap = None
        if final_lp_value is not None and ip_objective is not None and final_lp_value > 0:
            lp_ip_gap = (ip_objective - final_lp_value) / final_lp_value

        if self.verbose:
            print(f"\n{'='*70}")
            print("SUMMARY")
            print("="*70)
            print(f"Total time: {total_time:.2f}s")
            print(f"Iterations: {iteration}")
            print(f"Columns: {len(self.columns)}")
            if converged:
                print(f"Convergence: YES (no improving columns found)")
            else:
                print(f"Convergence: NO (stopped due to {stopped_reason})")
            if ip_objective is not None:
                print(f"IP solution: {ip_objective:.0f}")
            if final_lp_value is not None:
                print(f"Restricted master LP: {final_lp_value:.2f}")
            if has_proven_bound and lp_ip_gap is not None:
                print(f"LP-IP gap: {lp_ip_gap*100:.2f}% (proven bound)")
                print(f"  True optimal is between "
                      f"{final_lp_value:.1f} and {ip_objective:.0f}")
            elif lp_ip_gap is not None:
                print(f"LP-IP gap: {lp_ip_gap*100:.2f}% (NOT a proven bound)")
                if not converged:
                    print(f"  CG did not converge; restricted LP "
                          f"is not a valid lower bound")
                elif not exact_pricing:
                    print(f"  Heuristic pricing may have missed "
                          f"negative reduced cost columns")

        return {
            'status': status,
            'objective': ip_objective,
            'stats': self.stats,
            'selected_tuples': (self._get_selected_tuples()
                                if status == GRB.OPTIMAL else None),
            'converged': converged,
            'has_proven_bound': has_proven_bound,
            'restricted_lp_value': final_lp_value,
            'lp_ip_gap': lp_ip_gap if has_proven_bound else None,
            'lp_value_history': list(self.objective_history),
        }

    def _initialize_columns(self):
        """Initialize with greedy heuristic."""
        remaining = set(self.talks)

        for n_tau, p_tau in self.tuple_types:
            for _ in range(p_tau):
                if len(remaining) < n_tau:
                    break

                # Greedy: select low-cost tuple
                best_cost = float('inf')
                best_tuple = None

                # Sample candidates
                candidates = list(combinations(remaining, n_tau))
                if len(candidates) > 500:
                    import random
                    candidates = random.sample(candidates, 500)

                for cand in candidates:
                    cost = compute_tuple_cost(cand, self.preferences)
                    if cost < best_cost:
                        best_cost = cost
                        best_tuple = cand

                self._add_column(best_tuple)
                for talk in best_tuple:
                    remaining.discard(talk)

    def _add_column(self, ntuple: Tuple[str, ...]):
        """Add column to master."""
        if ntuple in self.column_set:
            return

        cost = compute_tuple_cost(ntuple, self.preferences)

        self.columns.append(ntuple)
        self.column_costs[ntuple] = cost
        self.column_sizes[ntuple] = len(ntuple)
        self.column_set.add(ntuple)

        if self.master_model is not None:
            var = self.master_model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=1, obj=cost,
                name=f"x_{'_'.join(ntuple)}"
            )
            self.x_vars[ntuple] = var

            for talk in ntuple:
                constr = self.master_model.getConstrByName(f"coverage_{talk}")
                if constr:
                    self.master_model.chgCoeff(constr, var, 1.0)

            size = len(ntuple)
            constr = self.master_model.getConstrByName(f"tuple_count_n{size}")
            if constr:
                self.master_model.chgCoeff(constr, var, 1.0)

            self.master_model.update()

    def _build_master_problem(self):
        """Build master problem."""
        self.master_model = gp.Model("Master", env=self.env)
        self.master_model.setParam('OutputFlag', 0)

        self.x_vars = {}
        for ntuple in self.columns:
            self.x_vars[ntuple] = self.master_model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=1,
                obj=self.column_costs[ntuple],
                name=f"x_{'_'.join(ntuple)}"
            )

        for talk in self.talks:
            tuples_with = [nt for nt in self.columns if talk in nt]
            self.master_model.addConstr(
                gp.quicksum(self.x_vars[nt] for nt in tuples_with) == 1,
                name=f"coverage_{talk}"
            )

        for n_tau, p_tau in self.tuple_types:
            tuples_of_size = [nt for nt in self.columns if len(nt) == n_tau]
            self.master_model.addConstr(
                gp.quicksum(self.x_vars[nt] for nt in tuples_of_size) == p_tau,
                name=f"tuple_count_n{n_tau}"
            )

        self.master_model.update()

    def _solve_master_lp(self) -> Tuple[float, Dict]:
        """Solve master LP."""
        if self.master_model is None:
            self._build_master_problem()

        for var in self.x_vars.values():
            var.vtype = GRB.CONTINUOUS

        self.master_model.optimize()

        if self.master_model.status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Master not optimal: {self.master_model.status}")

        duals = {'coverage': {}, 'tuple_count': {}}

        for talk in self.talks:
            constr = self.master_model.getConstrByName(f"coverage_{talk}")
            duals['coverage'][talk] = constr.Pi

        for n_tau, _ in self.tuple_types:
            constr = self.master_model.getConstrByName(f"tuple_count_n{n_tau}")
            duals['tuple_count'][n_tau] = constr.Pi

        return self.master_model.objVal, duals

    def _solve_pricing_problems_enhanced(self, duals: Dict, gap: float) -> List[Tuple[str, ...]]:
        """Solve pricing with enhanced strategies."""
        new_columns = []

        for n_tau, _ in self.tuple_types:
            talk_weights = {talk: -duals['coverage'][talk]
                            for talk in self.talks}
            lambda_tau = duals['tuple_count'][n_tau]

            solver = self.pricing_solvers[n_tau]
            best_tuple, reduced_cost = solver.solve(
                n_tau, talk_weights, lambda_tau, strategy=self.pricing_strategy
            )

            if reduced_cost < -gap:
                new_columns.append(best_tuple)
                if self.verbose:
                    print(f"    Size {n_tau}: RC={reduced_cost:.4f}")

        return new_columns

    def _solve_master_mip(self) -> int:
        """Solve final MIP."""
        # Build master model if not built yet (e.g., time limit during init)
        if self.master_model is None:
            self._build_master_problem()

        for var in self.x_vars.values():
            var.vtype = GRB.BINARY

        self.master_model.update()
        self.master_model.setParam('OutputFlag', 1 if self.verbose else 0)
        self.master_model.optimize()

        return self.master_model.status

    def _get_selected_tuples(self) -> List[Tuple[str, ...]]:
        """Extract solution."""
        if self.master_model.status != GRB.OPTIMAL:
            return None

        return [nt for nt, var in self.x_vars.items() if var.X > 0.5]

    def get_result(self) -> List[Tuple[str, ...]]:
        """Get Phase1Result."""
        return self._get_selected_tuples()

    def get_result_by_type(self) -> Dict[int, List[Tuple[str, ...]]]:
        """Get Phase1ResultByType."""
        selected = self._get_selected_tuples()
        if selected is None:
            return None

        by_type = {}
        for nt in selected:
            size = len(nt)
            if size not in by_type:
                by_type[size] = []
            by_type[size].append(nt)

        return by_type


if __name__ == "__main__":
    # Test with larger instance
    import random
    random.seed(42)

    talks = [f'T{i:03d}' for i in range(1, 31)]  # 30 talks
    participants = [f'P{i:03d}' for i in range(1, 21)]  # 20 participants

    preferences = {}
    for p_id in participants:
        num_prefs = random.randint(4, 8)
        preferences[p_id] = set(random.sample(talks, num_prefs))

    tuple_types = [(5, 3), (3, 5)]  # 15 + 15 = 30 talks

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()

        print("\nTesting enhanced pricing with local search:")
        print("="*70)

        solver = Phase1ColumnGenerationEnhanced(
            env, talks, participants, preferences, tuple_types,
            pricing_strategy='local_search', verbose=True
        )
        result = solver.solve(max_iterations=50, time_limit=120)

        print(f"\nFinal result: {result['objective']:.0f} missed attendances")
        print(f"Columns generated: {result['stats']['final_columns']}")
