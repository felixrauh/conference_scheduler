"""
Test Phase 2 on ORBEL 2017 data using Phase 1 results.

Supports variable n (different block types with different numbers of rooms).
"""

import csv
from src.phase2 import Phase2Input, solve_phase2
from src.phase1 import Phase1Optimizer
from src.instance import build_instance
from src.data_loader import load_from_csv_dir
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# SLOT ADJUSTMENT LOGIC
# =============================================================================

def adjust_block_types_for_talks(block_types: dict, num_talks: int) -> dict:
    """
    Adjust block types to match the number of talks.

    Rules:
    - If slots - talks = 1 or 2: Add dummy talks (caller handles this)
    - If slots - talks >= 3: Reduce block configuration

    Returns adjusted block_types and number of dummy talks needed.
    """
    def count_slots(bt):
        return sum(b['n'] * b['k'] * b['count'] for b in bt.values())

    total_slots = count_slots(block_types)
    diff = total_slots - num_talks

    print(
        f"\nSlot adjustment: {total_slots} slots for {num_talks} talks (diff = {diff})")

    if diff == 0:
        print("  Perfect fit - no adjustment needed")
        return block_types, 0

    if diff < 0:
        raise ValueError(
            f"Not enough slots: {total_slots} slots < {num_talks} talks. Add more blocks.")

    if diff <= 2:
        print(f"  Adding {diff} dummy talk(s)")
        return block_types, diff

    # diff >= 3: Need to reduce configuration
    print(f"  Reducing configuration (removing {diff} slots)")
    adjusted = {k: v.copy() for k, v in block_types.items()}
    remaining = diff

    # Strategy: Try to remove timeslots first, then rooms
    # Sort block types by k (timeslots) descending - prefer shortening longer blocks
    sorted_types = sorted(
        adjusted.keys(), key=lambda x: adjusted[x]['k'], reverse=True)

    for bt_name in sorted_types:
        bt = adjusted[bt_name]
        # Can we remove a timeslot from one block of this type?
        if bt['k'] > 1 and bt['count'] > 0:
            slots_removed = bt['n']  # removing 1 timeslot removes n slots
            if slots_removed <= remaining:
                # Create a new variant with one fewer timeslot
                new_k = bt['k'] - 1
                new_name = f"{bt['n']}R{new_k}T"

                if remaining >= slots_removed:
                    # Reduce count by 1, add one with shorter k
                    adjusted[bt_name]['count'] -= 1
                    if adjusted[bt_name]['count'] == 0:
                        del adjusted[bt_name]

                    if new_k > 0:
                        if new_name in adjusted:
                            adjusted[new_name]['count'] += 1
                        else:
                            adjusted[new_name] = {
                                'n': bt['n'], 'k': new_k, 'count': 1}

                    remaining -= slots_removed
                    print(
                        f"    Shortened one {bt_name} block to {new_name} (-{slots_removed} slots)")

                    if remaining == 0:
                        break

    # If still have remaining, try removing a room from a block
    if remaining > 0:
        for bt_name in list(adjusted.keys()):
            bt = adjusted[bt_name]
            if bt['n'] > 2 and bt['count'] > 0:
                slots_removed = bt['k']  # removing 1 room removes k slots
                if slots_removed <= remaining:
                    new_n = bt['n'] - 1
                    new_name = f"{new_n}R{bt['k']}T"

                    adjusted[bt_name]['count'] -= 1
                    if adjusted[bt_name]['count'] == 0:
                        del adjusted[bt_name]

                    if new_name in adjusted:
                        adjusted[new_name]['count'] += 1
                    else:
                        adjusted[new_name] = {
                            'n': new_n, 'k': bt['k'], 'count': 1}

                    remaining -= slots_removed
                    print(
                        f"    Reduced one {bt_name} block to {new_name} (-{slots_removed} slots)")

                    if remaining == 0:
                        break

    # Final check
    new_total = count_slots(adjusted)
    final_diff = new_total - num_talks

    if final_diff < 0:
        raise ValueError(
            f"Over-reduced: {new_total} slots < {num_talks} talks")

    if final_diff > 0 and final_diff <= 2:
        print(
            f"  After reduction: {new_total} slots, adding {final_diff} dummy talk(s)")
        return adjusted, final_diff
    elif final_diff > 2:
        # Recursively adjust
        return adjust_block_types_for_talks(adjusted, num_talks)

    return adjusted, 0


# =============================================================================
# CONFIGURATION
# =============================================================================
# User's requested block configuration
# Format: {block_type: {'n': rooms, 'k': timeslots, 'count': num_blocks}}
#
# For testing, we can use different configurations:
# - Same n (simpler): All blocks use same number of rooms
# - Variable n (harder): Blocks with different room counts
#
# Config A: All 4-room blocks (simpler for Phase 1)
# block_types = {
#     '4R5T': {'n': 4, 'k': 5, 'count': 4},   # 4 blocks × 4 rooms × 5 timeslots = 80
# }
#
# Config B: Mixed (user's request - requires generating ~25M tuples)
# block_types = {
#     '5R4T': {'n': 5, 'k': 4, 'count': 2},   # 40 slots
#     '4R4T': {'n': 4, 'k': 4, 'count': 2},   # 32 slots
#     '4R2T': {'n': 4, 'k': 2, 'count': 1},   # 8 slots
# }

# Use simpler single-n config for faster testing
block_types = {
    # 4 blocks × 4 rooms × 5 timeslots = 80
    '4R5T': {'n': 4, 'k': 5, 'count': 4},
}

# Load data
print("Loading ORBEL 2017 data...")
conference_data = load_from_csv_dir('examples/orbel2017')
num_talks = len(conference_data.talks)
n_participants = conference_data.preferences["participant_id"].nunique()
print(f"Loaded: {num_talks} talks, {n_participants} participants")

# Adjust block types for actual number of talks
block_types, num_dummy_talks = adjust_block_types_for_talks(
    block_types, num_talks)

# Add dummy talks if needed
if num_dummy_talks > 0:
    print(f"\nAdding {num_dummy_talks} dummy talks to fill remaining slots")
    for i in range(num_dummy_talks):
        dummy_id = f"DUMMY{i+1}"
        conference_data.talks.append(dummy_id)
    print(f"Total talks (including dummies): {len(conference_data.talks)}")

# Rebuild instance with potentially modified data
instance = build_instance(conference_data)

print(f"\nFinal configuration:")
for bt_name, bt in sorted(block_types.items()):
    slots = bt['n'] * bt['k'] * bt['count']
    print(
        f"  {bt_name}: {bt['count']} block(s) × {bt['n']} rooms × {bt['k']} timeslots = {slots} slots")
total_slots = sum(bt['n'] * bt['k'] * bt['count']
                  for bt in block_types.values())
print(f"  Total: {total_slots} slots for {len(instance.talks)} talks")

# Run Phase 1 with variable n
print("\n" + "=" * 70)
print("PHASE 1: MAXIMIZE ATTENDANCE (Variable n)")
print("=" * 70)

with Phase1Optimizer() as optimizer:
    optimizer.set_problem_instance(instance)

    # Show tuple types derived from block types
    print(f"Tuple types needed: {optimizer.tuple_types}")

    # Build and solve model
    optimizer.build_model(time_limit=120.0, verbose=True)
    optimizer.solve()

    # Get results grouped by tuple size
    phase1_tuples_by_n = optimizer.get_result_by_size()

    if phase1_tuples_by_n is None:
        print("Phase 1 failed - no solution found")
        exit(1)

    print(f"\nPhase 1 result:")
    for n, tuples in sorted(phase1_tuples_by_n.items()):
        print(f"  {len(tuples)} tuples of size {n}")

    phase1_objective = optimizer.get_objective_value()
    print(f"Objective (missed attendance): {phase1_objective:.0f}")

# Prepare Phase 2 input with new format
# block_specs: [(n, k, count, block_type), ...]
# Derive from the adjusted block_types dict
block_specs = [
    (bt['n'], bt['k'], bt['count'], bt_name)
    for bt_name, bt in sorted(block_types.items(), key=lambda x: (-x[1]['n'], -x[1]['k']))
]
print(f"\nBlock specs for Phase 2: {block_specs}")

phase2_input = Phase2Input(
    tuples_by_n=phase1_tuples_by_n,
    block_specs=block_specs,
    preferences=instance.preferences
)

# Run Phase 2
print()
phase2_result = solve_phase2(
    phase2_input,
    partition_strategy="greedy",
    ordering_strategy="enumerate",
    use_local_search=True,
    local_search_iterations=2000,
    verbose=True
)

# Print detailed results
print("\n" + "=" * 70)
print("PHASE 2 DETAILED RESULTS")
print("=" * 70)

for block in phase2_result.blocks:
    print(f"\n{block.block_id} ({block.block_type}) - Hopping: {block.hopping_cost}")
    for i, ntuple in enumerate(block.tuples, 1):
        print(f"  Position {i}: {ntuple}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Phase 1 - Missed attendance: {phase1_objective:.0f}")
print(f"Phase 2 - Total hopping: {phase2_result.total_hopping}")
print(f"Total blocks: {len(phase2_result.blocks)}")

# Compare with random partition baseline
print("\n--- Baseline comparison (random partition) ---")

# Run with random partition (no local search) multiple times
random_costs = []
for seed in range(10):
    import random
    random.seed(seed)
    result = solve_phase2(
        phase2_input,
        partition_strategy="random",
        ordering_strategy="enumerate",
        use_local_search=False,
        verbose=False
    )
    random_costs.append(result.total_hopping)

print(
    f"Random partition hopping (10 runs): min={min(random_costs)}, max={max(random_costs)}, avg={sum(random_costs)/len(random_costs):.1f}")
print(f"Greedy + local search hopping: {phase2_result.total_hopping}")
print(
    f"Improvement over random avg: {sum(random_costs)/len(random_costs) - phase2_result.total_hopping:.1f}")

# =============================================================================
# EXPORT: Session Schedule (What talks are in each session/block)
# =============================================================================
print("\n" + "=" * 70)
print("SESSION SCHEDULE (After Phase 2)")
print("=" * 70)
print("""
NOTE: Each block = one session. Within a session, talks in the same 
timeslot run in PARALLEL across different rooms. The specific room 
assignments are determined in Phase 3.

Format: Session → Timeslot → [Talk1 | Talk2 | Talk3 | Talk4] (parallel)
""")

for block in phase2_result.blocks:
    print(f"\n{'─' * 60}")
    print(
        f"SESSION: {block.block_id} (Type: {block.block_type}, Room Hops: {block.hopping_cost})")
    print(f"{'─' * 60}")

    n_rooms = len(block.tuples[0])

    # Print header
    room_headers = " | ".join([f"Room {r+1:^5}" for r in range(n_rooms)])
    print(f"  Timeslot  │ {room_headers}")
    print(f"  {'─' * 10}┼{'─' * (10 * n_rooms + 3 * (n_rooms - 1) + 2)}")

    for slot, ntuple in enumerate(block.tuples, 1):
        # Show talk numbers without T prefix for readability
        talks = " | ".join([f"{t[1:]:^7}" for t in ntuple])
        print(f"     {slot}      │ {talks}")

# Export to CSV
csv_path = 'session_schedule.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)

    # Determine max rooms across all blocks for header
    max_rooms = max(len(block.tuples[0]) for block in phase2_result.blocks)
    header = ['Session', 'Session_Type', 'Timeslot'] + \
        [f'Room_{r+1}' for r in range(max_rooms)]
    writer.writerow(header)

    for block in phase2_result.blocks:
        for slot, ntuple in enumerate(block.tuples, 1):
            # Pad with empty strings if fewer rooms than max
            padded_tuple = list(ntuple) + [''] * (max_rooms - len(ntuple))
            row = [block.block_id, block.block_type, slot] + padded_tuple
            writer.writerow(row)

print(f"\n\nSchedule exported to: {csv_path}")

# Also show which talks are together in the SAME SESSION (across all timeslots)
print("\n" + "=" * 70)
print("TALKS GROUPED BY SESSION")
print("=" * 70)
print("(All talks in a session share the same 'session block')\n")

for block in phase2_result.blocks:
    all_talks = []
    for ntuple in block.tuples:
        all_talks.extend(ntuple)
    # Sort by talk number
    all_talks_sorted = sorted(all_talks, key=lambda t: int(t[1:]))
    print(f"{block.block_id}: {', '.join(all_talks_sorted)}")
