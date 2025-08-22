"""
Utility functions for Fantasy Football Draft Assistant.
"""
import toml
import os

def read_config(path="config.toml"):
    """
    Read the config.toml file and return as a dict.
    """
    if not os.path.exists(path):
        return {}
    return toml.load(path)

def save_config(config, path="config.toml"):
    """
    Save the config dict to config.toml.
    """
    with open(path, "w") as f:
        toml.dump(config, f)

def snake_position(overall_pick, teams):
    """
    Given overall pick number (1-indexed) and number of teams,
    return (round_number, pick_in_round, draft_slot).
    Draft slot accounts for snake order.
    """
    if teams <= 0:
        return (0, 0, 0)
    round_num = (overall_pick - 1) // teams + 1
    pick_in_round = (overall_pick - 1) % teams + 1
    # Snake ordering: even rounds reversed
    if round_num % 2 == 0:
        draft_slot = teams - pick_in_round + 1
    else:
        draft_slot = pick_in_round
    return (round_num, pick_in_round, draft_slot)
