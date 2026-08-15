"""
utils.py
--------
Core Tic Tac Toe game mechanics shared by every algorithm.
No pygame / tkinter / rendering code lives here (see render.py),
and no value-iteration-specific state/Q-value generation lives here
(see value_iteration.py) - just board setup, moves, win checking,
and small helpers used to train/play/export any agent.
"""

import json
import random


def new_boards():
    """Return a fresh (display_board, logical_board) pair.

    display_board holds 1-9 placeholders (matches the numbered cells the
    player clicks on) which get overwritten with 'X' / 'O'.
    logical_board holds 0 = empty, 1 = X, 2 = O and is what the RL agents
    actually reason about.
    """
    board = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    logical_board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    return board, logical_board


def place_O(board, logical_board, position):
    row, col = position

    board[row][col] = 'O'
    logical_board[row][col] = 2

    return board, logical_board


def place_X(board, logical_board, position):
    row, col = position

    board[row][col] = 'X'
    logical_board[row][col] = 1

    return board, logical_board


def get_empty_spots(logical_board):
    empty_positions = []
    for i in range(3):
        for j in range(3):
            if logical_board[i][j] == 0:
                empty_positions.append((i, j))
    return empty_positions


def random_move(logical_board):
    """Pick a uniformly random empty spot."""
    return random.choice(get_empty_spots(logical_board))


def to_state(logical_board):
    """Convert a mutable logical_board into the hashable tuple-of-tuples
    representation used as a dictionary key everywhere (states, q_values)."""
    return tuple(tuple(row) for row in logical_board)


def check_win(board):
    """Winner check for the *display* board ('X' / 'O' / numbered cells).
    Returns 'X', 'O', 'Draw', or None (game still ongoing)."""
    winner = None
    # Check rows
    for row in range(0, 3):
        if board[row][0] == board[row][1] == board[row][2] and board[row][0] != 0:
            winner = board[row][0]
            return winner

    # Check columns
    for col in range(0, 3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != 0:
            winner = board[0][col]
            return winner

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0:
        winner = board[0][0]
        return winner

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0:
        winner = board[0][2]
        return winner

    if winner is None:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] != 'X' and board[i][j] != 'O':
                    return None  # Game is still ongoing

        return "Draw"  # No empty spaces left, it's a draw


def check_win_state(state_board):
    """Winner check for a purely numeric board (0 = empty, 1 = X, 2 = O).
    Returns 1, 2, 0 (draw), or None (game still ongoing). Used internally
    by algorithms that simulate on the logical/numeric board rather than
    the display board."""
    winner = None
    # Check rows
    for row in range(0, 3):
        if state_board[row][0] == state_board[row][1] == state_board[row][2] and state_board[row][0] != 0:
            winner = state_board[row][0]
            return winner

    # Check columns
    for col in range(0, 3):
        if state_board[0][col] == state_board[1][col] == state_board[2][col] and state_board[0][col] != 0:
            winner = state_board[0][col]
            return winner

    # Check diagonals
    if state_board[0][0] == state_board[1][1] == state_board[2][2] and state_board[0][0] != 0:
        winner = state_board[0][0]
        return winner

    if state_board[0][2] == state_board[1][1] == state_board[2][0] and state_board[0][2] != 0:
        winner = state_board[0][2]
        return winner

    if winner is None:
        for i in range(len(state_board)):
            for j in range(len(state_board[i])):
                if state_board[i][j] != 1 and state_board[i][j] != 2:
                    return None  # Game is still ongoing

        return 0  # No empty spaces left, it's a draw


def print_q_value(q_values):
    print("=== Q-Values Table ===")
    for state, actions in q_values.items():
        print("State:")
        for row in state:
            print("  " + " ".join(str(cell) for cell in row))

        print("Actions and Q-values:")
        for action, q_value in actions.items():
            print(f"  Action {action}: Q-value = {q_value:.2f}")
        print("------------------------")
    print("========================\n")


def print_state_q_values(q_values, state):
    print("=== Q-Values for the Given State ===")
    print("State:")
    for row in state:
        print("  " + " ".join(str(cell) for cell in row))
    print("\nActions and Q-values:")
    if state in q_values:
        actions = q_values[state]
        for action, q_value in actions.items():
            print(f"  Action {action}: Q-value = {q_value:.2f}")
    else:
        print("  No actions available for this state.")
    print("====================================\n")


# ---------------------------------------------------------------------------
# JSON export helpers
# ---------------------------------------------------------------------------

def _encode_state(state):
    """(0,1,2)-tuple-of-tuples -> compact string key, e.g. '120010200'."""
    return "".join(str(cell) for row in state for cell in row)


def _encode_action(action):
    """(row, col) -> 'row_col' string key."""
    return f"{action[0]}_{action[1]}"


def q_values_to_json_dict(q_values):
    """Turn a {state_tuple: {action_tuple: value}} mapping into a
    JSON-serialisable dict of {state_str: {'row_col': value}}."""
    out = {}
    for state, actions in q_values.items():
        out[_encode_state(state)] = {
            _encode_action(action): value for action, value in actions.items()
        }
    return out


def policy_from_q_values(q_values):
    """Greedy policy: best action per state, as {state_str: [row, col]}."""
    policy = {}
    for state, actions in q_values.items():
        if not actions:
            continue
        best_action, _ = max(actions.items(), key=lambda item: item[1])
        policy[_encode_state(state)] = [best_action[0], best_action[1]]
    return policy


def export_q_values(q_values, filepath):
    with open(filepath, "w") as f:
        json.dump(q_values_to_json_dict(q_values), f, indent=2)
    print(f"Saved Q-values to {filepath}")


def export_policy(q_values, filepath):
    with open(filepath, "w") as f:
        json.dump(policy_from_q_values(q_values), f, indent=2)
    print(f"Saved policy to {filepath}")
