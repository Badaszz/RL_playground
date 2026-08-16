"""
value_iteration.py
-------------------
Value Iteration agent for Tic Tac Toe (always plays 'O').

States are (board, player_to_move) pairs so the Bellman backup correctly
alternates turns: O's node takes the max over its actions, X's node takes
the min (X is treated as an adversary choosing the worst outcome for O),
so the learned policy never picks up on a phantom "O moves twice in a
row" shortcut.

Pipeline (see main()):
    1. Prompt for iterations / discount, then run value iteration over
       every reachable (board, player) state.
    2. Play a batch of games against a random opponent and report stats.
    3. Play interactively against the user (pygame window). The policy is
       fixed at this point - the agent does not learn while playing.
    4. Export the learned Q-values and greedy policy to JSON.
"""

import sys
import pygame
from tqdm import tqdm

import utils
import render


class ValueIteration:
    def __init__(self, discount=0.9):
        self.discount = discount
        self.states = set()
        self.state_values = {}
        self.q_values = {}

    # -- state / table generation (kept out of utils.py on purpose) --------

    def generate_all_states(self):
        """Enumerate every reachable (board, player_to_move) state via
        exhaustive recursion, then allocate a state-value and a Q-value
        entry (per legal action) for each non-terminal one."""

        def _generate(board, player, states):
            board_t = tuple(tuple(row) for row in board)
            state = (board_t, player)
            states.add(state)

            winner = utils.check_win_state(board_t)
            if winner is not None:
                return

            for i in range(3):
                for j in range(3):
                    if board[i][j] == 0:
                        new_board = [row[:] for row in board]
                        new_board[i][j] = player
                        next_player = 1 if player == 2 else 2
                        _generate(new_board, next_player, states)

        states = set()
        initial_board = [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]
        # Suppose player 1 (X) starts
        _generate(initial_board, 1, states)
        print(f"Total reachable states: {len(states)}")

        self.states = states
        self.state_values = {}
        self.q_values = {}

        for state in states:
            board, player = state

            # Terminal states have no actions
            if utils.check_win_state(board) is not None:
                self.state_values[state] = 0.0
                self.q_values[state] = {}
                continue

            actions = utils.get_empty_spots(board)
            self.state_values[state] = 0.0
            self.q_values[state] = {action: 0.0 for action in actions}

        count = sum(len(actions) for actions in self.q_values.values())
        print(f"Total state-action pairs: {count}")

        return self.states

    # -- training ------------------------------------------------------

    def train(self, iterations):
        """Sweep every non-terminal (board, player) state `iterations`
        times, updating state values and Q-values via the Bellman
        equation - maximizing at O's nodes, minimizing at X's nodes."""
        for _ in tqdm(range(iterations), desc="Value Iteration", ncols=80):
            for state in self.state_values:
                board, player = state

                # Terminal state
                if utils.check_win_state(board) is not None:
                    continue

                for action in utils.get_empty_spots(board):
                    new_board = [list(row) for row in board]
                    new_board[action[0]][action[1]] = player
                    new_state_board = tuple(tuple(row) for row in new_board)

                    winner = utils.check_win_state(new_board)
                    if winner == 1:
                        reward = -1.0
                    elif winner == 2:
                        reward = 1.0
                    elif winner == 0:
                        reward = 0.0
                    else:
                        reward = -0.05

                    if winner is not None:
                        future_value = 0.0
                    else:
                        # Switch to the other player
                        next_player = 1 if player == 2 else 2
                        next_state = (new_state_board, next_player)
                        future_value = self.state_values[next_state]

                    # Update Q-value using the Bellman equation
                    self.q_values[state][action] = reward + self.discount * future_value

                    # Choose max for our turn, min for opponent's turn
                    if player == 2:
                        self.state_values[state] = max(self.q_values[state].values())
                    else:
                        self.state_values[state] = min(self.q_values[state].values())

    # -- acting ----------------------------------------------------------

    def best_action(self, board_state):
        """Greedy O action for a board (tuple-of-tuples), looked up at the
        O-to-move node. Falls back to a random empty spot for boards the
        sweep never populated."""
        state = (board_state, 2)
        actions = self.q_values.get(state, {})
        if not actions:
            return utils.random_move(board_state)
        action, _ = max(actions.items(), key=lambda item: item[1])
        return action

    # -- evaluation / play -------------------------------------------------

    def play_vs_random(self, num_games):
        """Simulate `num_games` games (agent = O, opponent = random X, no
        rendering) and return (wins, losses, draws)."""
        win_count = 0
        loss_count = 0
        stalemate_count = 0

        for _ in tqdm(range(num_games), desc="Vs. Random", ncols=80):
            board, logical_board = utils.new_boards()
            to_move = "X"
            winner = None

            while winner is None:
                if to_move == "X":
                    row, col = utils.random_move(logical_board)
                    utils.place_X(board, logical_board, (row, col))
                    to_move = "O"
                else:
                    state = utils.to_state(logical_board)
                    row, col = self.best_action(state)
                    utils.place_O(board, logical_board, (row, col))
                    to_move = "X"

                winner = utils.check_win(board)

            if winner == "X":
                loss_count += 1
            elif winner == "O":
                win_count += 1
            else:
                stalemate_count += 1

        return win_count, loss_count, stalemate_count

    def play_vs_user(self, player_choice):
        """Interactive pygame game against the user. The policy is fixed -
        the agent does not learn while playing."""
        screen, board_img, x_img, o_img, font, small_font = render.init_window()

        play_count = 0
        play_win_count = 0
        play_loss_count = 0
        play_stalemate_count = 0

        win_rate_history = []
        game_intervals = []

        game_finished = True
        board, logical_board = utils.new_boards()
        graphical_board = render.new_graphical_board()
        to_move = player_choice

        render.draw_background(screen, board_img)
        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if play_count != 0:
                        print(f'At {play_count} games, the current stats are:')
                        print(f'Wins: {play_win_count}')
                        print(f'Losses: {play_loss_count}')
                        print(f'Stalemate: {play_stalemate_count}')
                        print(f'Win rate is {(play_win_count / play_count) * 100}%')

                        render.plot_win_rate(game_intervals, win_rate_history)
                    else:
                        print("You have not played yet!")
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    render.draw_background(screen, board_img)
                    render.render_board(board, x_img, o_img, graphical_board)
                    render.blit_graphical_board(screen, graphical_board)
                    render.draw_hover_highlight(screen, logical_board, event.pos)
                    render.draw_status_bar(screen, small_font, f"{'Your' if to_move == player_choice else 'Agent'} turn ({to_move})")
                    render.draw_stats_panel(screen, small_font, play_win_count, play_loss_count, play_stalemate_count, label="Value Iteration")
                    pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if game_finished:
                        board, logical_board = utils.new_boards()
                        graphical_board = render.new_graphical_board()
                        to_move = player_choice

                        render.draw_background(screen, board_img)
                        game_finished = False
                        pygame.display.update()

                    if to_move == player_choice:
                        row, col = render.get_cell_from_mouse(event.pos)
                        if 0 <= row < 3 and 0 <= col < 3 and logical_board[row][col] == 0:
                            if player_choice == "X":
                                utils.place_X(board, logical_board, (row, col))
                            else:
                                utils.place_O(board, logical_board, (row, col))
                            to_move = "O" if player_choice == "X" else "X"

                        render.render_board(board, x_img, o_img, graphical_board)
                        render.blit_graphical_board(screen, graphical_board)
                        render.draw_status_bar(screen, small_font, "Agent turn")
                        render.draw_stats_panel(screen, small_font, play_win_count, play_loss_count, play_stalemate_count, label="Value Iteration")
                        pygame.display.update()
                    else:
                        state = utils.to_state(logical_board)
                        row, col = self.best_action(state)

                        if player_choice == "X":
                            utils.place_O(board, logical_board, (row, col))
                            to_move = "X"
                        else:
                            utils.place_X(board, logical_board, (row, col))
                            to_move = "O"

                        render.render_board(board, x_img, o_img, graphical_board)
                        render.blit_graphical_board(screen, graphical_board)
                        render.draw_status_bar(screen, small_font, "Your turn")
                        render.draw_stats_panel(screen, small_font, play_win_count, play_loss_count, play_stalemate_count, label="Value Iteration")
                        pygame.display.update()

                    winner = render.check_win_update(board, graphical_board, screen)
                    if winner is not None:
                        agent_mark = "O" if player_choice == "X" else "X"
                        if winner == player_choice:
                            play_loss_count += 1
                        elif winner == agent_mark:
                            play_win_count += 1
                        else:
                            play_stalemate_count += 1

                        game_finished = True
                        play_count += 1
                        win_rate = play_win_count / play_count
                        win_rate_history.append(win_rate)
                        game_intervals.append(play_count)
                        if play_count % 3 == 0:
                            print(f'At {play_count} games, the current stats are:')
                            print(f'Wins: {play_win_count}')
                            print(f'Losses: {play_loss_count}')
                            print(f'Stalemate: {play_stalemate_count}')
                            print(f'Win rate is {win_rate * 100}%')

    # -- export --------------------------------------------------------

    def export_json(self, q_path="value_iteration_q_values.json", policy_path="value_iteration_policy.json"):
        """Only the O-to-move nodes make up a usable policy, so we strip
        the player index and export those boards' Q-values/greedy policy."""
        o_turn_q_values = {
            board: actions
            for (board, player), actions in self.q_values.items()
            if player == 2 and actions
        }
        utils.export_q_values(o_turn_q_values, q_path)
        utils.export_policy(o_turn_q_values, policy_path)


def main():
    iterations, discount = render.prompt_for_vi_params()

    agent = ValueIteration(discount=discount)
    agent.generate_all_states()
    agent.train(iterations)

    num_games = render.prompt_for_eval_games()
    win_count, loss_count, stalemate_count = agent.play_vs_random(num_games)
    
    ## Export Q_values and policy to json after training
    agent.export_json()

    print(f'=========== Vs. Random Results ===========')
    print(f'At {num_games} games, the current stats are:')
    print(f'Wins: {win_count}')
    print(f'Losses: {loss_count}')
    print(f'Stalemate: {stalemate_count}')
    print(f'Win rate is {(win_count / num_games) * 100}%')

    player_choice = render.prompt_for_player_choice()
    agent.play_vs_user(player_choice)

    


if __name__ == "__main__":
    main()