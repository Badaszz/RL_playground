"""
policy_iteration.py
-------------------
Policy Iteration agent for Tic Tac Toe.

The agent learns by alternating between:
1. Policy Evaluation: Calculating state values for the current policy.
2. Policy Improvement: Updating the policy to be greedy with respect to 
   the newly calculated values.
"""

import sys
import pygame
import random
from tqdm import tqdm

import utils
import render

class PolicyIteration:
    def __init__(self, discount=0.9):
        self.discount = discount
        self.states = set()
        self.state_values = {}
        self.q_values = {}
        self.policy = {}

    def generate_all_states(self):
        """Enumerate every reachable (board, player) state."""
        def _generate(board, player, states):
            board_tuple = tuple(tuple(row) for row in board)
            state = (board_tuple, player)
            if state in states:
                return
            states.add(state)

            winner = utils.check_win_state(board_tuple)
            if winner is not None:
                return

            for i in range(3):
                for j in range(3):
                    if board[i][j] == 0:
                        new_board = [list(row) for row in board]
                        new_board[i][j] = player
                        next_player = 1 if player == 2 else 2
                        _generate(new_board, next_player, states)

        initial_board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        _generate(initial_board, 1, self.states)
        
        for state in self.states:
            board, player = state
            self.state_values[state] = 0.0
            self.q_values[state] = {}
            
            winner = utils.check_win_state(board)
            if winner is None:
                empty_spots = utils.get_empty_spots(board)
                for action in empty_spots:
                    self.q_values[state][action] = 0.0
                # Initialize policy: pick the first available spot
                self.policy[state] = empty_spots[0]

        print(f"Total reachable states: {len(self.states)}")
        return self.states

    def train(self, iterations, inner_iterations=30):
        """Perform Policy Iteration: Evaluation followed by Improvement."""
        for _ in tqdm(range(iterations), desc="Policy Iteration", ncols=80):
            # 1. Policy Evaluation
            for _ in range(inner_iterations):
                delta_v = 0
                for state in self.states:
                    board, player = state
                    if utils.check_win_state(board) is not None:
                        continue
                    
                    old_v = self.state_values[state]
                    action = self.policy[state]
                    
                    # Simulate move
                    new_board = [list(row) for row in board]
                    new_board[action[0]][action[1]] = player
                    new_board_tuple = tuple(tuple(row) for row in new_board)
                    
                    winner = utils.check_win_state(new_board_tuple)
                    if winner == 1: reward = -1.0
                    elif winner == 2: reward = 1.0
                    elif winner == 0: reward = 0.0
                    else: reward = -0.05
                    
                    future_val = 0.0
                    if winner is None:
                        next_player = 1 if player == 2 else 2
                        future_val = self.state_values[(new_board_tuple, next_player)]
                    
                    self.state_values[state] = reward + self.discount * future_val
                    delta_v = max(delta_v, abs(old_v - self.state_values[state]))
                
                if delta_v < 0.01:
                    break
            
            # 2. Policy Improvement
            for state in self.states:
                board, player = state
                if utils.check_win_state(board) is not None:
                    continue
                
                for action in utils.get_empty_spots(board):
                    new_board = [list(row) for row in board]
                    new_board[action[0]][action[1]] = player
                    new_board_tuple = tuple(tuple(row) for row in new_board)
                    
                    winner = utils.check_win_state(new_board_tuple)
                    if winner == 1: reward = -1.0
                    elif winner == 2: reward = 1.0
                    elif winner == 0: reward = 0.0
                    else: reward = -0.05
                    
                    future_val = 0.0
                    if winner is None:
                        next_player = 1 if player == 2 else 2
                        future_val = self.state_values[(new_board_tuple, next_player)]
                    
                    self.q_values[state][action] = reward + self.discount * future_val
                
                # Update policy: O (2) maximizes, X (1) minimizes
                if player == 2:
                    self.policy[state] = max(self.q_values[state], key=self.q_values[state].get)
                else:
                    self.policy[state] = min(self.q_values[state], key=self.q_values[state].get)

    def best_action(self, board_logical):
        """Greedy action for the agent (always plays as 'O'/2)."""
        board_tuple = tuple(tuple(row) for row in board_logical)
        state = (board_tuple, 2)
        return self.policy.get(state, utils.random_move(board_logical))

    def play_vs_random(self, num_games):
        win_count, loss_count, stalemate_count = 0, 0, 0

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
                    board_logical = utils.to_state(logical_board)
                    row, col = self.best_action(board_logical)
                    utils.place_O(board, logical_board, (row, col))
                    to_move = "X"
                winner = utils.check_win(board)

            if winner == "X": loss_count += 1
            elif winner == "O": win_count += 1
            else: stalemate_count += 1

        return win_count, loss_count, stalemate_count

    def play_vs_user(self, player_choice):
        screen, board_img, x_img, o_img, font, small_font = render.init_window()
        play_count, play_win_count, play_loss_count, play_stalemate_count = 0, 0, 0, 0
        win_rate_history, game_intervals = [], []

        game_finished = True
        board, logical_board = utils.new_boards()
        graphical_board = render.new_graphical_board()
        to_move = player_choice

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if game_finished:
                        board, logical_board = utils.new_boards()
                        graphical_board = render.new_graphical_board()
                        to_move = player_choice
                        game_finished = False

                    if to_move == player_choice:
                        row, col = render.get_cell_from_mouse(event.pos)
                        if 0 <= row < 3 and 0 <= col < 3 and logical_board[row][col] == 0:
                            if player_choice == "X": utils.place_X(board, logical_board, (row, col))
                            else: utils.place_O(board, logical_board, (row, col))
                            to_move = "O" if player_choice == "X" else "X"
                    else:
                        board_logical = utils.to_state(logical_board)
                        row, col = self.best_action(board_logical)
                        if player_choice == "X": utils.place_O(board, logical_board, (row, col))
                        else: utils.place_X(board, logical_board, (row, col))
                        to_move = player_choice

                    render.draw_background(screen, board_img)
                    render.render_board(board, x_img, o_img, graphical_board)
                    render.blit_graphical_board(screen, graphical_board)
                    pygame.display.update()

                    winner = render.check_win_update(board, graphical_board, screen)
                    if winner is not None:
                        agent_mark = "O" if player_choice == "X" else "X"
                        if winner == player_choice: play_loss_count += 1
                        elif winner == agent_mark: play_win_count += 1
                        else: play_stalemate_count += 1
                        game_finished = True
                        play_count += 1

    def export_json(self, q_path="policy_iteration_q_values.json", policy_path="policy_iteration_policy.json"):
        utils.export_q_values(self.q_values, q_path)
        utils.export_policy(self.q_values, policy_path)

def main():
    iterations, discount = render.prompt_for_vi_params()
    agent = PolicyIteration(discount=discount)
    agent.generate_all_states()
    agent.train(iterations)

    num_games = render.prompt_for_eval_games()
    win_count, loss_count, stalemate_count = agent.play_vs_random(num_games)
    print(f"Wins: {win_count}, Losses: {loss_count}, Draws: {stalemate_count}")

    player_choice = render.prompt_for_player_choice()
    agent.play_vs_user(player_choice)
    agent.export_json()

if __name__ == "__main__":
    main()
