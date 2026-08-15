"""
q_learning.py
-------------
Tabular Q-learning agent for Tic Tac Toe, trained via self-play.

Pipeline (see main()):
    1. Prompt for episodes / learning rate / discount, then self-play:
       an X-agent and an O-agent train against each other. The O-agent
       is the one we keep - it is what plays vs. random/vs. you and
       what gets exported.
    2. Play a batch of games against a random opponent and report stats
       (greedy, no further learning).
    3. Play interactively against the user (pygame window) - the agent
       keeps learning (epsilon-greedy, TD updates) as it plays.
    4. Export both agents' Q-values (and the O-agent's greedy policy) to JSON.
"""

import random
import sys
from collections import defaultdict

import pygame
from tqdm import tqdm

import utils
import render


class QLearningTicTacToe:
    def __init__(self, alpha=0.3, gamma=0.9):
        self.alpha = alpha
        self.gamma = gamma

        # The main agent - always plays O, is what gets played against and exported.
        self.o_q_values = defaultdict(lambda: defaultdict(float))
        self.o_epsilon = 1.0

        # Its self-play sparring partner - always plays X. Its epsilon never
        # decays (stays fully random) so it acts like a random-but-tracked
        # opponent for the O-agent to learn against.
        self.x_q_values = defaultdict(lambda: defaultdict(float))
        self.x_epsilon = 1.0
        self.x_alpha = 0.3
        self.x_gamma = 0.9

    # -- self-play training ------------------------------------------------

    def self_play_train(self, episodes):
        game_finished = False

        count = 0
        win_count = 0
        loss_count = 0
        stalemate_count = 0

        board, logical_board = utils.new_boards()
        to_move = 'X'

        pbar = tqdm(total=episodes, desc="Self-Play Training", ncols=80)

        while count < episodes:
            if game_finished:
                board, logical_board = utils.new_boards()
                to_move = 'X'
                game_finished = False

            if to_move == "X":
                x_last_state = utils.to_state(logical_board)
                for spot in utils.get_empty_spots(logical_board):
                    if spot not in self.x_q_values[x_last_state]:
                        self.x_q_values[x_last_state][spot] = 0.0

                if random.random() < self.x_epsilon:
                    row1, col1 = utils.random_move(logical_board)
                else:
                    action1, _ = max(self.x_q_values.get(x_last_state, {}).items(), key=lambda item: item[1])
                    row1, col1 = action1

                utils.place_X(board, logical_board, (row1, col1))

                x_new_state = utils.to_state(logical_board)
                for spot in utils.get_empty_spots(logical_board):
                    if spot not in self.x_q_values[x_new_state]:
                        self.x_q_values[x_new_state][spot] = 0.0

                reward1 = 0
                actions_dict1 = self.x_q_values.get(x_new_state, {})
                max_value1 = max(actions_dict1.values(), default=0.0)
                self.x_q_values[x_last_state][(row1, col1)] += self.x_alpha * (
                    reward1 + self.x_gamma * max_value1 - self.x_q_values[x_last_state][(row1, col1)]
                )

                to_move = 'O'
            else:
                last_state = utils.to_state(logical_board)
                for spot in utils.get_empty_spots(logical_board):
                    if spot not in self.o_q_values[last_state]:
                        self.o_q_values[last_state][spot] = 0.0

                if random.random() < self.o_epsilon:
                    row, col = utils.random_move(logical_board)
                else:
                    action, _ = max(self.o_q_values.get(last_state, {}).items(), key=lambda item: item[1])
                    row, col = action

                utils.place_O(board, logical_board, (row, col))

                new_state = utils.to_state(logical_board)
                for spot in utils.get_empty_spots(logical_board):
                    if spot not in self.o_q_values[new_state]:
                        self.o_q_values[new_state][spot] = 0.0

                reward = -0.005
                actions_dict = self.o_q_values.get(new_state, {})
                max_value = max(actions_dict.values(), default=0.0)
                self.o_q_values[last_state][(row, col)] += self.alpha * (
                    reward + self.gamma * max_value - self.o_q_values[last_state][(row, col)]
                )

                to_move = 'X'

            winner = utils.check_win(board)
            if winner is not None:
                if winner == "X":
                    self.x_q_values[x_last_state][(row1, col1)] += self.x_alpha * (1 - self.x_q_values[x_last_state][(row1, col1)])
                    self.o_q_values[last_state][(row, col)] += self.alpha * (-1 - self.o_q_values[last_state][(row, col)])
                    loss_count += 1
                elif winner == "O":
                    self.x_q_values[x_last_state][(row1, col1)] += self.x_alpha * (-1 - self.x_q_values[x_last_state][(row1, col1)])
                    self.o_q_values[last_state][(row, col)] += self.alpha * (1 - self.o_q_values[last_state][(row, col)])
                    win_count += 1
                else:
                    self.x_q_values[x_last_state][(row1, col1)] += self.x_alpha * (0 - self.x_q_values[x_last_state][(row1, col1)])
                    self.o_q_values[last_state][(row, col)] += self.alpha * (0 - self.o_q_values[last_state][(row, col)])
                    stalemate_count += 1

                game_finished = True
                # 0.9999 Seems to yield the best results
                self.o_epsilon = max(self.o_epsilon * 0.99, 0.05)
                count += 1
                pbar.update(1)

        pbar.close()

        print(f'=========== Self-Play Training Results ===========')
        print(f'At {count} games, the current stats are:')
        print(f'Wins: {win_count}')
        print(f'Losses: {loss_count}')
        print(f'Stalemate: {stalemate_count}')
        print(f'Current epsilon value: {self.o_epsilon}')
        print(f'Win rate is {(win_count / count) * 100}%')

    # -- acting --------------------------------------------------------

    def best_action(self, state):
        """Greedy action for the O-agent, falling back to random for
        never-seen states."""
        actions = self.o_q_values.get(state, {})
        if not actions:
            return utils.random_move(state)
        action, _ = max(actions.items(), key=lambda item: item[1])
        return action

    # -- evaluation / play -------------------------------------------------

    def play_vs_random(self, num_games):
        """Simulate `num_games` games (agent = O, greedy, no learning;
        opponent = random X) and return (wins, losses, draws)."""
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
        """Interactive pygame game against the user. The O-agent keeps
        learning (epsilon-greedy, TD updates) as it plays."""
        screen, board_img, x_img, o_img, font, small_font = render.init_window()

        epsilon = 0.4
        agent_mark = "O" if player_choice == "X" else "X"

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
        last_state, last_action = None, None

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
                        print(f'Current epsilon value: {epsilon}')
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
                    render.draw_stats_panel(screen, small_font, play_win_count, play_loss_count, play_stalemate_count, label="Q-Learning")
                    pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if game_finished:
                        board, logical_board = utils.new_boards()
                        graphical_board = render.new_graphical_board()
                        to_move = player_choice
                        last_state, last_action = None, None

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
                            to_move = agent_mark

                        render.render_board(board, x_img, o_img, graphical_board)
                        render.blit_graphical_board(screen, graphical_board)
                        render.draw_status_bar(screen, small_font, "Agent turn")
                        render.draw_stats_panel(screen, small_font, play_win_count, play_loss_count, play_stalemate_count, label="Q-Learning")
                        pygame.display.update()
                    else:
                        last_state = utils.to_state(logical_board)
                        for spot in utils.get_empty_spots(logical_board):
                            if spot not in self.o_q_values[last_state]:
                                self.o_q_values[last_state][spot] = 0.0

                        if random.random() < epsilon:
                            row, col = utils.random_move(logical_board)
                            print(f"We chose to explore: ({row, col})")
                        else:
                            action, max_value = max(self.o_q_values.get(last_state, {}).items(), key=lambda item: item[1])
                            utils.print_state_q_values(self.o_q_values, last_state)
                            print(f'We chose {action} with value: {max_value}')
                            row, col = action

                        last_action = (row, col)

                        if agent_mark == "O":
                            utils.place_O(board, logical_board, (row, col))
                        else:
                            utils.place_X(board, logical_board, (row, col))
                        to_move = player_choice

                        render.render_board(board, x_img, o_img, graphical_board)
                        render.blit_graphical_board(screen, graphical_board)
                        render.draw_status_bar(screen, small_font, "Your turn")
                        render.draw_stats_panel(screen, small_font, play_win_count, play_loss_count, play_stalemate_count, label="Q-Learning")
                        pygame.display.update()

                        new_state = utils.to_state(logical_board)
                        for spot in utils.get_empty_spots(logical_board):
                            if spot not in self.o_q_values[new_state]:
                                self.o_q_values[new_state][spot] = 0.0

                        reward = 0.0
                        actions_dict = self.o_q_values.get(new_state, {})
                        max_value = max(actions_dict.values(), default=0.0)
                        self.o_q_values[last_state][(row, col)] += self.alpha * (
                            reward + self.gamma * max_value - self.o_q_values[last_state][(row, col)]
                        )

                    winner = render.check_win_update(board, graphical_board, screen)
                    if winner is not None:
                        if winner == player_choice:
                            reward = -1
                            play_loss_count += 1
                        elif winner == agent_mark:
                            reward = 1
                            play_win_count += 1
                        else:
                            reward = 0
                            play_stalemate_count += 1

                        if last_state is not None and last_action is not None:
                            self.o_q_values[last_state][last_action] += self.alpha * (
                                reward - self.o_q_values[last_state][last_action]
                            )

                        game_finished = True
                        # 0.9999 Seems to yield the best results
                        epsilon = max(epsilon * 0.93, 0.1)
                        play_count += 1
                        win_rate = play_win_count / play_count
                        win_rate_history.append(win_rate)
                        game_intervals.append(play_count)
                        if play_count % 3 == 0:
                            print(f'At {play_count} games, the current stats are:')
                            print(f'Wins: {play_win_count}')
                            print(f'Losses: {play_loss_count}')
                            print(f'Stalemate: {play_stalemate_count}')
                            print(f'Current epsilon value: {epsilon}')
                            print(f'Win rate is {win_rate * 100}%')

    # -- export --------------------------------------------------------

    def export_json(self, o_q_path="q_learning_o_q_values.json", x_q_path="q_learning_x_q_values.json",
                     policy_path="q_learning_policy.json"):
        utils.export_q_values(self.o_q_values, o_q_path)
        utils.export_q_values(self.x_q_values, x_q_path)
        utils.export_policy(self.o_q_values, policy_path)


def main():
    max_episodes, learning_rate, discount_factor = render.prompt_for_rl_params()

    agent = QLearningTicTacToe(alpha=learning_rate, gamma=discount_factor)
    agent.self_play_train(max_episodes)

    num_games = render.prompt_for_eval_games()
    win_count, loss_count, stalemate_count = agent.play_vs_random(num_games)

    print(f'=========== Vs. Random Results ===========')
    print(f'At {num_games} games, the current stats are:')
    print(f'Wins: {win_count}')
    print(f'Losses: {loss_count}')
    print(f'Stalemate: {stalemate_count}')
    print(f'Win rate is {(win_count / num_games) * 100}%')

    player_choice = render.prompt_for_player_choice()
    agent.play_vs_user(player_choice)

    agent.export_json()


if __name__ == "__main__":
    main()
