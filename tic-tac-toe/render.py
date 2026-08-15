"""
render.py
---------
Everything visual: the pygame board/graphics, the tkinter parameter
dialogs, and the win-rate matplotlib plot. Game *mechanics* (win
checking, placing marks, state handling) live in utils.py instead.

Assumes an `assets/` folder (relative to the working directory) containing:
    Board.png, X.png, O.png, "Winning X.png", "Winning O.png"
"""

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Suppress Pygame support prompt

import pygame
import sys
import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 900, 900
BG_COLOR = (214, 201, 227)
ASSETS_DIR = "assets"

HOVER_COLOR = (255, 255, 255, 90)
STATUS_BG = (60, 45, 80)
STATUS_TEXT_COLOR = (255, 255, 255)
STATS_BG = (245, 240, 250)
STATS_TEXT_COLOR = (40, 30, 55)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def init_window(title="Tic Tac Toe!"):
    """Initialise pygame, open the game window, and return
    (screen, board_img, x_img, o_img, font, small_font)."""
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(title)

    board_img = pygame.image.load(os.path.join(ASSETS_DIR, "Board.png"))
    x_img = pygame.image.load(os.path.join(ASSETS_DIR, "X.png"))
    o_img = pygame.image.load(os.path.join(ASSETS_DIR, "O.png"))

    font = pygame.font.SysFont("Helvetica", 28, bold=True)
    small_font = pygame.font.SysFont("Helvetica", 20)

    return screen, board_img, x_img, o_img, font, small_font


def new_graphical_board():
    return [[[None, None], [None, None], [None, None]],
            [[None, None], [None, None], [None, None]],
            [[None, None], [None, None], [None, None]]]


def draw_background(screen, board_img):
    screen.fill(BG_COLOR)
    screen.blit(board_img, (64, 64))


# ---------------------------------------------------------------------------
# Board drawing
# ---------------------------------------------------------------------------

def render_board(board, ximg, oimg, graphical_board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 'X':
                graphical_board[i][j][0] = ximg
                graphical_board[i][j][1] = ximg.get_rect(center=(j * 300 + 150, i * 300 + 150))
            elif board[i][j] == 'O':
                graphical_board[i][j][0] = oimg
                graphical_board[i][j][1] = oimg.get_rect(center=(j * 300 + 150, i * 300 + 150))


def blit_graphical_board(screen, graphical_board):
    for i in range(3):
        for j in range(3):
            if graphical_board[i][j][0] is not None:
                screen.blit(graphical_board[i][j][0], graphical_board[i][j][1])


def get_cell_from_mouse(mouse_pos):
    """Convert a raw pygame mouse position into a (row, col) board cell,
    matching the original click-to-cell math."""
    converted_x = (mouse_pos[0] - 65) / 835 * 2
    converted_y = mouse_pos[1] / 835 * 2

    row = round(converted_y)
    col = round(converted_x)
    return row, col


def add_XO(board, graphical_board, to_move, logical_board, screen, ximg, oimg):
    """Original click-handling helper, kept under its original name/shape
    for notebook parity: reads the current mouse position directly rather
    than taking an event position."""
    current_pos = pygame.mouse.get_pos()
    covnerted_x = (current_pos[0] - 65) / 835 * 2
    converted_y = current_pos[1] / 835 * 2

    row = round(converted_y)
    col = round(covnerted_x)

    if board[row][col] != 0 and board[row][col] != 'X':
        board[row][col] = to_move

        if to_move == 'X':
            logical_board[row][col] = 1
            to_move = 'O'
        else:
            logical_board[row][col] = 2
            to_move = 'X'

    render_board(board, ximg, oimg, graphical_board)

    for i in range(3):
        for j in range(3):
            if graphical_board[i][j][0] is not None:
                screen.blit(graphical_board[i][j][0], graphical_board[i][j][1])

    return board, to_move, logical_board


def draw_hover_highlight(screen, logical_board, mouse_pos):
    """Softly highlight the empty cell under the mouse - a small UX touch
    so it is clear where a click will land before you commit to it."""
    row, col = get_cell_from_mouse(mouse_pos)
    if 0 <= row < 3 and 0 <= col < 3 and logical_board[row][col] == 0:
        highlight = pygame.Surface((280, 280), pygame.SRCALPHA)
        highlight.fill(HOVER_COLOR)
        screen.blit(highlight, (col * 300 + 74, row * 300 + 74))


def draw_status_bar(screen, small_font, message):
    """A thin banner along the very top of the window announcing whose
    turn it is / the last result, so the state of the game is obvious
    without reading the console."""
    bar = pygame.Surface((WIDTH, 40))
    bar.fill(STATUS_BG)
    screen.blit(bar, (0, 0))
    text = small_font.render(message, True, STATUS_TEXT_COLOR)
    screen.blit(text, (14, 8))


def draw_stats_panel(screen, small_font, wins, losses, draws, label="Agent"):
    """A small scoreboard along the bottom of the window."""
    bar = pygame.Surface((WIDTH, 40))
    bar.fill(STATS_BG)
    screen.blit(bar, (0, HEIGHT - 40))
    played = wins + losses + draws
    win_rate = (wins / played * 100) if played else 0.0
    text = small_font.render(
        f"{label} - W:{wins}  L:{losses}  D:{draws}  ({win_rate:.1f}% win rate)  |  click after a game ends to play again",
        True, STATS_TEXT_COLOR,
    )
    screen.blit(text, (14, HEIGHT - 32))


def check_win_update(board, graphical_board, screen):
    """Winner check for the display board that ALSO paints the winning
    line with the 'Winning X/O' asset when there is one. Returns the
    winner ('X'/'O'), 'DRAW', or None (game still ongoing)."""
    winner = None
    for row in range(0, 3):
        if (board[row][0] == board[row][1] == board[row][2]) and (board[row][0] is not None):
            winner = board[row][0]
            for i in range(0, 3):
                graphical_board[row][i][0] = pygame.image.load(os.path.join(ASSETS_DIR, f"Winning {winner}.png"))
                screen.blit(graphical_board[row][i][0], graphical_board[row][i][1])
            pygame.display.update()
            return winner

    for col in range(0, 3):
        if (board[0][col] == board[1][col] == board[2][col]) and (board[0][col] is not None):
            winner = board[0][col]
            for i in range(0, 3):
                graphical_board[i][col][0] = pygame.image.load(os.path.join(ASSETS_DIR, f"Winning {winner}.png"))
                screen.blit(graphical_board[i][col][0], graphical_board[i][col][1])
            pygame.display.update()
            return winner

    if (board[0][0] == board[1][1] == board[2][2]) and (board[0][0] is not None):
        winner = board[0][0]
        graphical_board[0][0][0] = pygame.image.load(os.path.join(ASSETS_DIR, f"Winning {winner}.png"))
        screen.blit(graphical_board[0][0][0], graphical_board[0][0][1])
        graphical_board[1][1][0] = pygame.image.load(os.path.join(ASSETS_DIR, f"Winning {winner}.png"))
        screen.blit(graphical_board[1][1][0], graphical_board[1][1][1])
        graphical_board[2][2][0] = pygame.image.load(os.path.join(ASSETS_DIR, f"Winning {winner}.png"))
        screen.blit(graphical_board[2][2][0], graphical_board[2][2][1])
        pygame.display.update()
        return winner

    if (board[0][2] == board[1][1] == board[2][0]) and (board[0][2] is not None):
        winner = board[0][2]
        graphical_board[0][2][0] = pygame.image.load(os.path.join(ASSETS_DIR, f"Winning {winner}.png"))
        screen.blit(graphical_board[0][2][0], graphical_board[0][2][1])
        graphical_board[1][1][0] = pygame.image.load(os.path.join(ASSETS_DIR, f"Winning {winner}.png"))
        screen.blit(graphical_board[1][1][0], graphical_board[1][1][1])
        graphical_board[2][0][0] = pygame.image.load(os.path.join(ASSETS_DIR, f"Winning {winner}.png"))
        screen.blit(graphical_board[2][0][0], graphical_board[2][0][1])
        pygame.display.update()
        return winner

    if winner is None:
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] != 'X' and board[i][j] != 'O':
                    return None
        return "DRAW"


# ---------------------------------------------------------------------------
# tkinter dialogs
# ---------------------------------------------------------------------------

def prompt_for_rl_params():
    """Ask for Q-learning hyperparameters: episodes, learning rate, discount."""
    root = tk.Tk()
    root.withdraw()

    max_episodes = simpledialog.askinteger(
        title="Max Episodes",
        prompt="Enter the number of episodes you want to train for:",
        minvalue=1,
        maxvalue=10_000_000
    )
    if max_episodes is None:
        print("No input provided for max_episodes. Using default = 20000.")
        max_episodes = 20000

    learning_rate = simpledialog.askfloat(
        title="Learning Rate",
        prompt="Enter the learning rate (alpha):\n(e.g., 0.1, 0.3, etc.)",
        minvalue=0.000001,
        maxvalue=1.0
    )
    if learning_rate is None:
        print("No input provided for alpha. Using default = 0.3.")
        learning_rate = 0.3

    discount_factor = simpledialog.askfloat(
        title="Discount Factor",
        prompt="Enter the discount factor (gamma):\n(e.g., 0.9, 0.99, etc.)",
        minvalue=0.0,
        maxvalue=1.0
    )
    if discount_factor is None:
        print("No input provided for gamma. Using default = 0.9.")
        discount_factor = 0.9

    root.destroy()

    return max_episodes, learning_rate, discount_factor


def prompt_for_vi_params():
    """Ask for Value Iteration hyperparameters: iterations, discount.
    (No learning rate - value iteration sweeps the full Bellman update
    rather than using a step-size.)"""
    root = tk.Tk()
    root.withdraw()

    iterations = simpledialog.askinteger(
        title="Iterations",
        prompt="Enter the number of value-iteration sweeps you want to run:",
        minvalue=1,
        maxvalue=100_000
    )
    if iterations is None:
        print("No input provided for iterations. Using default = 100.")
        iterations = 100

    discount_factor = simpledialog.askfloat(
        title="Discount Factor",
        prompt="Enter the discount factor (gamma):\n(e.g., 0.9, 0.99, etc.)",
        minvalue=0.0,
        maxvalue=1.0
    )
    if discount_factor is None:
        print("No input provided for gamma. Using default = 0.9.")
        discount_factor = 0.9

    root.destroy()

    return iterations, discount_factor


def prompt_for_player_choice():
    def set_choice(selected_choice):
        nonlocal choice
        choice = selected_choice
        root.destroy()  # Close the window once a choice is made

    root = tk.Tk()
    root.title("Player Choice")

    # Set the window size and position
    root.geometry("300x150")
    root.eval('tk::PlaceWindow . center')  # Center the window

    # Add label
    label = tk.Label(root, text="Would you like to go first or second?", font=("Helvetica", 12))
    label.pack(pady=10)

    # Add buttons for "X" and "O"
    button_x = tk.Button(root, text="First", font=("Helvetica", 14), width=10, command=lambda: set_choice("X"))
    button_x.pack(pady=5)

    button_o = tk.Button(root, text="Second", font=("Helvetica", 14), width=10, command=lambda: set_choice("O"))
    button_o.pack(pady=5)

    # Initialize choice
    choice = None
    root.mainloop()  # Run the Tkinter event loop

    # Default to "X" if no choice was made (e.g., the window was closed)
    if choice is None:
        choice = "X"

    return choice


def prompt_for_eval_games():
    """Ask how many games to play against the random opponent before
    reporting results."""
    root = tk.Tk()
    root.withdraw()

    num_games = simpledialog.askinteger(
        title="Evaluation Games",
        prompt="How many games vs. a random opponent should be played?",
        minvalue=1,
        maxvalue=1_000_000
    )
    if num_games is None:
        print("No input provided for num_games. Using default = 100.")
        num_games = 100

    root.destroy()

    return num_games


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_win_rate(game_intervals, win_rate_history, title="Win Rate Over Time"):
    plt.figure(figsize=(10, 6))
    plt.plot(game_intervals, win_rate_history, label="Win Rate", color='blue')
    plt.title(title)
    plt.xlabel("Number of Games")
    plt.ylabel("Win Rate")
    plt.grid(True)
    plt.legend()
    plt.show()
