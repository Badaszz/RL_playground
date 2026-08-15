# Tic-Tac-Toe with Reinforcement Learning

This project implements and compares different Reinforcement Learning (RL) agents trained to play Tic-Tac-Toe. It features interactive gameplay using Pygame, allowing users to test their skills against trained models.

## Agents and Algorithms

### 1. Q-Learning Agent
A tabular Q-learning implementation that learns through self-play.
-   **Self-Play**: Two agents (X and O) play against each other to populate their Q-tables.
-   **On-policy Learning**: The agent can continue to learn during interactive play against the user using $\epsilon$-greedy exploration.
-   **Files**: `q_learning.py`, `q_learning.ipynb`

### 2. Value Iteration Agent
A model-based approach that computes the optimal value function by iterating over all possible reachable states of the Tic-Tac-Toe board.
-   **State Enumeration**: Exhaustively generates all unique board configurations.
-   **Bellman Optimality**: Updates state and Q-values using the Bellman equation until convergence.
-   **Fixed Policy**: Once trained, the agent follows a deterministic greedy policy.
-   **Files**: `value_iteration.py`, `value_iteration.ipynb`

## Project Structure

-   `q_learning.py` / `value_iteration.py`: Main entry points for training and playing against the respective agents.
-   `render.py`: Handles the Pygame visualization, including the board, animations, and UI components.
-   `utils.py`: Contains game logic (win checking, board state conversions, move generation) and data export functions.
-   `assets/`: Contains images for the board and marks (X and O).

## How to Play

1.  **Install dependencies**: Ensure you have `pygame`, `tqdm`, and `matplotlib` installed.
2.  **Run an agent**:
    -   For Q-Learning: `python tic-tac-toe/q_learning.py`
    -   For Value Iteration: `python tic-tac-toe/value_iteration.py`
3.  **Configure**: Follow the terminal prompts to set hyperparameters (learning rate, discount factor, etc.).
4.  **Interact**: After training, a Pygame window will open. Choose your mark (X or O) and click on the grid to make your move!

## Exporting Data
After training and playing, the agents automatically export their learned Q-values and greedy policies to JSON files (e.g., `q_learning_policy.json`).
