# Dynamic Programming in Reinforcement Learning

This directory contains implementations of fundamental Dynamic Programming (DP) algorithms used in Reinforcement Learning to solve Markov Decision Processes (MDPs). These algorithms assume full knowledge of the environment's dynamics (state transitions and rewards).

## Algorithms Implemented

### 1. Value Iteration
Value Iteration is an iterative algorithm that finds the optimal state values by repeatedly applying the Bellman optimality equation. It combines policy evaluation and policy improvement into a single step.

*   **File**: `value_iteration.ipynb`
*   **Key Concept**: $V_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')]$
*   **Environment**: A grid world where the agent learns the shortest path to a goal state while avoiding obstacles.

### 2. Policy Iteration
Policy Iteration alternates between two distinct phases:
1.  **Policy Evaluation**: Calculating the state-value function $V^\pi$ for the current policy $\pi$.
2.  **Policy Improvement**: Updating the policy $\pi$ to be greedy with respect to $V^\pi$.

This process continues until the policy becomes stable.

*   **File**: `policy_iteration.ipynb`
*   **Key Concept**: Iterates through Evaluation and Improvement until $\pi$ converges to $\pi^*$.
*   **Environment**: A flexible GridWorld implementation that supports custom rewards and grid sizes.

## Environment: GridWorld
Both notebooks utilize a GridWorld environment where:
-   **Actions**: Up (↑), Down (↓), Left (←), Right (→).
-   **States**: Discrete tiles in a 2D grid.
-   **Goal**: Maximize cumulative reward (reach the target state with the highest value).
-   **Obstacles**: Tiles with negative rewards that the agent learns to avoid.

## Usage
The implementations are provided in Jupyter notebooks for easy experimentation and visualization of the learned policies and value functions.
