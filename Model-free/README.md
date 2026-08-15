# Model-Free Reinforcement Learning

This directory contains implementations of model-free reinforcement learning algorithms. Unlike Dynamic Programming, these methods do not require a model of the environment's dynamics (state transitions and rewards) and instead learn directly from experience by interacting with the environment.

## Algorithms Implemented

### 1. Monte Carlo Methods
Monte Carlo (MC) methods learn value functions and optimal policies from sample episodes. They are based on averaging the returns observed after visiting a state.

*   **File**: `monte-carlo.ipynb`
*   **Key Concept**: $V(s) \approx \text{average}(G_t)$ where $G_t$ is the total return from time $t$.

#### Variations Included:
1.  **Monte Carlo Exploring Starts (MCES)**:
    *   Ensures all state-action pairs are visited by starting episodes from every possible state-action combination.
    *   Follows the current policy after the initial exploring start.
2.  **Monte Carlo Epsilon-Greedy ($\epsilon$-greedy)**:
    *   An on-policy method that ensures continuous exploration by choosing a random action with probability $\epsilon$ and the greedy action with probability $1-\epsilon$.
    *   Eliminates the requirement for exploring starts by maintaining exploration throughout the learning process.

## Environment: GridWorld
The algorithms are tested on a 2D GridWorld environment:
-   **States**: Discrete tiles in a grid (e.g., 4x4).
-   **Actions**: Up (↑), Down (↓), Left (←), Right (→).
-   **Rewards**: Terminal states have non-zero rewards (e.g., +1 for a goal, -1 for a trap). All other steps usually have a reward of 0.
-   **Goal**: Find the optimal policy that navigates to the goal state while maximizing total expected return.

## Usage
Open `monte-carlo.ipynb` to see the step-by-step implementation of the algorithms, including the episode generation, return calculation, and policy improvement steps.
