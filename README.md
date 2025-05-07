# Hierarchical Reinforcement Learning for AntMaze

This project implements and compares Hierarchical Reinforcement Learning (HRL) agents, specifically a custom Hierarchical Actor-Critic (HAC) agent, against a Proximal Policy Optimization (PPO) agent in the AntMaze environment from `gymnasium-robotics`.


## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hfzizz/hrl_antmaze
    cd hrl_antmaze
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate  

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Training Agents

*   **Train HAC Agent:**
    To train the custom HAC agent:
    ```bash
    python simplified_hrl/train.py
    ```
    This script will run the HAC agent training loop as defined in ['train.py`].

*   **Train PPO Agent:**
    To train the PPO agent using Stable Baselines3:
    ```bash
    python simplified_hrl/train_ppo.py --timesteps 1000000
    ```
    You can adjust the `--timesteps` argument. Logs and models will be saved in `logs/ppo` and `saved_models` respectively. See [`train_ppo.py`] for more options.

### Comparing Agents

The [`compare.py`] script provides functionality to train both HAC and PPO, evaluate them, and record videos.

*   **Train and Evaluate Both:**
    ```bash
    python compare.py train_and_evaluate
    ```

*   **Evaluate Pre-trained Agents:**
    (Assumes models are saved in `saved_models/`)
    ```bash
    python compare.py evaluate_only
    ```

*   **Record Videos of Pre-trained Agents:**
    ```bash
    python compare.py record_only
    ```
    Videos will be saved in `logs/hac_videos/` and `logs/ppo_videos/`.

### TensorBoard Logging

Training progress for both agents can be monitored using TensorBoard:
```bash
tensorboard --logdir logs
```
Navigate to `http://localhost:6006/` in your browser.

## Agents

### Hierarchical Actor-Critic (HAC)
A custom two-layer Hierarchical Actor-Critic agent.
*   **Low-level Layer:** Learns to achieve subgoals (represented as target states) by executing primitive actions in the environment.
*   **High-level Layer:** Learns to set appropriate subgoals for the low-level layer to achieve the overall task goal.
Implementation details can be found in:
*   [`hac_agent.py`]
*   [`layer.py`]
*   [`network.py`]
*   [`replaybuffer.py`]

### Proximal Policy Optimization (PPO)
A standard PPO agent implemented using the Stable Baselines3 library. It operates on a flat policy structure.
Training and evaluation are handled by [`train_ppo.py`].

## Environment

The project uses the `AntMaze_UMaze-v5` environment from `gymnasium-robotics`. The maze configuration is defined within the scripts e.g., in [`environment.py`]. The agent's task is to navigate the Ant robot from a starting position to a goal position within the maze.

The observation space includes the agent's current state (position, orientation, velocities) and the desired goal state. The action space consists of continuous joint motor controls.
