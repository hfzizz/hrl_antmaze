# HRL Ant Maze Project

This project implements reinforcement learning agents using the Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) algorithms in the AntMaze environment. The AntMaze environment is built using the Gymnasium library and serves as a benchmark for evaluating the performance of the agents.

## Project Structure

```
HRL_Ant_Maze
├── env
│   ├── antmaze_env.py        # Implementation of the AntMaze environment
│   └── __init__.py           # Package initialization
├── agents
│   ├── __init__.py           # Package initialization
│   ├── ppo_agent.py          # PPO agent implementation
│   └── sac_agent.py          # SAC agent implementation
├── utils
│   ├── __init__.py           # Package initialization
│   ├── buffer.py             # Replay buffer for experience storage
│   └── evaluation.py         # Evaluation metrics for agents
├── models
│   ├── __init__.py           # Package initialization
│   ├── actor.py              # Actor network implementation
│   └── critic.py             # Critic network implementation
├── configs
│   ├── ppo_config.yaml       # Configuration for PPO training
│   └── sac_config.yaml       # Configuration for SAC training
├── train.py                  # Main training script for agents
├── evaluate.py               # Script for evaluating trained agents
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training Agents

To train the PPO agent, run:

```bash
python train.py --agent ppo
```

To train the SAC agent, run:

```bash
python train.py --agent sac
```

### Evaluating Agents

To evaluate a trained agent, run:

```bash
python evaluate.py --agent ppo
```

or

```bash
python evaluate.py --agent sac
```

## Agents

### Proximal Policy Optimization (PPO)

PPO is a popular policy gradient method that balances exploration and exploitation by using a clipped objective function. The implementation can be found in `agents/ppo_agent.py`.

### Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic algorithm that uses a stochastic policy and incorporates entropy maximization to encourage exploration. The implementation can be found in `agents/sac_agent.py`.

## Environment

The AntMaze environment is designed to simulate a maze navigation task for an ant-like agent. The environment is implemented in `env/antmaze_env.py` and follows the Gymnasium API.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.