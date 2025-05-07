import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from torch import optim

from hac_agent import HACAgent  # The two-layer agent

MAZE = [[1, 1, 1, 1, 1],
        [1, "g", 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, "r", 0, 0, 1],
        [1, 1, 1, 1, 1]]

# Initialize environment
env = gym.make("AntMaze_UMaze-v5", maze_map=MAZE, render_mode="human")
state_dim = env.observation_space["observation"].shape[0]
action_dim = env.action_space.shape[0]
goal_dim = env.observation_space["desired_goal"].shape[0]
max_action = float(env.action_space.high[0])

# Initialize HAC agent with stability rewards
agent = HACAgent(state_dim, action_dim, goal_dim, max_action, time_limit=50)

## Training parameters
num_episodes = 1000
batch_size = 256
exploration_episodes = 100  # Increase from 50 to 100

# Training loop using the hierarchical structure
for episode in range(num_episodes):
    # Use higher exploration for the first episodes
    if episode < exploration_episodes:
        exploration_scale = 2.0 - (1.5 * episode / exploration_episodes)  # Start at 2.0, decay to 0.5
    else:
        exploration_scale = 0.3
        
    success = agent.run_episode(env, episode, exploration_scale=exploration_scale)