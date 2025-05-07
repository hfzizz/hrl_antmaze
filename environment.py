import gymnasium as gym
import gymnasium_robotics
import numpy as np

MAZE =  [[1,1,1,1,1],
         [1,"g",0,0,1],
         [1,1,1,0,1],
         [1,"r",0,0,1],
         [1,1,1,1,1]]

env = gym.make("AntMaze_UMaze-v5", maze_map=MAZE, render_mode="human")  # use "human" if you want visual
obs, info = env.reset()

# Print to verify
print("Initial observation keys:", obs.keys())
print("Agent position (first 2):", obs['observation'][:2])
print("Goal:", obs['desired_goal'])