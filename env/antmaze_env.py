import gymnasium as gym
import gymnasium_robotics
import time

# Register the robotics environments
gym.register_envs(gymnasium_robotics)

print("Creating environment...")
# Create a simple AntMaze environment
env = gym.make('AntMaze_UMaze-v5', render_mode="human")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

print("Resetting environment...")
observation, info = env.reset()
print("Environment reset complete")

# Run a few random actions to verify everything works
print("Starting simulation loop...")
for i in range(10000):
    print(f"Step {i}")
    action = env.action_space.sample()  # Sample a random action
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    time.sleep(0.1)  # Add delay to see what's happening
    
    if terminated or truncated:
        print("Episode ended, resetting...")
        observation, info = env.reset()

print("Closing environment...")
env.close()
print("Environment closed")