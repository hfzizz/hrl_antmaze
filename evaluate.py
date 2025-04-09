import gymnasium as gym
import numpy as np
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent

def evaluate_agent(agent, env_name, num_episodes=10):
    env = gym.make(env_name, render_mode="human")
    total_rewards = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

    env.close()

if __name__ == "__main__":
    # Load your trained PPO or SAC agent here
    # For example:
    # ppo_agent = PPOAgent.load('path_to_trained_ppo_model')
    # sac_agent = SACAgent.load('path_to_trained_sac_model')

    # Evaluate the PPO agent
    # evaluate_agent(ppo_agent, 'AntMaze_UMaze-v5')

    # Evaluate the SAC agent
    # evaluate_agent(sac_agent, 'AntMaze_UMaze-v5')