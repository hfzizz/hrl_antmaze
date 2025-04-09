import gymnasium as gym
import gymnasium_robotics
import yaml
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configurations for PPO and SAC
    ppo_config = load_config('configs/ppo_config.yaml')
    sac_config = load_config('configs/sac_config.yaml')

    # Create the AntMaze environment
    env = gym.make('AntMaze_UMaze-v5', render_mode="human")

    # Initialize agents
    ppo_agent = PPOAgent(env.observation_space, env.action_space, ppo_config)
    sac_agent = SACAgent(env.observation_space, env.action_space, sac_config)

    # Training loop for PPO
    print("Training PPO agent...")
    for episode in range(ppo_config['num_episodes']):
        observation, info = env.reset()
        done = False
        while not done:
            action = ppo_agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            ppo_agent.store_transition(observation, action, reward, terminated)
            done = terminated or truncated
        ppo_agent.update_policy()

    # Training loop for SAC
    print("Training SAC agent...")
    for episode in range(sac_config['num_episodes']):
        observation, info = env.reset()
        done = False
        while not done:
            action = sac_agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            sac_agent.store_transition(observation, action, reward, terminated)
            done = terminated or truncated
        sac_agent.update_policy()

    print("Training complete.")
    env.close()

if __name__ == "__main__":
    main()