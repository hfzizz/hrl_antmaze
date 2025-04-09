def evaluate_agent(env, agent, num_episodes=10):
    total_rewards = []
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward: {episode_reward}")
    
    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

def evaluate_ppo(env, ppo_agent, num_episodes=10):
    print("Evaluating PPO agent...")
    return evaluate_agent(env, ppo_agent, num_episodes)

def evaluate_sac(env, sac_agent, num_episodes=10):
    print("Evaluating SAC agent...")
    return evaluate_agent(env, sac_agent, num_episodes)