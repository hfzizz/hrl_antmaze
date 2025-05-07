import os
import time
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Import your HAC agent
from hac_agent import HACAgent

# Set up directories
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
hac_log_dir = os.path.join(log_dir, "hac")
ppo_log_dir = os.path.join(log_dir, "ppo")
os.makedirs(hac_log_dir, exist_ok=True)
os.makedirs(ppo_log_dir, exist_ok=True)

# Create TensorBoard writers
hac_writer = SummaryWriter(log_dir=hac_log_dir)
ppo_writer = SummaryWriter(log_dir=ppo_log_dir)

# Define maze layout
MAZE = [[1, 1, 1, 1, 1],
        [1, "g", 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, "r", 0, 0, 1],
        [1, 1, 1, 1, 1]]

# Create environment wrapper for TensorBoard logging
class TensorboardCallback(BaseCallback):
    def __init__(self, writer, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self):
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # If episode is done
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log episode statistics
            self.writer.add_scalar("charts/episode_reward", self.current_episode_reward, self.episode_count)
            self.writer.add_scalar("charts/episode_length", self.current_episode_length, self.episode_count)
            
            # Reset counters
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_count += 1
        
        return True

def train_hac(num_episodes=1000):
    """Train HAC agent with TensorBoard logging"""
    print(f"\nTraining HAC for {num_episodes} episodes...")
    
    # Create environment
    env = gym.make("AntMaze_UMaze-v5", maze_map=MAZE, render_mode="human")
    state_dim = env.observation_space["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize HAC agent
    agent = HACAgent(state_dim, action_dim, goal_dim, max_action, time_limit=50)
    
    # Training parameters
    batch_size = 256
    exploration_episodes = min(100, num_episodes // 10)
    
    # Training metrics
    all_rewards = []
    all_success = []
    training_time = 0
    
    # Training loop
    for episode in range(num_episodes):
        start_time = time.time()
        
        # Use higher exploration for the first episodes
        if episode < exploration_episodes:
            exploration_scale = 2.0 - (1.5 * episode / exploration_episodes)
        else:
            exploration_scale = 0.3
        
        # Run episode
        episode_reward = 0
        success = agent.run_episode(env, episode, exploration_scale=exploration_scale)
        
        # Train both layers periodically
        if episode % 5 == 0 and agent.layers[0].replay_buffer.size >= batch_size:
            agent.train(batch_size)
        
        # Calculate episode duration
        episode_time = time.time() - start_time
        training_time += episode_time
        
        # Log to TensorBoard
        hac_writer.add_scalar("charts/success_rate", float(success), episode)
        hac_writer.add_scalar("charts/exploration_scale", exploration_scale, episode)
        hac_writer.add_scalar("charts/buffer_size_layer0", agent.layers[0].replay_buffer.size, episode)
        hac_writer.add_scalar("charts/buffer_size_layer1", agent.layers[1].replay_buffer.size, episode)
        hac_writer.add_scalar("charts/episode_time", episode_time, episode)
        
        # Track success rate over time
        all_success.append(float(success))
        
        if (episode + 1) % 10 == 0:
            success_rate = np.mean(all_success[-10:])
            hac_writer.add_scalar("charts/success_rate_moving_avg", success_rate, episode)
            print(f"HAC Episode {episode+1}/{num_episodes}, Success Rate (last 10): {success_rate:.2f}")
    
    hac_writer.add_scalar("charts/total_training_time", training_time, num_episodes)
    env.close()
    return agent

def train_ppo(num_timesteps=50000):
    """Train PPO agent with TensorBoard logging"""
    print(f"\nTraining PPO for {num_timesteps} timesteps...")
    
    # Create vectorized environment for PPO
    env = make_vec_env(
        lambda: gym.make("AntMaze_UMaze-v5", maze_map=MAZE), 
        n_envs=1
    )
    
    # Create PPO model
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=ppo_log_dir
    )
    
    # Create callback
    callback = TensorboardCallback(ppo_writer)
    
    # Train model
    start_time = time.time()
    model.learn(total_timesteps=num_timesteps, callback=callback)
    training_time = time.time() - start_time
    
    # Log total training time
    ppo_writer.add_scalar("charts/total_training_time", training_time, num_timesteps)
    
    env.close()
    return model

import imageio
from gymnasium.wrappers import RecordVideo

def evaluate_agent(agent_type, agent, episodes=10, record=True):
    """Evaluate trained agent with optional video recording"""
    print(f"\nEvaluating {agent_type}...")
    
    # Create directory for videos
    videos_dir = os.path.join(log_dir, f"{agent_type.lower()}_videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Create environment with rendering
    if record:
        video_recorder = imageio.get_writer(
            os.path.join(videos_dir, f"{agent_type.lower()}_evaluation.mp4"),
            fps=30
        )
        env = gym.make("AntMaze_UMaze-v5", maze_map=MAZE, render_mode="rgb_array")
    else:
        env = gym.make("AntMaze_UMaze-v5", maze_map=MAZE, render_mode="human")
    
    successes = 0
    for episode in range(episodes):
        if agent_type == "HAC":
            state, _ = env.reset()
            goal = state['desired_goal']
            obs = state['observation']
            agent.latest_observation = obs
            
            # Track steps for HAC (since we don't have a while loop)
            step_count = 0
            
            # Define a custom run_episode function to capture frames
            def custom_run_episode(step_count):
                success = False
                # Run the top layer for the full episode
                success = agent._run_layer(
                    layer_idx=agent.num_layers - 1,
                    state=obs,
                    goal=goal,
                    env=env,
                    episode_reward=0,
                    record_video=record,
                    video_recorder=video_recorder if record else None
                )
                return success
            
            if record:
                # Modify the _run_layer method temporarily to record frames
                orig_run_layer = agent._run_layer
                
                def recording_run_layer(self, layer_idx, state, goal, env, episode_reward, record_video=False, video_recorder=None):
                    layer = self.layers[layer_idx]
                    current_state = state
                    timesteps = 0
                    max_horizon = self.time_limit if layer_idx == 0 else self.time_limit // 2
                    
                    while timesteps < max_horizon:
                        if layer_idx == 0:
                            # Execute primitive action
                            action = layer.select_action(current_state, goal, exploration=False)  # No exploration during evaluation
                            next_state_dict, reward, terminated, truncated, info = env.step(action)
                            
                            # Record frame if at the lowest level
                            if record_video:
                                frame = env.render()
                                video_recorder.append_data(frame)
                            
                            next_state = next_state_dict["observation"]
                            self.latest_observation = next_state
                            
                            # Rest of the low-level logic remains the same
                            stability_reward = self.calculate_stability_reward(next_state)
                            adjusted_reward = reward + stability_reward
                            episode_reward += adjusted_reward
                            
                            self._store_transition_with_hindsight(
                                layer, current_state, action, next_state,
                                adjusted_reward, terminated or truncated, goal
                            )
                            
                            current_state = next_state
                            if terminated or truncated:
                                return False
                        else:
                            # High-level layer logic
                            subgoal = layer.select_action(current_state, goal, exploration=False)
                            subgoal_achieved = self._run_layer(
                                layer_idx - 1,
                                current_state,
                                subgoal,
                                env,
                                episode_reward,
                                record_video,
                                video_recorder
                            )
                            
                            # Rest of high-level logic
                            if not subgoal_achieved and hasattr(env, 'terminated') and env.terminated:
                                return False
                            
                            next_state = self.latest_observation
                            subgoal_distance = np.linalg.norm(next_state[:self.goal_dim] - subgoal)
                            subgoal_reward = 1.0 if subgoal_distance < 0.5 else -0.1
                            
                            self._store_transition_with_hindsight(
                                layer, current_state, subgoal, next_state,
                                subgoal_reward, False, goal
                            )
                            
                            current_state = next_state
                        
                        timesteps += 1
                    
                    # Check if goal was reached at the end of the episode
                    goal_distance = np.linalg.norm(current_state[:self.goal_dim] - goal)
                    goal_reached = goal_distance < 1.0
                    
                    return goal_reached
                
                # Replace the method temporarily
                agent._run_layer = lambda *args, **kwargs: recording_run_layer(agent, *args, **kwargs)
                
                success = agent.run_episode(env, episode, exploration_scale=0)
                
                # Restore original method
                agent._run_layer = orig_run_layer
            else:
                success = agent.run_episode(env, episode, exploration_scale=0)
            
            if success:
                successes += 1
                
        else:  # PPO
            obs, _ = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 500:  # Limit to 500 steps for safety
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                
                # Record frame
                if record:
                    frame = env.render()
                    video_recorder.append_data(frame)
                
                done = terminated or truncated
                step_count += 1
                
                if done and 'success' in info and info['success']:
                    successes += 1
                    break
        
        print(f"{agent_type} Episode {episode+1}/{episodes}, Success: {successes}/{episode+1}")
    
    success_rate = successes / episodes
    print(f"{agent_type} Success Rate: {success_rate:.2f}")
    
    if record:
        video_recorder.close()
        print(f"Video saved to {os.path.join(videos_dir, f'{agent_type.lower()}_evaluation.mp4')}")
    
    env.close()
    return success_rate

def main(mode="train_and_evaluate"):
    """
    Run the comparison with different modes:
    - "train_and_evaluate": Train both agents and then evaluate them (default)
    - "evaluate_only": Only evaluate pre-trained agents
    - "record_only": Only record videos of pre-trained agents
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Training parameters
    hac_episodes = 1000
    ppo_timesteps = hac_episodes * 50  # Assuming average HAC episode length of 50 steps
    
    if mode == "train_and_evaluate" or mode == "train_only":
        # Train agents
        hac_agent = train_hac(hac_episodes)
        ppo_agent = train_ppo(ppo_timesteps)
        
        # Save trained agents
        save_agents(hac_agent, ppo_agent)
        
        if mode == "train_only":
            print("Training completed and models saved. Skipping evaluation.")
            return
    else:
        # Load pre-trained agents
        hac_agent, ppo_agent = load_agents()
    
    # Evaluate agents
    if mode == "record_only":
        # Only record videos, don't compute metrics
        evaluate_agent("HAC", hac_agent, episodes=3, record=True)
        evaluate_agent("PPO", ppo_agent, episodes=3, record=True)
    else:
        # Full evaluation with metrics
        hac_success_rate = evaluate_agent("HAC", hac_agent, record=(mode == "evaluate_and_record"))
        ppo_success_rate = evaluate_agent("PPO", ppo_agent, record=(mode == "evaluate_and_record"))
        
        # Final logging
        print("\n--- Final Comparison ---")
        print(f"HAC Success Rate: {hac_success_rate:.2f}")
        print(f"PPO Success Rate: {ppo_success_rate:.2f}")
    
    # Close TensorBoard writers
    hac_writer.close()
    ppo_writer.close()

def save_agents(hac_agent, ppo_agent):
    """Save trained agents to disk"""
    os.makedirs("saved_models", exist_ok=True)
    
    # Save PPO (Stable Baselines3 has built-in save functionality)
    ppo_agent.save("saved_models/ppo_agent")
    
    # Save HAC agent layers (requires custom pickle implementation)
    import pickle
    with open("saved_models/hac_agent.pkl", "wb") as f:
        pickle.dump(hac_agent, f)
    
    print("Agents saved successfully.")

def load_agents():
    """Load trained agents from disk"""
    # Load PPO
    ppo_agent = PPO.load("saved_models/ppo_agent")
    
    # Load HAC
    import pickle
    with open("saved_models/hac_agent.pkl", "rb") as f:
        hac_agent = pickle.load(f)
    
    print("Agents loaded successfully.")
    return hac_agent, ppo_agent

if __name__ == "__main__":
    import sys
    
    # Default mode is train_and_evaluate
    mode = "train_and_evaluate" if len(sys.argv) <= 1 else sys.argv[1]
    
    if mode not in ["train_and_evaluate", "train_only", "evaluate_only", "record_only", "evaluate_and_record"]:
        print(f"Unknown mode: {mode}")
        print("Available modes: train_and_evaluate, train_only, evaluate_only, record_only, evaluate_and_record")
        sys.exit(1)
        
    main(mode)

if __name__ == "__main__":
    main()