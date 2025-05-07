import os
import time
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Set up directories
log_dir = "logs/ppo"
model_dir = "saved_models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Create TensorBoard writer
writer = SummaryWriter(log_dir=log_dir)

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
            self.writer.add_scalar("metrics/episode_reward", self.current_episode_reward, self.episode_count)
            self.writer.add_scalar("metrics/episode_length", self.current_episode_length, self.episode_count)
            
            # Log moving averages
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.writer.add_scalar("metrics/avg_reward_last_10", avg_reward, self.episode_count)
            
            # Reset counters
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_count += 1
            
            # Print progress
            if self.episode_count % 10 == 0:
                print(f"Episode {self.episode_count}")
                if len(self.episode_rewards) >= 10:
                    print(f"  Avg reward (last 10): {avg_reward:.2f}")
                    print(f"  Success rate (last 10): {self.locals['infos'][0].get('is_success', 0):.2f}")
        
        return True

def train_ppo(total_timesteps=1000000):
    """Train PPO agent with TensorBoard logging"""
    print("\n=== Starting PPO Training ===")
    print(f"Training for {total_timesteps} timesteps...")
    print(f"Logs will be saved to {log_dir}")
    print(f"Models will be saved to {model_dir}")
    
    # Create vectorized environment for PPO
    env = make_vec_env(
        lambda: gym.make("AntMaze_UMaze-v5", maze_map=MAZE), 
        n_envs=4  # Multiple environments for faster training
    )
    
    # Create PPO model with improved hyperparameters for AntMaze
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Create callbacks
    tensorboard_callback = TensorboardCallback(writer)
    
    # Save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save every 100k steps
        save_path=model_dir,
        name_prefix="ppo_antmaze",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Train model
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[tensorboard_callback, checkpoint_callback],
        progress_bar=True
    )
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = os.path.join(model_dir, "ppo_antmaze_final")
    model.save(final_model_path)
    
    # Log total training time
    writer.add_scalar("metrics/total_training_time_minutes", training_time/60, 0)
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Final model saved to {final_model_path}")
    
    env.close()
    writer.close()
    
    # Return path to final model
    return final_model_path

def evaluate_trained_model(model_path, num_episodes=10):
    """Evaluate the trained model"""
    print("\n=== Evaluating Trained Model ===")
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create environment for evaluation
    env = gym.make("AntMaze_UMaze-v5", maze_map=MAZE, render_mode="human")
    
    # Evaluation loop
    successes = 0
    rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if done and info.get('is_success', False):
                successes += 1
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Success = {info.get('is_success', False)}")
    
    success_rate = successes / num_episodes
    avg_reward = np.mean(rewards)
    
    print(f"\nEvaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Success Rate: {success_rate:.2f}")
    
    env.close()
    return success_rate, avg_reward

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a PPO agent for AntMaze")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model after training")
    parser.add_argument("--load", type=str, default=None, help="Path to load a pre-trained model instead of training")
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    if args.load:
        model_path = args.load
        print(f"Loading pre-trained model from {model_path}")
    else:
        model_path = train_ppo(args.timesteps)
    
    if args.evaluate or args.load:
        evaluate_trained_model(model_path)