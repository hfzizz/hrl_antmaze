import copy
import numpy as np
import torch
import torch.nn.functional as F
from network import Actor, Critic
from replaybuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Layer:
    def __init__(self, state_dim, action_dim, goal_dim, max_action, layer_number, max_goal_threshold=0.05, discount=0.98, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, actor_lr=3e-4, critic_lr=3e-4, buffer_size=int(1e6), exploration_noise=0.1, layer_name=""):
        self.layer_number = layer_number
        self.layer_name = layer_name if layer_name else f"Layer_{layer_number}"
        
        self.actor = Actor(state_dim, action_dim, goal_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, action_dim, goal_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.base_exploration_noise = 0.4
        self.exploration_noise = 0.4
        self.total_updates = 0
        
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, goal_dim, buffer_size)
        
    def select_action(self, state, goal, exploration=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        action = self.actor(state, goal).cpu().data.numpy().flatten()
        
        if exploration:
            # Add exploration noise - higher noise for better exploration
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            
            # Clip to ensure actions are within valid range
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def store_transition(self, state, action, next_state, reward, done, goal):
        self.replay_buffer.add(state, action, next_state, reward, done, goal)
    
    def train(self, batch_size=256):
        # If not enough samples in buffer yet, skip training
        if self.replay_buffer.size < batch_size:
            return
            
        state, action, next_state, reward, not_done, goal = self.replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (self.actor_target(next_state, goal) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, goal)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, goal)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_updates % self.policy_freq == 0:
            # Compute actor loss - THIS IS WHERE THE ERROR IS HAPPENING
            # Detach critic gradients to ensure we don't update it during actor optimization
            actor_loss = -self.critic.Q1(state, self.actor(state, goal), goal).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            # Add retain_graph=True to fix the double backward issue
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()
            
            # Update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Increment update counter
        self.total_updates += 1