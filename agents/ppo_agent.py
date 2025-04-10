import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import gymnasium as gym

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6, device=None):
        super(ActorCritic, self).__init__()

        self.device = device if device is not None else torch.device("cpu")

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        ).to(self.device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)

        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_var.size()), new_action_std * new_action_std).to(self.device)

    def forward(self, state):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)

        # Use current action_var to create distribution
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = Normal(action_mean, torch.sqrt(torch.diag(cov_mat[0])))

        # Sample action from disctibution
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=1)

        return action, action_logprob 
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = Normal(action_mean, torch.sqrt(action_var))

        # Log probability of the action
        action_logprobs = dist.log_prob(action).sum(dim=1)

        # Get state value
        state_value = self.critic(state)

        # Calculate entropy
        dist_entropy = dist.entropy().sum(dim=1)

        return action_logprobs, state_value, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, action_std_init=0.6, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4, update_timestep=40000, gae_lambda=0.95, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_std = action_std_init
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.update_timestep = update_timestep
        self.gae_lambda = gae_lambda

        # Set device
        self.device = device if device is not None else torch.device("cpu")

        self.policy = ActorCritic(state_dim, action_dim, action_std_init, device=self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr},
            {'params': self.policy.critic.parameters(), 'lr': self.lr}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init, device=self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.memory = []
        self.timestep = 0
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action, action_logprob = self.policy_old.act(state)
        
        self.memory.append((state, action, action_logprob, 0, False)) # reward and done will be updated later

        return action.detach().cpu().numpy().flatten()
    
    def update_reward_done(self, reward, done):
        """Update the last transition in memory with actual reward and done flag"""
        if self.memory:
            state, action, action_logprob, _, _ = self.memory[-1]
            self.memory[-1] = (state, action, action_logprob, reward, done)
            
        self.timestep += 1
        
        # Update policy if it's time
        if self.timestep % self.update_timestep == 0:
            self.update()
    
    def store_transition(self, state, action, logprob, reward, done):
        self.memory.append((state, action, logprob, reward, done))
    
    def update(self):
        if len(self.memory) < self.update_timestep:
            return
            
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for _, _, _, reward, done in reversed(self.memory):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - done))
            rewards.insert(0, discounted_reward)
            
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            
        # Convert list to tensor
        old_states = torch.cat([t[0] for t in self.memory])
        old_actions = torch.cat([t[1] for t in self.memory])
        old_logprobs = torch.cat([t[2] for t in self.memory])
            
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Calculate ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Calculate advantages
            advantages = rewards - state_values.detach()
            
            # PPO update
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final PPO loss (negative because we're using gradient descent)
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.MseLoss(state_values.squeeze(), rewards)
            entropy_loss = -0.01 * dist_entropy.mean()  # Encourage exploration
            
            loss = actor_loss + critic_loss + entropy_loss
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
            
        # Clear memory
        self.memory = []
    
    def decay_action_std(self, min_action_std=0.1, decay_rate=0.05):
        """Decay action standard deviation to help with exploration vs exploitation"""
        self.action_std = max(min_action_std, self.action_std - decay_rate)
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)
        
    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_std': self.action_std
        }, filepath)
        
    @classmethod
    def load(cls, filepath, state_dim, action_dim, device=None):
        device = device if device is not None else torch.device("cpu")
        agent = cls(state_dim, action_dim)
        
        checkpoint = torch.load(filepath, map_location=device)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.action_std = checkpoint['action_std']
        agent.policy.set_action_std(agent.action_std)
        agent.policy_old.set_action_std(agent.action_std)
        
        return agent