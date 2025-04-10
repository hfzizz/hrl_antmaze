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

class BatchPPOAgent:
    def __init__(self, state_dim, action_dim, n_envs=8, action_std_init=0.6, lr=3e-4, gamma=0.99, 
                eps_clip=0.2, K_epochs=4, update_timestep=4000, gae_lambda=0.95, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.action_std = action_std_init
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = int(K_epochs)  # Ensure this is an integer
        self.update_timestep = max(1, update_timestep // n_envs)  # Adjust for parallel envs
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
        
        # For batched processing, we maintain separate memory for each environment
        self.memory = [[] for _ in range(n_envs)]
        self.timestep = 0
    
    def select_action(self, state):
        """Select action for a single state (used for evaluation)"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action, _ = self.policy_old.act(state)
        
        return action.detach().cpu().numpy().flatten()
    
    def select_actions(self, states):
        """Select actions for a batch of states from multiple environments"""
        with torch.no_grad():
            # Convert states to tensor
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions, action_logprobs = self.policy_old.act(states_tensor)
        
        # Store transitions in memory (reward and done will be filled later)
        for i in range(self.n_envs):
            self.memory[i].append((
                states_tensor[i].unsqueeze(0),  # Keep state tensor shape consistent
                actions[i].unsqueeze(0),        # Keep action tensor shape consistent 
                action_logprobs[i].unsqueeze(0), # Keep logprob tensor shape consistent
                0.0,  # Initial reward (will be updated)
                False  # Initial done flag (will be updated)
            ))
        
        # Return numpy actions for the environment
        return actions.detach().cpu().numpy()
    
    def update_rewards_dones(self, rewards, dones):
        """Update the last transition in each environment's memory with rewards and dones"""
        for i in range(self.n_envs):
            if self.memory[i]:  # Check that there are transitions stored for this environment
                state, action, action_logprob, _, _ = self.memory[i][-1]
                self.memory[i][-1] = (state, action, action_logprob, rewards[i], dones[i])
        
        self.timestep += 1
        
        # Update policy if it's time
        if self.timestep % self.update_timestep == 0:
            self.update()
    
    def update(self):
        """Update policy using collected experiences from all environments"""
        # Collect all experiences from all environments
        all_states, all_actions, all_logprobs, all_rewards, all_dones = [], [], [], [], []
        
        for env_memory in self.memory:
            if not env_memory:  # Skip if this environment hasn't collected any experience
                continue
                
            # Extract states, actions, and logprobs
            states = torch.cat([t[0] for t in env_memory])
            actions = torch.cat([t[1] for t in env_memory])
            logprobs = torch.cat([t[2] for t in env_memory])
            
            # Extract rewards and dones
            rewards = [t[3] for t in env_memory]
            dones = [t[4] for t in env_memory]
            
            # Add to our collections
            all_states.append(states)
            all_actions.append(actions)
            all_logprobs.append(logprobs)
            all_rewards.extend(rewards)
            all_dones.extend(dones)
        
        if not all_states:  # No experiences collected
            print("No experiences collected for update, skipping")
            return
            
        # Concatenate all experiences
        old_states = torch.cat(all_states)
        old_actions = torch.cat(all_actions)
        old_logprobs = torch.cat(all_logprobs)
        
        # Compute returns with advantage
        with torch.no_grad():
            state_values = self.policy_old.critic(old_states).squeeze(-1)
        
        rewards = np.array(all_rewards)
        dones = np.array(all_dones)
        values = state_values.cpu().numpy()

        # Debug information
        print(f"Update called: {len(rewards)} transitions, {sum(dones)} episode endings")

        # Compute returns with GAE
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1 or dones[t]:
                next_value = 0
                next_advantage = 0
            else:
                next_value = values[t + 1]
                next_advantage = advantages[t + 1]
                
            # Calculate TD error and advantage
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (1 - dones[t])
            returns[t] = advantages[t] + values[t]
        
        # Convert to tensors
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Optimize policy for K epochs
        for epoch in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Ensure proper shapes
            state_values = state_values.squeeze()
            
            # Calculate ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Use pre-calculated advantages (don't recalculate here)
            
            # PPO update
            surr1 = ratios * advantages_tensor
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_tensor
            
            # Final PPO loss (negative because we're using gradient descent)
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.MseLoss(state_values, returns_tensor)
            entropy_loss = -0.01 * dist_entropy.mean()  # Encourage exploration
            
            loss = actor_loss + critic_loss + entropy_loss
            
            if epoch == 0:  # Print diagnostics on first epoch
                print(f"PPO Update - Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
                print(f"Mean advantage: {advantages_tensor.mean().item():.4f}, Return: {returns_tensor.mean().item():.4f}")
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.memory = [[] for _ in range(self.n_envs)]
        print("Update completed, memory cleared")
    
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
    def load(cls, filepath, state_dim, action_dim, n_envs=8, device=None):
        device = device if device is not None else torch.device("cpu")
        agent = cls(state_dim, action_dim, n_envs=n_envs, device=device)
        
        checkpoint = torch.load(filepath, map_location=device)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.action_std = checkpoint['action_std']
        agent.policy.set_action_std(agent.action_std)
        agent.policy_old.set_action_std(agent.action_std)
        
        return agent