import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, config):
        self.actor = Actor(state_dim, action_dim).to(config['device'])
        self.critic1 = Critic(state_dim, action_dim).to(config['device'])
        self.critic2 = Critic(state_dim, action_dim).to(config['device'])
        self.target_critic1 = Critic(state_dim, action_dim).to(config['device'])
        self.target_critic2 = Critic(state_dim, action_dim).to(config['device'])
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config['critic_lr'])
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config['critic_lr'])
        
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.replay_buffer = deque(maxlen=config['buffer_size'])
        self.batch_size = config['batch_size']
        
        self.update_target_networks(tau=1.0)

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(self.actor.fc1.weight.device)
        action = self.actor(state).detach().cpu().numpy()
        action += noise * np.random.randn(*action.shape)
        return np.clip(action, -1, 1)

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.actor.fc1.weight.device)
        actions = torch.FloatTensor(actions).to(self.actor.fc1.weight.device)
        rewards = torch.FloatTensor(rewards).to(self.actor.fc1.weight.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.actor.fc1.weight.device)
        dones = torch.FloatTensor(dones).to(self.actor.fc1.weight.device).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss = -self.critic1(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks()