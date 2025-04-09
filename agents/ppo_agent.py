import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = self.build_policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.old_log_probs = None
        self.memory = []

    def build_policy(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state)
        action = action_probs.detach().numpy()[0]
        self.old_log_probs = torch.log(action_probs).detach()
        return action

    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) == 0:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        returns = self.compute_returns(rewards, dones)

        for _ in range(self.K_epochs):
            log_probs = torch.log(self.policy(states))
            ratios = torch.exp(log_probs - self.old_log_probs)

            advantages = returns - log_probs.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def compute_returns(self, rewards, dones):
        returns = []
        discounted_return = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted_return = reward + (self.gamma * discounted_return * (1 - done))
            returns.insert(0, discounted_return)
        return torch.FloatTensor(returns)