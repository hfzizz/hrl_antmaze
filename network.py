import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, max_action, hidden_sizes=[256, 256]):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim + goal_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)
        
        self.max_action = max_action
        
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)  # Combine state and goal
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))  # Output action within [-max_action, max_action]


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + goal_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)
        
        # Q2 architecture (for double Q-learning)
        self.l4 = nn.Linear(state_dim + action_dim + goal_dim, hidden_sizes[0])
        self.l5 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l6 = nn.Linear(hidden_sizes[1], 1)
        
    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=1)
        
        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=1)
        
        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1
