# Define behavior cloning model
import torch
import torch.nn as nn

class BehaviorCloningModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(BehaviorCloningModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))  # Output in range [-1, 1]
