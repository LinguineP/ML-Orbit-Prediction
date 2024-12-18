import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the model
class ffNNet(nn.Module):
    def __init__(self):
        super(ffNNet, self).__init__()
        self.fc1 = nn.Linear(27, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.output = nn.Linear(100, 3)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

def train_nn(model, inputs, targets, optimizer, criterion=nn.MSELoss(),max_grad_norm=1.0,scheduler=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model.to(device)
    
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    
    outputs = model(inputs)
    
    
    
    
    loss = criterion(outputs, targets)

    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    

    
    return loss.item(),outputs


