import torch
import torch.nn as nn
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockPredictorFNN(nn.Module):
    def __init__(self, input_dim):
        super(StockPredictorFNN, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.SiLU()  # Swish
        self.drop1 = nn.Dropout(0.1)
        
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(0.05)
        
        self.layer3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.act3 = nn.SiLU()
        self.drop3 = nn.Dropout(0.0)
        
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        x = self.input_bn(x)
        
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)
        
        out = self.output_layer(x)
        return out
