# model.py

import torch.nn as nn

class StationSignLanguageModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(StationSignLanguageModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3)  # input_dim = 126
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm1 = nn.LSTM(64, 32, batch_first=True)
        self.lstm2 = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch_size, frames, 126)
        x = x.permute(0, 2, 1)  # Reshape for Conv1D: (batch_size, 126, frames)
        x = self.pool(self.relu(self.conv1(x)))  # Conv1D + ReLU + MaxPool
        x = x.permute(0, 2, 1)  # Reshape for LSTM: (batch_size, frames', 64)
        x, _ = self.lstm1(x)  # LSTM 1
        x, _ = self.lstm2(x)  # LSTM 2
        x = x[:, -1, :]  # Take the last output from LSTM
        x = self.fc(x)  # Fully connected layer
        return x