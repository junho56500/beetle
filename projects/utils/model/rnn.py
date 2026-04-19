import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.module):
    def __init(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.rnn(x, h0)
        
        out = self.fc(out[:, -1, :])
        return out

input_size = 28
seq_len = 28
hidden_size = 128
num_layers = 2
num_classes = 2

model = RNN