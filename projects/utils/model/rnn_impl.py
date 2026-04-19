import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.module):
    def __init(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        b = x.size(0)
        seq_len = x.size(1)
        
        h_t = torch.zero(b, self.hidden_size).to(x.device)
        
        for t in range(seq_len):
            current_row = x[:, t, :]
            h_t = self.tanh(self.i2h(current_row)+self.h2h(h_t))
    
        out = self.fc(h_t)
        return out
    
    


model = RNN(input_size=28, hidden_size=128, num_classes=10)
img = torch.randn(4, 64, 28, 28)

output = model(img)