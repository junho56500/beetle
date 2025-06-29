import torch

batch_size = 8
feature_size = 128

in_features = 128
out_features = 64

bias=True

layer1 = torch.nn.Linear(in_features=in_features,
                        out_features=out_features,
                        bias=bias)
print(layer1.weight.shape)
print(layer1.bias.shape)

in_features = 64
act_fnout_feature = 16
layer2 = torch.nn.Linear(in_features=64,
                out_features=16,
                bias=bias)

in_features = 16
act_fnout_feature = 1
layer3 = torch.nn.Linear(in_features=16,
                out_features=1,
                bias=bias)

#activation funcions
# act_fn = torch.nn.Sigmoid()
# act_fn1 = torch.nn.ReLU()
act_fn1 = torch.nn.Tanh()
act_fn2 = torch.nn.Tanh()
act_fn3 = torch.nn.Sigmoid()

model = torch.nn.Sequential(
    torch.nn.Linear(in_features=128,
                    out_features=64,
                    bias=True),
    torch.nn.Tanh(),
    torch.nn.Linear(in_features=64,
                    out_features=16,
                    bias=True),
    torch.nn.Tanh(),
    torch.nn.Linear(in_features=16,
                    out_features=1,
                    bias=True),
    torch.nn.Sigmoid()
)


x = torch.rand(batch_size,feature_size)
print(x.shape)

output1 = layer1(x)
output2 = x @ layer1.weight.T + layer1.bias

print(output1[0,0])
print(output2[0,0])

out1 = act_fn1(output1)
print(out1.shape)

out2 = act_fn1(output2)
print(out2.shape)

logits = torch.rand(1,20)

softmax = torch.nn.Softmax(dim=1)

pred = softmax(logits)

print(torch.sum(pred,dim=1).item())

print(pred)

out1 = act_fn1(layer1(x))

out2 = act_fn2(layer2(out1))

out3 = act_fn3(layer3(out2))

out4 = model(x)
        
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers=torch.nn.Sequential(
            torch.nn.Linear(in_features=128,
                            out_features=64,
                            bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=64,
                            out_features=16,
                            bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=16,
                            out_features=1,
                            bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def main():
    x=torch.rand(32,128)
    nn = NeuralNetwork()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters (including non-trainable): {total_params:,}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of TRAINABLE parameters: {trainable_params:,}")
    
    for param in model.parameters():
        print(f"Shape: {param.shape}, Requires grad: {param.requires_grad}")
    
    #print(nn(x))

if __name__ == '__main__':
    main()