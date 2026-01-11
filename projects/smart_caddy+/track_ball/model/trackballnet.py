import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict
import torch.optim as optim
from torchvision import datasets, transforms

import torch
print(f"PyTorch version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Prepare Data
transform = transforms.Compose([
    transforms.Resize((7, 7)),            # Resize from 28x28 to 7x7
    transforms.Grayscale(num_output_channels=3), # Convert 1 channel to 3 channels
    transforms.ToTensor(),                # Convert to tensor [3, 7, 7]
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)


class DetectionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 1. BACKBONE: Load ResNet and choose layers to extract
        # These are standard exit points for ResNet (1/4, 1/8, 1/16, 1/32 scale)
        backbone_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        return_nodes = {
            'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'
        }
        self.backbone = create_feature_extractor(backbone_model, return_nodes=return_nodes)
        
        # 2. NECK: Connect FPN
        # ResNet-18 layers have [64, 128, 256, 512] output channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512], 
            out_channels=256
        )
        
        # 3. HEAD: A simple classifier or detector head
        # Here we use a Global Average Pool and a Linear layer as a placeholder
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x) # Returns dict of 4 tensors
        
        # Fuse features in FPN
        fpn_features = self.fpn(features) # Returns dict of 4 fused tensors
        
        # Use the highest-level feature (usually '3') for global classification
        last_feat = fpn_features['3']
        x = self.avgpool(last_feat)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        
        return logits

model = DetectionModel().to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 3. The Training Loop (The "Reference" Pattern)
print("Starting training...")
model.train()
for epoch in range(1, 3): # 2 quick epochs
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()        # Reset gradients
        output = model(data)         # Forward pass
        loss = criterion(output, target) # Calculate loss
        loss.backward()              # Backward pass (Backprop)
        optimizer.step()             # Update weights
        
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

print("Training Complete!")