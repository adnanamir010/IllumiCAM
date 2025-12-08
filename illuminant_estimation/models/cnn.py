import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class IlluminantCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(IlluminantCNN, self).__init__()

        # Conv Block 1: Conv(32, 10x10) -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=10)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: Conv(64, 7x7) -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: Conv(96, 5x5) -> BN -> ReLU
        self.conv3 = nn.Conv2d(64, 96, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(96)

        # Conv Block 4: Conv(128, 5x5) -> BN -> ReLU
        self.conv4 = nn.Conv2d(96, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(128)

        # Conv Block 5: Conv(256, 3x3) -> BN -> ReLU
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(256)

        # Global Max Pooling
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Fully Connected
        self.fc1 = nn.Linear(256, 1024)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.bn3(self.conv3(x)))

        # Block 4
        x = self.relu(self.bn4(self.conv4(x)))

        # Block 5
        x = self.relu(self.bn5(self.conv5(x)))

        # Global pooling + FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class IllumiCam3(nn.Module):
    def __init__(self, num_classes=5):
        super(IllumiCam3, self).__init__()

        # Conv Block 1: Conv(32, 10x10) -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=10)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: Conv(64, 7x7) -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: Conv(96, 5x5) -> BN -> ReLU
        self.conv3 = nn.Conv2d(64, 96, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(96)

        # Conv Block 4: Conv(128, 5x5) -> BN -> ReLU
        self.conv4 = nn.Conv2d(96, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(128)

        # Conv Block 5: Conv(256, 3x3) -> BN -> ReLU
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(256)

        # Global Average Pooling (changed from Max)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected (unchanged)
        self.fc1 = nn.Linear(256, 1024)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.bn3(self.conv3(x)))

        # Block 4
        x = self.relu(self.bn4(self.conv4(x)))

        # Block 5
        x = self.relu(self.bn5(self.conv5(x)))

        # Global pooling + FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Hardcoded configuration
NUM_CLASSES = 5
DROPOUT_RATE = 0.25


class ConfidenceWeightedCNN(nn.Module):
    """
    CNN with Confidence-Weighted Pooling.
    
    Instead of Max or Avg pooling, this network learns a confidence map
    to weight the spatial features before aggregation.
    """
    
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(ConfidenceWeightedCNN, self).__init__()

        # Shared Feature Extractor (Same as original)
        # Conv Block 1: Conv(32, 10x10) -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=10)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: Conv(64, 7x7) -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: Conv(96, 5x5) -> BN -> ReLU
        self.conv3 = nn.Conv2d(64, 96, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(96)

        # Conv Block 4: Conv(128, 5x5) -> BN -> ReLU
        self.conv4 = nn.Conv2d(96, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm2d(128)

        # Conv Block 5: Conv(256, 3x3) -> BN -> ReLU
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(256)
        
        # --- BRANCHING POINT ---
        
        # 1. Confidence Branch: Outputs a 1-channel mask (H x W)
        # We use a 1x1 conv to collapse 256 channels to 1
        self.confidence_conv = nn.Conv2d(256, 1, kernel_size=1)
        
        # 2. Fully Connected Layers (applied AFTER pooling)
        self.fc1 = nn.Linear(256, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Feature Extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x))) # Shape: [B, 256, H, W]
        
        # --- CONFIDENCE POOLING ---
        
        # 1. Compute Confidence Map
        # Shape: [B, 1, H, W]
        confidence = self.confidence_conv(x)
        
        # 2. Normalize confidence (softmax over spatial dimensions H*W)
        # This ensures weights sum to 1 for each image
        B, C, H, W = x.shape
        confidence_flat = confidence.view(B, 1, -1) # [B, 1, H*W]
        weights = F.softmax(confidence_flat, dim=2) # [B, 1, H*W]
        
        # 3. Reshape weights back to spatial for later visualization if needed
        spatial_weights = weights.view(B, 1, H, W)
        
        # 4. Weighted Pooling
        # Multiply features by weights and sum over spatial dimensions
        x_flat = x.view(B, C, -1) # [B, 256, H*W]
        
        # [B, 256, H*W] * [B, 1, H*W] -> [B, 256, H*W] -> sum -> [B, 256]
        weighted_features = torch.sum(x_flat * weights, dim=2)
        
        # --- CLASSIFICATION ---
        x = self.relu(self.fc1(weighted_features))
        x = self.dropout(x)
        x = self.fc2(x)

        return x, spatial_weights

    def get_confidence_map(self, x):
        """Helper to extract just the confidence map for visualization."""
        # Run forward pass up to confidence
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        confidence = self.confidence_conv(x)
        return confidence


class ColorConstancyCNN(nn.Module):
    def __init__(self, K, pretrained=True):
        """
        K: The number of illuminant clusters (output classes).
        pretrained: Whether to use ImageNet pretrained weights.
        """
        super(ColorConstancyCNN, self).__init__()
        
        # Load Pretrained AlexNet (Closest architecture to the paper's custom model)
        # The paper describes a 5-conv layer network very similar to AlexNet/CaffeNet
        if pretrained:
            print("Loading ImageNet pretrained AlexNet weights...")
            self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.alexnet = models.alexnet(weights=None)

        # Feature extractor (Conv1 - Conv5 + MaxPool)
        self.features = self.alexnet.features
        
        # The classifier in AlexNet is:
        # (0): Dropout(p=0.5, inplace=False)
        # (1): Linear(in_features=9216, out_features=4096, bias=True)
        # (2): ReLU(inplace=True)
        # (3): Dropout(p=0.5, inplace=False)
        # (4): Linear(in_features=4096, out_features=4096, bias=True)
        # (5): ReLU(inplace=True)
        # (6): Linear(in_features=4096, out_features=1000, bias=True)
        
        # We replace the classifier to match the paper's specific FC structure (if different)
        # or just modify the last layer. The paper specifies:
        # FC6 (4096) -> FC7 (4096) -> FC8 (K)
        # This matches AlexNet exactly, except for the last layer.
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096), # AlexNet uses 6x6 adaptive pooling
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, K), # Replace 1000 with K classes
        )
        
        # Initialize the new classifier layers
        self._initialize_classifier()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def _initialize_classifier(self):
        # Initialize only the new classifier layers
        # Copy weights from pre-trained AlexNet classifier layers 1 and 4 if possible,
        # otherwise initialize randomly.
        
        # 1. Copy FC6 and FC7 weights from pretrained model to speed up convergence
        # self.classifier[1].weight.data = self.alexnet.classifier[1].weight.data
        # self.classifier[1].bias.data = self.alexnet.classifier[1].bias.data
        # self.classifier[4].weight.data = self.alexnet.classifier[4].weight.data
        # self.classifier[4].bias.data = self.alexnet.classifier[4].bias.data
        
        # For the final layer (K classes), initialize with Normal dist
        nn.init.normal_(self.classifier[6].weight, 0, 0.01)
        nn.init.constant_(self.classifier[6].bias, 0)

def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
