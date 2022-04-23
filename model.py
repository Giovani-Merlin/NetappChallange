import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, 1)

    def forward(self, x, images_size):
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        original_features = []
        actual_index = 0
        for i in images_size:
            original_features.append(torch.max(pooled_features[actual_index:actual_index+i],0)[0])
            actual_index = i
        output = self.classifer(torch.vstack(original_features))[:,0]
        return output
