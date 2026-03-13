import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model(num_classes):

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # Freeze most layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze last 2 blocks
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model