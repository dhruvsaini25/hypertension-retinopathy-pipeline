import torch.nn as nn
import torchvision.models as models

def get_model(num_classes):

    model = models.efficientnet_b0(pretrained=True)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model