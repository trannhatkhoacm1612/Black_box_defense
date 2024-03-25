import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained VGG16 model
vgg_model = models.vgg16(pretrained=True)
vgg_features = vgg_model.features

# Freeze the convolutional layers
for param in vgg_features.parameters():
    param.requires_grad = False

# Define custom regression and classification heads
regression_head = nn.Sequential(
    nn.Linear(512 * 4 * 4, 128),  # Input shape based on VGG feature map size
    nn.ReLU(),
    nn.Linear(128, 8)  # 4 pairs of coordinates
)

classification_head = nn.Sequential(
    nn.Linear(8, 4),  # assuming num_classes is defined
    nn.Softmax(dim=1)
)

# Combine VGG features with custom heads
class VGGCustom(nn.Module):
    def __init__(self, vgg_features, regression_head, classification_head):
        super(VGGCustom, self).__init__()
        self.features = vgg_features
        self.regression_head = regression_head
        self.classification_head = classification_head

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        coordinates_output = self.regression_head(x)
        classification_output = self.classification_head(coordinates_output)
        return coordinates_output, classification_output

# Create the model
model = VGGCustom(vgg_features, regression_head, classification_head)
input_image = torch.randn(1, 3, 128, 128)
output = model(input_image)
print(output)