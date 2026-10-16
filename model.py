import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DOSEModel(nn.Module):
    """Encoder f_theta (ImageNet backbone + projection head) and scalar severity head alpha_psi.

        z = f_theta(x) / ||f_theta(x)||_2,    s = alpha_psi(z)          (Eq. 2)
    """

    def __init__(self, backbone='convnext_tiny', projection_dim=128, pretrained=True):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        if not hasattr(models, backbone):
            raise ValueError(f"Backbone {backbone} not found in torchvision.models.")
        model = getattr(models, backbone)(weights=weights)
        if backbone.startswith('resnet') or backbone.startswith('resnext'):
            backbone_dim = model.fc.in_features
            self.backbone = nn.Sequential(*list(model.children())[:-1])
        elif backbone.startswith('densenet'):
            backbone_dim = model.classifier.in_features
            self.backbone = nn.Sequential(model.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        elif backbone.startswith('convnext'):
            backbone_dim = model.classifier[2].in_features
            self.backbone = nn.Sequential(model.features, model.avgpool)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use resnet*, densenet* or convnext*.")

        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim, bias=True),
        )
        self.ordinal_slope = nn.Linear(projection_dim, 1, bias=False)   # severity head: s = w . z
        self.projection_dim = projection_dim

    def forward(self, x):
        """Returns the L2-normalised embedding z (B, d) and the severity score s (B,)."""
        h = self.backbone(x).view(x.size(0), -1)
        z = F.normalize(self.projection_head(h), dim=1)
        s = self.ordinal_slope(z).squeeze(-1)
        return z, s


def create_model(backbone='convnext_tiny', projection_dim=128, pretrained=True):
    return DOSEModel(backbone=backbone, projection_dim=projection_dim, pretrained=pretrained)
