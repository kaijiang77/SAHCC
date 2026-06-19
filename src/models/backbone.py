from torch import nn

from src.models import vgg


class VGGBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, name: str, return_interm_layers: bool = True):
        super().__init__()
        features = list(backbone.features.children())
        if not return_interm_layers:
            raise ValueError("Only return_interm_layers=True is supported")

        if name == "vgg16_bn":
            self.body1 = nn.Sequential(*features[:13])
            self.body2 = nn.Sequential(*features[13:23])
            self.body3 = nn.Sequential(*features[23:33])
            self.body4 = nn.Sequential(*features[33:43])
        elif name == "vgg16":
            self.body1 = nn.Sequential(*features[:9])
            self.body2 = nn.Sequential(*features[9:16])
            self.body3 = nn.Sequential(*features[16:23])
            self.body4 = nn.Sequential(*features[23:30])
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x):
        out = []
        for layer in (self.body1, self.body2, self.body3, self.body4):
            x = layer(x)
            out.append(x)
        return out


def build_backbone(backbone_name: str):
    if backbone_name == "vgg16_bn":
        net = vgg.vgg16_bn(pretrained=True)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return VGGBackbone(net, name=backbone_name, return_interm_layers=True)
