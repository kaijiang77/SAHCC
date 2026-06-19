import torch
from torch import nn


class ClassificationAndRegressionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchor: int = 1,
        num_classes: int = 1,
        hidden_dim: int = 256,
        offset_scale: float = 100.0,
    ):
        super().__init__()
        self.offset_scale = float(offset_scale)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.cls_head = nn.Conv2d(hidden_dim, num_classes * num_anchor, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(hidden_dim, 2 * num_anchor, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        logits = self.cls_head(x).permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 1)
        offsets = self.reg_head(x).permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 2)
        return logits, offsets * self.offset_scale


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_level: int = 2, row: int = 1, line: int = 1):
        super().__init__()
        self.stride = 2**pyramid_level
        self.row = row
        self.line = line
        self._cache = {}
        self.register_buffer("cell_offsets", self._build_cell_offsets(), persistent=False)

    def _build_cell_offsets(self):
        row_step = self.stride / self.row
        line_step = self.stride / self.line
        shift_x = (torch.arange(self.line, dtype=torch.float32) + 0.5) * line_step - self.stride / 2
        shift_y = (torch.arange(self.row, dtype=torch.float32) + 0.5) * row_step - self.stride / 2
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    def forward(self, images: torch.Tensor):
        b, _, h, w = images.shape
        feat_h = (h + self.stride - 1) // self.stride
        feat_w = (w + self.stride - 1) // self.stride
        key = (int(feat_h), int(feat_w), images.device)
        if key not in self._cache:
            shift_x = (torch.arange(feat_w, device=images.device, dtype=torch.float32) + 0.5) * self.stride
            shift_y = (torch.arange(feat_h, device=images.device, dtype=torch.float32) + 0.5) * self.stride
            yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
            centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
            self._cache[key] = (centers[:, None, :] + self.cell_offsets.to(images.device)[None, :, :]).reshape(-1, 2)
        return self._cache[key].unsqueeze(0).expand(b, -1, -1)


class Decoder(nn.Module):
    def __init__(self, c4: int = 512, c5: int = 512, out_channels: int = 256):
        super().__init__()
        self.p5_1 = nn.Conv2d(c5, out_channels, kernel_size=1)
        self.p4_1 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, feats):
        c4, c5 = feats[2], feats[3]
        p5 = self.p5_1(c5)
        return self.p4_2(self.p4_1(c4) + self.up(p5))


class P2PNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        row: int,
        line: int,
        num_classes: int = 1,
        head_hidden_dim: int = 256,
        decoder_c4_channels: int = 512,
        decoder_c5_channels: int = 512,
        decoder_out_channels: int = 256,
        anchor_pyramid_level: int = 3,
        offset_scale: float = 100.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_anchor = row * line
        self.decoder = Decoder(c4=decoder_c4_channels, c5=decoder_c5_channels, out_channels=decoder_out_channels)
        self.head = ClassificationAndRegressionHead(
            decoder_out_channels,
            num_anchor=self.num_anchor,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            offset_scale=offset_scale,
        )
        self.anchor_points = AnchorPoints(pyramid_level=anchor_pyramid_level, row=row, line=line)

    def forward(self, samples: torch.Tensor):
        feats = self.backbone(samples)
        x = self.decoder(feats)
        pred_logits, pred_offsets = self.head(x)
        anchor_points = self.anchor_points(samples)
        pred_points = pred_offsets + anchor_points
        return {"pred_logits": pred_logits, "pred_points": pred_points, "anchor_points": anchor_points}
