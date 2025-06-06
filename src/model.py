import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetFPN(nn.Module):
    """
    output of conv: {c2, c3, c4, c5}
    strides of conv: {4, 8, 16, 32}
    fpn_dim: 64
    """
    def __init__(self):
        super().__init__()
            

        # Backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone_stages = nn.ModuleList([
            self.conv1 = nn.Sequential(*list(resnet.children())[:4])
            self.conv2 = resnet.layer1  # [B, 64, 160, 160]
            self.conv3 = resnet.layer2  # [B, 128, 80, 80]
            self.conv4 = resnet.layer3  # [B, 256, 40, 40]
            self.conv5 = resnet.layer4  # [B, 512, 20, 20]
        ])

        for param in self.backbone_stages.parameters():
            param.requires_grad = False

        # FPN Layers
        in_channels = [64, 128, 256, 512]
        fpn_dim = 64

        self.lat_convs = nn.ModuleList([nn.Conv2d(c, fpn_dim, kernel_size=1) for c in in_channels])
        self.smooth_convs = nn.ModuleList([nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1) for _ in in_channels])

        self.box_head = nn.Conv2d(fpn_dim, 4, kernel_size=1)
        self.conf_head = nn.Conv2d(fpn_dim, 1, kernel_size=1)

        self._init_bias()

    def _init_bias(self):
        pi = 0.01
        initial_bias = -torch.log(torch.tensor((1.0 - pi) / pi))
        nn.init.constant_(self.conf_head.bias, initial_bias.to(self.conf_head.bias.device))

    def forward(self, x):
        B = x.size(0)
        x = self.backbone_stages[0](x)
        c2 = self.backbone_stages[1](x)      # [B, 64, 160, 160]
        c3 = self.backbone_stages[2](c2)     # [B, 128, 80, 80]
        c4 = self.backbone_stages[3](c3)     # [B, 256, 40, 40]
        c5 = self.backbone_stages[4](c4)     # [B, 512, 20, 20]

        p5 = self.lat_convs[3](c5)
        p4 = self.lat_convs[2](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lat_convs[1](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lat_convs[0](c2) + F.interpolate(p3, scale_factor=2, mode='nearest')

        p5_out = self.smooth_convs[3](p5)
        p4_out = self.smooth_convs[2](p4)
        p3_out = self.smooth_convs[1](p3)
        p2_out = self.smooth_convs[0](p2)
        
        raw_box_20 = self.box_head(p5_out).permute(0, 2, 3, 1).reshape(B, -1, 4)
        raw_box_40 = self.box_head(p4_out).permute(0, 2, 3, 1).reshape(B, -1, 4)
        raw_box_80 = self.box_head(p3_out).permute(0, 2, 3, 1).reshape(B, -1, 4)
        raw_box_160 = self.box_head(p2_out).permute(0, 2, 3, 1).reshape(B, -1, 4)

        raw_box_pred_batch = torch.cat([raw_box_20, raw_box_40, raw_box_80, raw_box_160], dim=1)
        normalized_box_pred_batch = torch.sigmoid(raw_box_pred_batch)

        conf_20 = self.conf_head(p5_out).permute(0, 2, 3, 1).reshape(B, -1, 1)
        conf_40 = self.conf_head(p4_out).permute(0, 2, 3, 1).reshape(B, -1, 1)
        conf_80 = self.conf_head(p3_out).permute(0, 2, 3, 1).reshape(B, -1, 1)
        conf_160 = self.conf_head(p2_out).permute(0, 2, 3, 1).reshape(B, -1, 1)

        conf_logits_batch = torch.cat([conf_20, conf_40, conf_80, conf_160], dim=1)
        pred = torch.cat([normalized_box_pred_batch, conf_logits_batch], dim=-1)  # [B, N, 5]

        return pred

