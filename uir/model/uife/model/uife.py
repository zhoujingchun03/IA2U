import torch
from torch import nn
from mmpretrain.registry import MODELS
from model.uife.model.components import UpScalex2, DownScalex2


class DegradedPriorGen(nn.Module):
    def __init__(self, width=24, norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU):
        super().__init__()

        self.feat1 = nn.Conv2d(in_channels=64, out_channels=width, kernel_size=1, bias=False)
        self.feat2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=width, kernel_size=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.feat3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=width, kernel_size=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.feat4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=width, kernel_size=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )
        self.up = nn.Sequential(
            norm_layer(width),
            nn.Conv2d(in_channels=width, out_channels=16 * width, kernel_size=1, bias=False),
            nn.PixelShuffle(upscale_factor=4),
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, bias=False),
            act_layer(inplace=True)
        )

    def forward(self, res_feats):
        res_feats = list(res_feats)
        res_feats[0] = self.feat1(res_feats[0])
        res_feats[1] = self.feat2(res_feats[1])
        res_feats[2] = self.feat3(res_feats[2])
        res_feats[3] = self.feat4(res_feats[3])
        res_feats = res_feats[0] + res_feats[1] + res_feats[2] + res_feats[3]
        res_feats = self.up(res_feats)

        return res_feats


class ScaleAggregation(nn.Module):
    def __init__(self, width=24, up='BiLinear', down='AvgPool', norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU):
        super().__init__()

        self.norm = norm_layer(width)
        self.width = width

        self.conv1 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, groups=width, bias=False)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=2, groups=width,
                               padding=3 // 2, padding_mode='reflect', bias=False)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=5, stride=4, groups=width,
                               padding=5 // 2, padding_mode='reflect', bias=False)

        self.up = UpScalex2(type=up, in_c=width)
        self.down = DownScalex2(type=down, in_c=width)
        self.mlp1 = nn.Conv2d(in_channels=width * 3, out_channels=width, kernel_size=1, bias=False)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels=width * 3, out_channels=width * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, bias=False),
            act_layer(inplace=True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv2d(in_channels=width * 3, out_channels=width * 16, kernel_size=1, bias=False),
            nn.PixelShuffle(upscale_factor=4),
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, bias=False),
            act_layer(inplace=True)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        res = x
        s1 = self.conv1(x)
        s2 = self.conv2(x)
        s3 = self.conv3(x)
        s2to1 = self.up(s2)
        s3to1 = self.up(self.up(s3))
        s1to2 = self.down(s1)
        s3to2 = self.up(s3)
        s1to3 = self.down(self.down(s1))
        s2to3 = self.down(s2)
        s1 = torch.concat([s1, s2to1, s3to1], dim=1)
        s2 = torch.concat([s1to2, s2, s3to2], dim=1)
        s3 = torch.concat([s1to3, s2to3, s3], dim=1)
        s1 = self.mlp1(s1)
        s2 = self.mlp2(s2)
        s3 = self.mlp3(s3)
        s_all = torch.concat([s1.unsqueeze(1), s2.unsqueeze(1), s3.unsqueeze(1)], dim=1)
        B, C, H, W = x.shape
        weight = torch.concat([
            self.avg(s1).sum(dim=1),
            self.avg(s2).sum(dim=1),
            self.avg(s3).sum(dim=1)
        ], dim=1)
        weight = torch.softmax(weight, dim=1)
        weight = weight.expand(B, 3, C * H * W)[..., None].view(B, 3, C, -1)[..., None].view(B, 3, C, H, -1)

        out = (s_all * weight).sum(dim=1) + res
        return out


class Attention(nn.Module):
    def __init__(self, n_heads=6, width=24, norm_layer=nn.InstanceNorm2d):
        super().__init__()

        self.n_heads = n_heads

        self.norm_layer = norm_layer(width)
        self.k = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, groups=width, bias=False)
        self.v = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, groups=width, bias=False)
        self.q = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, groups=width, bias=False)
        self.sig = nn.Sigmoid()
        self.out = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, groups=width, bias=False)

    def forward(self, x, degraded_prior, cls_prior):
        B, C, H, W = x.shape
        cls_prior = torch.argmax(cls_prior, dim=-1, keepdim=True)
        cls_prior = cls_prior.expand(-1, C * H * W).reshape(B, C, H, W)
        res = x
        q = x + degraded_prior + cls_prior
        q = self.norm_layer(q)
        x = self.norm_layer(x)
        q = self.q(q).reshape(B, self.n_heads, C // self.n_heads, H, W)
        k = self.k(x).reshape(B, self.n_heads, C // self.n_heads, H, W)
        v = self.v(x).reshape(B, self.n_heads, C // self.n_heads, H, W)
        out = self.out((self.sig(q * k) * v).reshape(B, C, H, W)) + res
        return out


class BasicBlock(nn.Module):
    def __init__(self, width=24, n_heads=6, norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU):
        super().__init__()

        self.sa1 = ScaleAggregation(width=width, norm_layer=norm_layer, act_layer=act_layer)
        self.sa2 = ScaleAggregation(width=width, norm_layer=norm_layer, act_layer=act_layer)
        self.attn = Attention(n_heads=n_heads, width=width, norm_layer=norm_layer)

    def forward(self, x, degraded_prior, cls_prior):
        x = self.sa1(x)
        x = self.sa2(x)
        out = self.attn(x, degraded_prior, cls_prior)
        return out


class UIFE(nn.Module):
    def __init__(self, depth=2, width=24, n_heads=6, norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU, cls_config=None, cls_pth=r'./'):
        super().__init__()

        self.depth = depth
        self.classifier = MODELS.build(cls_config)
        self.classifier.load_state_dict(torch.load(cls_pth)['state_dict'])
        for name, parameter in self.classifier.named_parameters():
            parameter.requires_grad = False

        self.degraded_prior_gen = DegradedPriorGen(width=width, norm_layer=norm_layer, act_layer=act_layer)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=width, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, bias=False),
            act_layer(),
        )

        for i in range(depth):
            setattr(self, f'block{i}', BasicBlock(width=width, n_heads=n_heads, norm_layer=norm_layer, act_layer=act_layer))

        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(in_channels=width, out_channels=3, kernel_size=1, bias=False),
        )

    def forward(self, x):
        res = x
        cls = self.classifier(x)
        mid_feats = self.classifier.backbone(x)
        degraded_prior = self.degraded_prior_gen(mid_feats)
        x = self.in_proj(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x, degraded_prior, cls)
        x = self.out_proj(x)
        out = x + res
        return out
