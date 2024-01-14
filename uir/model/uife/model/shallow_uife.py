from model.uife.model.shallow import UWnet
from model.uife.model.uife import UIFE
from torch import nn


class ShallowUIFE(nn.Module):
    def __init__(self, cls_cfg, cls_pth):
        super(ShallowUIFE, self).__init__()

        self.uife = UIFE(cls_config=cls_cfg, cls_pth=cls_pth)
        self.restormer = UWnet()

    def forward(self, x):
        enhanced_feat = self.uife(x)
        out = self.restormer(enhanced_feat)
        return out