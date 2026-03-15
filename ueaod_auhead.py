import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, ConvTranspose2d
from mmdet.utils import ConfigType
from mmengine.dist import get_dist_info
from torchvision.transforms.functional import rgb_to_grayscale

from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS


class GuidedFilter(nn.Module):
    """可微分的引导滤波层"""

    def __init__(self, radius=15, eps=1e-6):
        super(GuidedFilter, self).__init__()
        self.radius = radius
        self.eps = eps
        self.box_filter = nn.AvgPool2d(kernel_size=2 * radius + 1, stride=1, padding=radius)

    def forward(self, guide, input):
        if guide.shape[1] == 3:
            guide = rgb_to_grayscale(guide)

        mean_I = self.box_filter(guide)
        mean_p = self.box_filter(input)
        corr_I = self.box_filter(guide * guide)
        corr_Ip = self.box_filter(guide * input)

        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = self.box_filter(a)
        mean_b = self.box_filter(b)

        return mean_a * guide + mean_b


class PhysicalGuidedEstimator(nn.Module):
    """
    物理引导的背景光和透射图估计模块 (修正版)
    输入:
        I - 原始水下图像 [B, 3, H, W]
        bg_feat - FFT背景特征图 [B, C, H//8, W//8]
        fg_feat - FFT前景特征图 [B, C, H//8, W//8]
    输出:
        B - 背景光图像 [B, 3, H, W]
        t - 透射图 [B, 1, H, W]
    """

    def __init__(self,
                 in_channels,
                 alpha: float = 0.6,
                 beta: float = 0.7,
                 omega: float = 0.9,
                 sigma: float=5.0,
                 norm_cfg: ConfigType = dict(type='BN', eps=1e-3, momentum=0.01),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),):
        super().__init__()

        self.alpha = alpha 
        self.omega = omega 
        self.beta = beta


        self.bg_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvModule(in_channels, in_channels//2, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvModule(in_channels//2, 3, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(3, 3, kernel_size=3, padding=1, norm_cfg=None, act_cfg=None)
        )

        kernel_size = min(31, int(6*sigma + 1))  # 经验公式
        self.physical_blur = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=kernel_size // 2, groups=3, bias=False)
        self._init_gaussian_kernel(kernel_size, sigma=sigma)

        self.fg_feature_extractor = nn.Sequential(
            ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            ConvTranspose2d(in_channels // 2, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True)
        )

        self.physical_prior = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True)
        )

        self.t_fusion = nn.Sequential(
            ConvModule(32, 16, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        self.guided_filter = GuidedFilter(radius=15, eps=1e-6)

        self.t_range = lambda x: 0.05 + 0.9 * torch.sigmoid(x)

    def _init_gaussian_kernel(self, kernel_size, sigma=5.0):

        x = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()

        kernel = g.unsqueeze(1) * g.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,15,15]

        self.physical_blur.weight.data = kernel.repeat(3, 1, 1, 1)
        self.physical_blur.weight.requires_grad = False  # 固定参数

    def estimate_t_physical(self, I, B):
        """基于物理先验估计透射图"""
        I_red = I[:, 0:1, :, :]
        I_green = I[:, 1:2, :, :]
        diff_rg = (I_green - I_red) / (I_green + I_red + 1e-6)

        diff_bg = torch.sum(torch.abs(I - B), dim=1, keepdim=True)
        diff_bg_norm = (diff_bg - diff_bg.min()) / (diff_bg.max() - diff_bg.min() + 1e-6)

        feature_map = self.alpha * diff_rg + self.omega * (1 - diff_bg_norm)

        t_initial = 1 - feature_map

        return t_initial

    def forward(self, I, fg_feat, bg_feat):
        # ========== 背景光估计 ==========
        B_initial = self.bg_upsample(bg_feat)

        B = self.physical_blur(B_initial)

        # ========== 透射图估计 ==========
        t_physical = self.estimate_t_physical(I, B)

        fg_feat_upscaled = self.fg_feature_extractor(fg_feat)

        physical_feat = self.physical_prior(I)

        fused_feat = torch.cat([fg_feat_upscaled, physical_feat], dim=1)
        t_feature = self.t_fusion(fused_feat)

        t_combined = self.beta * t_physical + (1 - self.beta) * t_feature

        I_gray = rgb_to_grayscale(I)
        t_filtered = self.guided_filter(I_gray, t_combined)

        t = self.t_range(t_filtered)

        return B, t


@MODELS.register_module()
class PhysicsGuidedHead(nn.Module):
    """
    物理引导的辅助损失头
    输入:
        I - 原始水下图像 [B, 3, H, W]
        fg_feat - 前景特征图 [B, C, H//8, W//8]
        bg_feat - 背景特征图 [B, C, H//8, W//8]
        j_feat - 干净图像特征图 [B, C, H//8, W//8]
    输出:
        loss - 物理模型重建损失
    """
    def __init__(self,
                 in_channels,
                 widen_factor: float = 1.0,
                 alpha: float = 0.6,
                 beta: float = 0.7,
                 omega: float = 0.9,
                 sigma: float = 5.0,
                 delta: float = 0.1,
                 recon_weight: float = 4.0,
                 smooth_weight: float = 1.0,
                 consist_weight: float = 3.0,
                 norm_cfg: ConfigType=dict(type='BN', eps=1e-3, momentum=0.01),
                 act_cfg: ConfigType=dict(type='SiLU', inplace=True)
                 ):
        super().__init__()
        self.recon_weight = recon_weight
        self.smooth_weight = smooth_weight
        self.consist_weight = consist_weight
        self.delta = delta

        in_channels = make_divisible(in_channels, widen_factor)

        self.j_reconstructor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvModule(in_channels, in_channels//2, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvModule(in_channels//2, 3, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(3, 3, kernel_size=3, padding=1, norm_cfg=None, act_cfg=None)
        )

        self.estimator = PhysicalGuidedEstimator(in_channels, alpha, beta, omega, sigma, norm_cfg, act_cfg)

    def loss(self, I, bg_feat, fg_feat, j_feat):
        I = F.interpolate(
                I,
                size=(I.shape[-2] // 2, I.shape[-1] // 2), 
                mode='bilinear',
                align_corners=False, 
                recompute_scale_factor=False
            )

        J_rec = self.j_reconstructor(j_feat)

        B_est, t_est = self.estimator(I, fg_feat, bg_feat)

        I_recon = J_rec * t_est + B_est * (1 - t_est)

        recon_loss = F.l1_loss(I_recon, I)

        bg_blur = F.avg_pool2d(B_est, kernel_size=15, stride=1, padding=7)
        bg_smooth_loss = F.mse_loss(B_est, bg_blur)

        I_red = I[:, 0:1, :, :]
        I_green = I[:, 1:2, :, :]
        physical_rg_diff = (I_green - I_red) / (I_green + I_red + 1e-6)

        predicted_rg_diff = t_est

        rg_corr_loss = F.huber_loss(
            physical_rg_diff,
            predicted_rg_diff,
            reduction='mean',
            delta=self.delta
        )

        _, world_size = get_dist_info()
        return dict(
            loss_recon=self.recon_weight * recon_loss * I.shape[0] * world_size,
            loss_smooth=self.smooth_weight * bg_smooth_loss * I.shape[0] * world_size,
            loss_consist=self.consist_weight * rg_corr_loss * I.shape[0] * world_size,
        )
    
    def forward(self, I, fg_feat, bg_feat, j_feat):
        I = F.interpolate(
                I,
                size=(I.shape[-2] // 2, I.shape[-1] // 2),
                mode='bilinear', 
                align_corners=False, 
                recompute_scale_factor=False
            )

        J_rec = self.j_reconstructor(j_feat)

        B_est, t_est = self.estimator(I, fg_feat, bg_feat)

        I_recon = J_rec * t_est + B_est * (1 - t_est)

        return I_recon, J_rec, B_est, t_est