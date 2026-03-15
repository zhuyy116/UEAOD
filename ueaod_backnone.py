from typing import Union, List, Tuple


from mmengine.model import BaseModule
from ..layers.cduod import FFDM
from ..layers import DarknetBottleneck
import torch
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import nn

from mmyolo.models.backbones.csp_darknet import BaseBackbone, CSPLayerWithTwoConv
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.registry import MODELS




class C3k2(CSPLayerWithTwoConv):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int = 1,
            c3k: bool = False,
            expand_ratio: float = 0.5,
            groups: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            expand_ratio=expand_ratio,
            groups=groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.blocks = nn.ModuleList(
            C3k(self.mid_channels, self.mid_channels, 2, add_identity, groups) if c3k else DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                groups=groups,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        )


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 ):
        """
        Initializes an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = ConvModule(dim, all_head_dim * 3, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.proj = ConvModule(all_head_dim, dim, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.pe = ConvModule(all_head_dim, dim, 7, 1, 3, groups=dim, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention."""
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 ):
        """
        Initializes an Area-attention block module for efficient feature extraction in YOLO models.

        This module implements an area-attention mechanism combined with a feed-forward network for processing feature
        maps. It uses a novel area-based attention approach that is more efficient than traditional self-attention
        while maintaining effectiveness.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(ConvModule(dim, mlp_hidden_dim, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                                 ConvModule(mlp_hidden_dim, dim, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(BaseModule):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 a2: bool = True,
                 area: int=1,
                 residual: bool = False,
                 mlp_ratio: float = 2.0,
                 expand_ratio: float = 0.5,
                 groups: int = 1,
                 shortcut: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        """
        Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__(init_cfg=init_cfg)
        c_ = int(out_channels * expand_ratio)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = ConvModule(in_channels, c_,
                              kernel_size=1,
                              stride=1,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=act_cfg
                              )
        self.cv2 = ConvModule((1 + num_blocks) * c_, out_channels,
                              kernel_size=1,
                              conv_cfg = conv_cfg,
                              norm_cfg = norm_cfg,
                              act_cfg = act_cfg
                              )

        self.gamma = nn.Parameter(0.01 * torch.ones(out_channels), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, groups)
            for _ in range(num_blocks)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self,
                 c1, c2, n=1, shortcut=True, g=1, e=0.5,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvModule(c1, c_, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(c1, c_, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv3 = ConvModule(2 * c_, c2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(DarknetBottleneck(c_, c_, expansion=1.0,  kernel_size=(1,3), add_identity=shortcut) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        p = k // 2
        self.m = nn.Sequential(*(DarknetBottleneck(c_, c_, expansion=1.0, kernel_size=(k, k), padding=(p, p), add_identity=shortcut) for _ in range(n)))


@MODELS.register_module()
class YOLO12CSPDarknet(BaseBackbone):
    """CSP-Darknet backbone used in YOLO11.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5}.
            Defaults to P5.
        last_stage_out_channels (int): Final layer output channel.
            Defaults to 1024.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to: 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (Union[dict,list[dict]], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmyolo.models import YOLOv8CSPDarknet
        >>> import torch
        >>> model = YOLOv8CSPDarknet()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.
    arch_settings = {
        'P5': [[64, 256, 128, 2, False, 0.25], [256, 512, 256, 2, False, 0.25],
               [512, 512, 512, 4, True, 4], [512, 1024, 1024, 4, True, 1]],
    }
 

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 use_c3k: bool = False,
                 residual: bool = False,
                 mlp_ratio: float = 2.0,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        self.arch_settings[arch][-1][2] = last_stage_out_channels
        self.use_c3k = use_c3k
        self.residual = residual
        self.mlp_ratio = mlp_ratio
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, mid_channels, num_blocks, att, arg = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        mid_channels = make_divisible(mid_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer =  A2C2f(mid_channels, out_channels,
                           num_blocks=num_blocks,
                           a2=True,
                           area=arg,
                           residual=self.residual,
                           mlp_ratio=self.mlp_ratio,
                           norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg)  \
            if att else C3k2(mid_channels, out_channels,
                             num_blocks=num_blocks,
                             c3k=self.use_c3k,
                             expand_ratio=arg,
                             norm_cfg=self.norm_cfg,
                             act_cfg=self.act_cfg)
        stage.append(csp_layer)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.reset_parameters()
        else:
            super().init_weights()


class FFDM(nn.Module):
    '''
    Fourier Frequency Decoupling Module
    训练时：使用 STE 使得硬掩码 (hard_mask) 前向，软掩码 (soft_mask) 后向；
    测试时：直接用硬掩码 (hard_mask) 截断频谱。
    '''
    def __init__(self, channels, alpha=30,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 act_cfg=dict(type='SiLU', inplace=True)):
        super().__init__()

        self.alpha = alpha

        self.short_conv = nn.Sequential(
            ConvModule(channels, channels, kernel_size=(1, 7), stride=1,
                       padding=(0, 3), groups=channels, norm_cfg=norm_cfg, act_cfg=None),
            ConvModule(channels, channels, kernel_size=(7, 1), stride=1,
                       padding=(3, 0), groups=channels, norm_cfg=norm_cfg, act_cfg=None)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x, temp=None):
        """
        x: 形状 [B, C, H, W] 的输入特征
        返回：
            high_part: 高频部分 Tensor, shape [B, C, H, W]
            low_part:  低频部分 Tensor, shape [B, C, H, W]
        """
        b, c, h, w = x.shape

        x_fft = torch.fft.rfft2(x, norm='ortho')

        ratio = self.act(self.global_pool(self.short_conv(x)))  # requires_grad=True

        freq_h = torch.fft.fftfreq(h, device=ratio.device).unsqueeze(1)   # (h, 1)
        freq_w = torch.fft.rfftfreq(w, device=ratio.device).unsqueeze(0)  # (1, w_rfft)
        dist_mat = torch.sqrt(freq_h**2 + freq_w**2)                      # (h, w_rfft)
        max_dist = dist_mat.max()                                         # scalar

        threshold = ratio * max_dist

        dist_expand = dist_mat.unsqueeze(0).unsqueeze(0)  # (1,1,h,w_rfft)

        hard_mask = (dist_expand < threshold).float()  # shape (B, C, h, w_rfft)

        if self.training:
            temp = self.alpha if temp is None else temp
            soft_mask = torch.sigmoid(temp * (threshold - dist_expand))  # (B, C, h, w_rfft)，requires_grad=True
            mask = (hard_mask - soft_mask).detach() + soft_mask 
        else:
            mask = hard_mask

        x_fft_masked = x_fft * mask  # 这样等价于同时对 real & imag 乘以 hard_mask

        low_part = torch.fft.irfft2(x_fft_masked, s=(h, w), norm='ortho')
        high_part = x - low_part

        return low_part, high_part
            

@MODELS.register_module()
class CDUOD_Backbone(YOLO12CSPDarknet):
    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 use_c3k: bool = False,
                 residual: bool = False,
                 mlp_ratio: float = 2.0,
                 alpha: float = 30,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            arch=arch,
            last_stage_out_channels=last_stage_out_channels,
            plugins=plugins,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_c3k=use_c3k,
            residual=residual,
            mlp_ratio=mlp_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg
        )
        self.alpha = alpha

        self.cover = nn.ModuleDict()
        for idx in out_indices:
            channels = make_divisible(self.arch_settings[arch][idx -1][1], self.widen_factor)
            self.cover[str(idx)] = FFDM(channels=channels,
                                        alpha=self.alpha,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs_f = []
        outs_b = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                x, x_b = self.cover[str(i)](x, temp=self.alpha)
                outs_f.append(x)
                outs_b.append(x_b)

        return tuple(outs_f), tuple(outs_b)
