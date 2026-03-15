import torch
from mmcv.cnn import ConvModule
from mmcv.ops import CARAFEPack
from mmdet.utils import ConfigType, OptMultiConfig
from typing import List, Union
from torch import nn


from mmyolo.models.utils import make_divisible, make_round
from mmyolo.registry import MODELS
from mmyolo.models import YOLOv5PAFPN




@MODELS.register_module()
class YOLO12PAFPN(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 2,
                 residual: bool = False,
                 mlp_ratio: float = 2.0,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.residual = residual
        self.mlp_ratio = mlp_ratio
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return A2C2f(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor) if idx==2 else make_divisible((self.in_channels[idx - 1] + self.out_channels[idx]),
                           self.widen_factor) ,
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            a2=False,
            area=-1,
            residual=self.residual,
            mlp_ratio=self.mlp_ratio,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """

        if idx == 1:
            layer = C3k2(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            c3k=True,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        else:
            layer = A2C2f(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            a2=False,
            area=-1,
            residual=self.residual,
            mlp_ratio=self.mlp_ratio,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        return layer

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            make_divisible(self.out_channels[idx], self.widen_factor),
            make_divisible(self.out_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)



@MODELS.register_module()
class CDSN(YOLO12PAFPN):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 2,
                 residual: bool = False,
                 mlp_ratio: float = 2.0,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            residual=residual,
            mlp_ratio=mlp_ratio,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg
        )

        self.down_b = None

        self.conconv = ConvModule(
            make_divisible(sum(self.in_channels), self.widen_factor),
            make_divisible(self.out_channels[0], self.widen_factor),
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.up_b4 = CARAFEPack(
            channels=make_divisible(self.out_channels[1], self.widen_factor),
            scale_factor=2)
        self.up_b5 = CARAFEPack(
            channels=make_divisible(self.out_channels[2], self.widen_factor),
            scale_factor=4)

        self.conv_p3 = A2C2f(
            make_divisible(self.out_channels[0], self.widen_factor),
            make_divisible(self.out_channels[0], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            a2=False,
            area=-1,
            residual=self.residual,
            mlp_ratio=self.mlp_ratio,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        x_f, x_b = inputs[0], inputs[1]
        assert len(x_f) == len(self.in_channels) or len(x_b) == len(self.in_channels)

        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](x_f[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        x_b4_up = self.up_b4(x_b[1])
        x_b5_up = self.up_b5(x_b[2])
        x_b_down = self.conconv(torch.cat([x_b[0], x_b4_up, x_b5_up], dim=1))

        # bottom-up path
        outs = [self.conv_p3(inner_outs[0] + x_b_down)]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            # x_b_down = self.down_b[idx](x_b_down)
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        if self.training:
            return tuple(results), inner_outs[0], x_b_down, outs[0]
        else:
            return tuple(results)