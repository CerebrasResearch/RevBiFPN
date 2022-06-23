"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.

RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network
"""
import torch
from torch import nn
from typing import Union, List, Any, Tuple

# add file dir to path
# alternatively pip install revstructs and `from revstructs import *`
import os, sys
sys.path.append(os.path.dirname(__file__))
from rev_structs.revstructs import (
    RevSequential,
    RevSpatialDownsample,
    RevResidualSilo,
    RevResidualBlock,
    RevSilo,
    RecomputeSilo,
)


__all__ = [
    "RevBiFPN",
    "RevBiFPN_S",
    "revbifpn_s0",
    "revbifpn_s1",
    "revbifpn_s2",
    "revbifpn_s3",
    "revbifpn_s4",
    "revbifpn_s5",
    "revbifpn_s6",
    "model_fns",
]


norm_kwargs_defaults = {
    "eps": 1e-3,
    "momentum": 0.1,
    "affine": True,
    "track_running_stats": True,
}


def stochastic_depth_fn(
    x,
    p: float,
    mode: str = "row",
    training: bool = True
):
    assert 0.0 < p < 1.0, f"drop probability has to be between 0 and 1, but got {p}"
    assert mode in ["batch", "row"], f"mode has to be either 'batch' or 'row', but got {mode}"

    if not training or p == 0.0:
        # inference mode
        return x

    survival_rate = 1.0 - p

    size = [1] * x.ndim # drop entire batch
    if mode == "row":
        # drop per sample
        size = [x.shape[0]] + [1] * (x.ndim - 1)

    mask = torch.empty(size, dtype=x.dtype, device=x.device)
    mask = mask.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        mask.div_(survival_rate)
    return x * mask


class GlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        n, c = x.shape[:2]
        if self.flatten:
            return x.view(n, c, -1).mean(dim=2)
        return x.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ch, act_fn=nn.Hardswish, se_sig_act=nn.Hardsigmoid):
        super().__init__()
        self.se = nn.Sequential(
            GlobalAvgPool2d(),
            nn.Conv2d(in_channels=in_channels, out_channels=se_ch, kernel_size=1),
            act_fn(),
            nn.Conv2d(in_channels=se_ch, out_channels=in_channels, kernel_size=1),
            se_sig_act(),
        )

    def forward(self, x):
        return x * self.se(x)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        conv_bias: bool = None,
        norm: bool = True,
        norm_fn: nn.Module = nn.BatchNorm2d,
        norm_kwargs: dict = norm_kwargs_defaults,
        zero_init: bool = False,
        act: bool = True,
        act_fn: nn.Module = nn.Hardswish,
    ):
        super().__init__()
        self.zero_init = zero_init

        if padding is None:
            padding = (kernel_size - 1) // 2
        if conv_bias is None:
            conv_bias = not norm

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=conv_bias,
        )

        self.norm = None
        if norm:
            self.norm = norm_fn(out_channels, **norm_kwargs)
            if zero_init:
                nn.init.zeros_(self.norm.weight)

        self.act = act_fn() if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block.

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expand_ratio: int = 2,
        stride: int = 1,
        se_ratio: float = 0.125,
        id_skip: bool = False,
        norm: bool = True,
        norm_fn: nn.Module = nn.BatchNorm2d,
        norm_kwargs: dict = norm_kwargs_defaults,
        zero_init: bool = False,
        stochastic_depth: float = 0.0,
        act_fn: nn.Module = nn.Hardswish,
        se_sig_act: nn.Module = nn.Hardsigmoid,
    ):
        super().__init__()

        # skip connection
        if id_skip:
            assert stride == 1
            assert in_channels == out_channels
        self.id_skip = id_skip

        int_ch = int(in_channels * expand_ratio)

        self.exp_conv = None
        if expand_ratio != 1:
            self.exp_conv = ConvNormAct(
                in_channels=in_channels,
                out_channels=int_ch,
                kernel_size=1,
                norm=norm,
                norm_fn=norm_fn,
                norm_kwargs=norm_kwargs,
                act_fn=act_fn,
            )

        # Depthwise convolution
        self.dw_conv = ConvNormAct(
            in_channels=int_ch,
            out_channels=int_ch,
            kernel_size=kernel_size,
            stride=stride,
            groups=int_ch,
            norm=norm,
            norm_fn=norm_fn,
            norm_kwargs=norm_kwargs,
            act_fn=act_fn,
        )

        # Squeeze and Excitation layer, if desired
        self.se = None
        if 0 < se_ratio <= 1:
            self.se = SqueezeExcite(
                int_ch, max(1, int(in_channels * se_ratio)),
                act_fn=act_fn, se_sig_act=se_sig_act
            )

        # Pointwise convolution
        self.project_conv = ConvNormAct(
            in_channels=int_ch,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm,
            norm_fn=norm_fn,
            norm_kwargs=norm_kwargs,
            zero_init=zero_init,
            act=False,
        )

        self.stochastic_depth = stochastic_depth

    def forward(self, inputs):
        """
        MBConvBlock's fwd function.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs

        if self.exp_conv is not None:
            x = self.exp_conv(x)
        x = self.dw_conv(x)

        # Squeeze and Excitation
        if self.se is not None:
            x = self.se(x)

        # Pointwise Convolution
        x = self.project_conv(x)

        if self.stochastic_depth:
            x = stochastic_depth_fn(
                x, self.stochastic_depth, training=self.training)

        # Skip connection and drop connect
        if self.id_skip:
            # Skip connection
            return x + inputs
        return x


class FPN_Neck(nn.Module):
    """
    Neck for feature pyramid network
    """
    def __init__(
        self,
        in_channels: List[int] = [48, 64, 80, 160],
        out_channels: List[int] = [48, 64, 128, 320],
        dwk: List[int] = [3, 3, 3, 3],
        dwkp: List[int] = [2, 2, 2, 2],
        expand_ratios: List[int] = [1, 2, 2, 2],
        se_ratios: List[float] = [0.25, 0.25, 0.125, 0.125],
        norm: bool = True,
        norm_fn: nn.Module = nn.BatchNorm2d,
        norm_kwargs: dict = norm_kwargs_defaults,
        act_fn: nn.Module = nn.Hardswish,
        se_sig_act: nn.Module = nn.Hardsigmoid,
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels)

        self.xforms = nn.ModuleList()
        for params in zip(in_channels, out_channels, se_ratios, expand_ratios, dwk, dwkp):
            in_ch, out_ch, ser, expr, _dwk, _dwkp = params
            self.xforms.append(MBConvBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=_dwk + _dwkp,
                expand_ratio=expr,
                stride=1,
                se_ratio=ser,
                id_skip=False,
                norm=norm,
                norm_fn=norm_fn,
                norm_kwargs=norm_kwargs,
                stochastic_depth=0.0,
                act_fn=act_fn,
                se_sig_act=se_sig_act,
            ))

    def forward(self, x):
        # run xforms
        return [xform(x_i) for x_i, xform in zip(x, self.xforms)]


class ClassHead(nn.Module):
    """
    Implements a classification head to be attached to an FPN
    """
    def __init__(
        self,
        channels: List[int] = [48, 64, 128, 320],
        dwk: List[int] = [3, 3, 3, 3],
        dwkp: List[int] = [2, 2, 2, 2],
        expand_ratios: List[int] = [1, 2, 2, 2],
        se_ratios: List[float] = [0.25, 0.25, 0.125, 0.125],
        fc_features: int = 1280,
        dropout: float = 0.2,
        classes: int = 1000,
        norm: bool = True,
        norm_fn: nn.Module = nn.BatchNorm2d,
        norm_kwargs: dict = norm_kwargs_defaults,
        act_fn: nn.Module = nn.Hardswish,
        se_sig_act: nn.Module = nn.Hardsigmoid,
    ):
        super().__init__()
        self.pyramid_heads = nn.ModuleList()
        for height, ch in enumerate(channels[:-1]):
            self.pyramid_heads.append(MBConvBlock(
                in_channels=ch,
                out_channels=channels[height + 1],
                kernel_size=3 + dwkp[height + 1],
                expand_ratio=expand_ratios[height],
                stride=2,
                se_ratio=se_ratios[height],
                id_skip=False,
                norm=norm,
                norm_fn=norm_fn,
                norm_kwargs=norm_kwargs,
                stochastic_depth=0.0,
                act_fn=act_fn,
                se_sig_act=se_sig_act,
            ))

        self.final_conv = None
        if fc_features:
            self.final_conv = ConvNormAct(
                in_channels=channels[-1],
                out_channels=fc_features,
                kernel_size=1,
                stride=1,
                norm=norm,
                norm_fn=norm_fn,
                norm_kwargs=norm_kwargs,
                act_fn=act_fn,
            )
        else:
            fc_features = channels[-1]

        self.avgpool = GlobalAvgPool2d(flatten=True)

        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fc = nn.Linear(fc_features, classes)

    def forward(self, x):
        # combine all resolution paths
        out = x[0]
        for idx, h in enumerate(self.pyramid_heads):
            out = x[idx + 1] + h(out)

        if self.final_conv is not None:
            out = self.final_conv(out)

        out = self.avgpool(out)

        if self.dropout is not None:
            out = self.dropout(out)

        return self.fc(out)


def s_based_k(stride, p=0):
    kernel_size = 2 * stride - 1 + p
    return kernel_size


def s_based_k_alt(stride, p=0):
    # alternative s_based_k which should probably have been used
    # but re-training networks takes too much time...
    kernel_size = stride + 1 + p
    return kernel_size


class RevBiFPN(nn.Module):
    """
    Implements RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network
    with classification head for training on ImageNet1k
    """
    def __init__(
        self,
        stem_downsampling: int = 2,
        channels: Union[int, List[int]] = [48, 64, 80, 160],
        dwk: List[int] = [3, 5, 5, 3],
        dwkp: List[int] = [0, 0, 0, 2],
        inv_blk_rep: int = 2,
        expand_ratios: List[int] = [1, 1, 2, 2],
        se_ratios: List[float] = [0.25, 0.25, 0.125, 0.0],
        num_ext_layers: int = 2,
        zero_init: bool = False,
        stochastic_depth: float = 0.0,
        upscale_mode: str = "bilinear",
        disable_rev: bool = False,
        disable_recomp: bool = False,
        head_channels: List[int] = [48, 64, 128, 320],
        head_dwk: List[int] = [3, 3, 3, 3],
        head_dwkp: List[int] = [2, 2, 2, 2],
        head_expand_ratios: List[int] = [1, 2, 2, 2],
        head_se_ratios: List[float] = [0.25, 0.25, 0.125, 0.125],
        fc_features: int = 1280,
        head_dropout: float = 0.2,
        classes: int = 1000,
        norm: bool = True,
        norm_fn: nn.Module = nn.BatchNorm2d,
        norm_kwargs: dict = norm_kwargs_defaults,
        act_fn: nn.Module = nn.Hardswish,
        se_sig_act: nn.Module = nn.Hardsigmoid,
    ):
        super().__init__()
        assert len(channels) == len(dwk) == len(dwkp) == len(expand_ratios) == len(se_ratios)

        self.zero_init = zero_init

        self.rev_stack = RevSequential(
            [], disable_rev=disable_rev, disable_recomp=disable_recomp)

        int_ch = channels[0]
        self.in_channels = int_ch
        if stem_downsampling:
            self.in_channels = int_ch // (4 ** stem_downsampling)
            self.rev_stack.append(
                RevSpatialDownsample(2 ** stem_downsampling, disable_rev=disable_rev)
            )
        assert self.in_channels >= 3

        blk_idx = 0
        total_num_blks = len(channels) - 1 + num_ext_layers

        # create RevBiFPN ie multi-resolution feature pyramid
        mk_layer_args = {}
        mk_layer_args["norm"] = norm
        mk_layer_args["norm_fn"] = norm_fn
        mk_layer_args["norm_kwargs"] = norm_kwargs
        mk_layer_args["act_fn"] = act_fn
        mk_layer_args["se_sig_act"] = se_sig_act
        mk_layer_args["zero_init"] = zero_init
        mk_layer_args["disable_rev"] = disable_rev
        mk_layer_args["expand_ratios"] = expand_ratios
        mk_layer_args["se_ratios"] = se_ratios
        mk_layer_args["dwkp"] = dwkp
        mk_layer_args["dwk"] = dwk
        mk_layer_args["inv_blk_rep"] = inv_blk_rep
        mk_layer_args["upscale_mode"] = upscale_mode
        for new_ch in channels[1:]:
            blk_idx += 1
            mk_layer_args["channels"] = channels[:blk_idx]
            mk_layer_args["new_ch"] = new_ch
            mk_layer_args["stochastic_depth"] = stochastic_depth * (blk_idx / total_num_blks)
            self.add_rev_make_layer(**mk_layer_args)

        # extend RevBiFPN ie multi-resolution feature pyramid
        mk_layer_args["channels"] = channels
        mk_layer_args["new_ch"] = None
        for idx in range(num_ext_layers):
            blk_idx += 1
            mk_layer_args["stochastic_depth"] = stochastic_depth * (blk_idx / total_num_blks)
            self.add_rev_make_layer(**mk_layer_args)

        # make non-rev neck
        # the neck uses rev-ckpt (aka grad-ckpt) which can be disbled using disable_recomp
        # set head_channels to None create the model without the neck.
        if head_channels and head_dwk and head_dwkp and head_expand_ratios and head_se_ratios:
            assert len(channels) == len(head_channels) == len(head_dwk) == \
                len(head_dwkp) == len(head_expand_ratios) == len(head_se_ratios)
            neck = FPN_Neck(
                in_channels=channels,
                out_channels=head_channels,
                expand_ratios=head_expand_ratios,
                se_ratios=head_se_ratios,
                dwk=head_dwk,
                dwkp=head_dwkp,
                norm=norm,
                norm_fn=norm_fn,
                norm_kwargs=norm_kwargs,
                act_fn=act_fn,
                se_sig_act=se_sig_act,
            )
            self.rev_stack.recomp_op = RecomputeSilo(
                recomp_transforms=list(neck.xforms),
                disable_recomp=disable_recomp,
            )

        # non-rev head
        # set classes to 0 to instantiate the model without the classification head.
        self.head = None
        if classes:
            if head_channels is None or len(channels) != len(head_channels):
                head_channels = channels
            if head_dwk is None or len(channels) != len(head_dwk):
                head_dwk = dwk
            if head_dwkp is None or len(channels) != len(head_dwkp):
                head_dwkp = dwkp
            if head_expand_ratios is None or len(channels) != len(head_expand_ratios):
                head_expand_ratios = expand_ratios
            if head_se_ratios is None or len(channels) != len(head_se_ratios):
                head_se_ratios = se_ratios
            self.head = ClassHead(
                channels=head_channels,
                dwk=head_dwk,
                dwkp=head_dwkp,
                fc_features=fc_features,
                expand_ratios=head_expand_ratios,
                se_ratios=head_se_ratios,
                classes=classes,
                dropout=head_dropout,
                norm=norm,
                norm_fn=norm_fn,
                norm_kwargs=norm_kwargs,
                act_fn=act_fn,
                se_sig_act=se_sig_act,
            )

        self._initialize()

    def add_rev_make_layer(
        self,
        channels: List[int] = [48, 64, 80],
        dwk: List[int] = [3, 5, 5, 3],
        dwkp: List[int] = [0, 0, 0, 2],
        new_ch: int = 160,
        inv_blk_rep: int = 2,
        expand_ratios: List[int] = [1, 1, 2, 2],
        se_ratios: List[float] = [0.25, 0.25, 0.125, 0.0],
        zero_init: bool = False,
        stochastic_depth: float = 0.0,
        upscale_mode: str = "bilinear",
        disable_rev: bool = False,
        norm: bool = True,
        norm_fn: nn.Module = nn.BatchNorm2d,
        norm_kwargs: dict = norm_kwargs_defaults,
        act_fn: nn.Module = nn.Hardswish,
        se_sig_act: nn.Module = nn.Hardsigmoid,
    ):
        # make RevSilo
        blk_args = {}
        blk_args["stochastic_depth"] = stochastic_depth
        blk_args["zero_init"] = zero_init
        blk_args["norm"] = norm
        blk_args["norm_fn"] = norm_fn
        blk_args["norm_kwargs"] = norm_kwargs
        blk_args["act_fn"] = act_fn
        blk_args["se_sig_act"] = se_sig_act
        for idx in range(inv_blk_rep):
            xforms = []
            for ch, expr, ser, _dwk, _dwkp in zip(channels, expand_ratios, se_ratios, dwk, dwkp):
                blk_args["kernel_size"] = _dwk + _dwkp
                blk_args["out_channels"] = ch // 2
                blk_args["in_channels"] = ch - blk_args["out_channels"]
                blk_args["expand_ratio"] = expr
                blk_args["se_ratio"] = ser

                g_blk_args = blk_args.copy()
                g_blk_args["in_channels"] = blk_args["out_channels"]
                g_blk_args["out_channels"] = blk_args["in_channels"]

                xforms += [RevResidualBlock(
                    f_transform=MBConvBlock(**blk_args),
                    g_transform=MBConvBlock(**g_blk_args),
                    disable_rev=disable_rev,
                )]
            self.rev_stack.append(RevSilo(xforms, disable_rev=disable_rev,))

        # create / add Fuse layer using RevResidualSilo
        if new_ch: channels += [new_ch]

        f_xforms, g_xforms = [], []
        for in_height, in_ch in enumerate(channels):
            f_xforms += [[]]
            g_xforms += [[]]
            for out_height, out_ch in enumerate(channels):
                h_diff = out_height - in_height

                if h_diff > 0:
                    blk_args["in_channels"] = channels[in_height]
                    blk_args["out_channels"] = channels[out_height]
                    blk_args["expand_ratio"] = expand_ratios[in_height]
                    blk_args["se_ratio"] = se_ratios[in_height]
                    blk_args["zero_init"] = zero_init

                    blk_args["stride"] = 2 ** h_diff
                    blk_args["kernel_size"] = s_based_k(blk_args["stride"], p=dwkp[in_height])

                    if new_ch and out_height == len(channels) - 1:
                        blk_args["zero_init"] = False
                        blk_args["stochastic_depth"] = 0.0
                    f_xforms[in_height] += [MBConvBlock(**blk_args)]
                    if new_ch and out_height == len(channels) - 1:
                        blk_args["zero_init"] = zero_init
                        blk_args["stochastic_depth"] = stochastic_depth

                elif h_diff < 0:
                    blk_args["stride"] = 1
                    blk_args["kernel_size"] = 3 + dwkp[in_height]
                    blk_args["in_channels"] = in_ch
                    blk_args["out_channels"] = out_ch
                    blk_args["expand_ratio"] = expand_ratios[out_height]
                    blk_args["se_ratio"] = se_ratios[out_height]
                    blk_args["zero_init"] = zero_init

                    xform = [
                        MBConvBlock(**blk_args),
                        nn.Upsample(
                            scale_factor=2 ** (-h_diff), mode=upscale_mode,
                        ),
                    ]
                    g_xforms[in_height] += [nn.Sequential(*xform)]

        self.rev_stack.append(RevResidualSilo(
            f_xforms,
            g_xforms,
            disable_rev=disable_rev,
        ))

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # zero_init
        for m in self.modules():
            if isinstance(m, ConvNormAct) and hasattr(m, "zero_init") and m.zero_init:
                nn.init.zeros_(m.norm.weight)

    def _stack_x(self, x):
        mc = self.in_channels % 3
        c_stack = (self.in_channels // 3) * [x]
        c_stack += [x[:,:mc]]
        return torch.cat(c_stack, dim=1)

    def _rev_stack_fwd(self, x):
        # Since networks is fully reversible, graph is not built unless input, x,
        # requires grad
        if isinstance(x, torch.Tensor):
            x = self._stack_x(x)
            x.requires_grad = True
        else:
            x = [self._stack_x(_x) for _x in x]
            for _x in x: _x.requires_grad = True

        return self.rev_stack(x)

    def forward(self, x):
        x = self._rev_stack_fwd(x)
        if self.head:
            return self.head(x)
        return x


ckpt_root = "CHECKPOINT_ROOT/revbifpn"
arch_params = {
    "revbifpn_s0": {
        "model_ckpt": f"{ckpt_root}/revbifpn_s0.pth.tar",
        "img_size": 224, "width_multplier": 1,
        "num_ext_layers": 2, "head_dropout": 0.25, "stochastic_depth": None
    },
    "revbifpn_s1": {
        "model_ckpt": f"{ckpt_root}/revbifpn_s1.pth.tar",
        "img_size": 256, "width_multplier": 4 / 3,
        "num_ext_layers": 2, "head_dropout": 0.25, "stochastic_depth": None
    },
    "revbifpn_s2": {
        "model_ckpt": f"{ckpt_root}/revbifpn_s2.pth.tar",
        "img_size": 256, "width_multplier": 2,
        "num_ext_layers": 2, "head_dropout": 0.3, "stochastic_depth": None
    },
    "revbifpn_s3": {
        "model_ckpt": f"{ckpt_root}/revbifpn_s3.pth.tar",
        "img_size": 288, "width_multplier": 8 / 3,
        "num_ext_layers": 3, "head_dropout": 0.3, "stochastic_depth": 0.05
    },
    "revbifpn_s4": {
        "model_ckpt": f"{ckpt_root}/revbifpn_s4.pth.tar",
        "img_size": 320, "width_multplier": 4,
        "num_ext_layers": 4, "head_dropout": 0.4, "stochastic_depth": 0.1
    },
    "revbifpn_s5": {
        "model_ckpt": f"{ckpt_root}/revbifpn_s5.pth.tar",
        "img_size": 352, "width_multplier": 16 / 3,
        "num_ext_layers": 4, "head_dropout": 0.4, "stochastic_depth": 0.1
    },
    "revbifpn_s6": {
        "model_ckpt": f"{ckpt_root}/revbifpn_s6.pth.tar",
        "img_size": 352, "width_multplier": 20 / 3,
        "num_ext_layers": 5, "head_dropout": 0.6, "stochastic_depth": 0.3
    },


default_args = {
    "channels": [48, 64, 80, 160],
    "head_channels": [48, 64, 128, 320],
    "fc_features": 1280,
}


def _revbifpn_args(
    arch: str,
    **kwargs_overrides: Any
) -> dict:
    img_size = arch_params[arch]["img_size"]
    print(f"Note: {arch} is pretrained using an input image size of {img_size}.")

    width_multplier = arch_params[arch]["width_multplier"]

    model_args = default_args.copy()
    # scale channel counts
    model_args["channels"] = [
        int(width_multplier * ch) // 16 * 16 for ch in model_args["channels"]]
    model_args["head_channels"] = [
        int(width_multplier * ch) // 16 * 16 for ch in model_args["head_channels"]]
    model_args["fc_features"] = int(width_multplier * model_args["fc_features"]) // 16 * 16
    # other model specific args
    model_args["num_ext_layers"] = arch_params[arch]["num_ext_layers"]
    model_args["head_dropout"] = arch_params[arch]["head_dropout"]
    if arch_params[arch]["stochastic_depth"]:
        model_args["stochastic_depth"] = arch_params[arch]["stochastic_depth"]

    # capture any overrides
    for k, v in kwargs_overrides.items():
        model_args[k] = v

    return model_args


class RevBiFPN_S(RevBiFPN):
    """
    RevBiFPN model of specicified scale
    """
    def __init__(
        self,
        arch: str,
        pretrained: bool = False,
        strict: bool = True,
        **kwargs_overrides
    ):
        model_args = _revbifpn_args(arch, **kwargs_overrides)
        super().__init__(**model_args)

        # load pretrained model
        if pretrained:
            if "norm_fn" in model_args:
                assert model_args["norm_fn"] in (nn.BatchNorm2d, nn.SyncBatchNorm)
            state_dict = torch.load(
                arch_params[arch]["model_ckpt"],
                map_location=next(self.parameters()).device,
            )
            self.load_state_dict(state_dict, strict=strict)


def _revbifpn(
    arch: str,
    pretrained: bool = False,
    strict: bool = True,
    **kwargs_overrides: Any
) -> RevBiFPN:
    model_args = _revbifpn_args(arch, **kwargs_overrides)

    # get model
    model = RevBiFPN(**model_args)

    # load pretrained model
    if pretrained:
        if "norm_fn" in model_args:
            assert model_args["norm_fn"] in (nn.BatchNorm2d, nn.SyncBatchNorm)
        state_dict = torch.load(
            arch_params[arch]["model_ckpt"],
            map_location=next(model.parameters()).device,
        )
        model.load_state_dict(state_dict, strict=strict)
    return model


def revbifpn_s0(
    pretrained: bool = False,
    strict: bool = True,
    **kwargs_overrides: Any
) -> Tuple[RevBiFPN, int]:
    r"""RevBiFPN-S0 model from
    `"RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network" <https://arxiv.org/abs/TODO>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        strict (bool): using kwargs and setting to False enables loading backbone without head
            setting head_channels=None instantiates a model without a neck
            setting classes=None instantiates a model without a head
    """
    return _revbifpn("revbifpn_s0", pretrained, strict, **kwargs_overrides)


def revbifpn_s1(
    pretrained: bool = False,
    strict: bool = True,
    **kwargs_overrides: Any
) -> Tuple[RevBiFPN, int]:
    r"""RevBiFPN-S1 model from
    `"RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network" <https://arxiv.org/abs/TODO>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        strict (bool): using kwargs and setting to False enables loading backbone without head
            setting head_channels=None instantiates a model without a neck
            setting classes=None instantiates a model without a head
    """
    return _revbifpn("revbifpn_s1", pretrained, strict, **kwargs_overrides)


def revbifpn_s2(
    pretrained: bool = False,
    strict: bool = True,
    **kwargs_overrides: Any
) -> Tuple[RevBiFPN, int]:
    r"""RevBiFPN-S2 model from
    `"RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network" <https://arxiv.org/abs/TODO>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        strict (bool): using kwargs and setting to False enables loading backbone without head
            setting head_channels=None instantiates a model without a neck
            setting classes=None instantiates a model without a head
    """
    return _revbifpn("revbifpn_s2", pretrained, strict, **kwargs_overrides)


def revbifpn_s3(
    pretrained: bool = False,
    strict: bool = True,
    **kwargs_overrides: Any
) -> Tuple[RevBiFPN, int]:
    r"""RevBiFPN-S3 model from
    `"RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network" <https://arxiv.org/abs/TODO>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        strict (bool): using kwargs and setting to False enables loading backbone without head
            setting head_channels=None instantiates a model without a neck
            setting classes=None instantiates a model without a head
    """
    return _revbifpn("revbifpn_s3", pretrained, strict, **kwargs_overrides)


def revbifpn_s4(
    pretrained: bool = False,
    strict: bool = True,
    **kwargs_overrides: Any
) -> Tuple[RevBiFPN, int]:
    r"""RevBiFPN-S4 model from
    `"RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network" <https://arxiv.org/abs/TODO>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        strict (bool): using kwargs and setting to False enables loading backbone without head
            setting head_channels=None instantiates a model without a neck
            setting classes=None instantiates a model without a head
    """
    return _revbifpn("revbifpn_s4", pretrained, strict, **kwargs_overrides)


def revbifpn_s5(
    pretrained: bool = False,
    strict: bool = True,
    **kwargs_overrides: Any
) -> Tuple[RevBiFPN, int]:
    r"""RevBiFPN-S5 model from
    `"RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network" <https://arxiv.org/abs/TODO>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        strict (bool): using kwargs and setting to False enables loading backbone without head
            setting head_channels=None instantiates a model without a neck
            setting classes=None instantiates a model without a head
    """
    return _revbifpn("revbifpn_s5", pretrained, strict, **kwargs_overrides)


def revbifpn_s6(
    pretrained: bool = False,
    strict: bool = True,
    **kwargs_overrides: Any
) -> Tuple[RevBiFPN, int]:
    r"""RevBiFPN-S6 model from
    `"RevBiFPN: The Fully Reversible Bidirectional Feature Pyramid Network" <https://arxiv.org/abs/TODO>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        strict (bool): using kwargs and setting to False enables loading backbone without head
            setting head_channels=None instantiates a model without a neck
                Note pretrained head is invalid without neck
            setting classes=None instantiates a model without a head
    """
    return _revbifpn("revbifpn_s6", pretrained, strict, **kwargs_overrides)


model_fns = {
    "revbifpn_s0": revbifpn_s0, "revbifpn_s1": revbifpn_s1, "revbifpn_s2": revbifpn_s2,
    "revbifpn_s3": revbifpn_s3, "revbifpn_s4": revbifpn_s4, "revbifpn_s5": revbifpn_s5,
    "revbifpn_s6": revbifpn_s6,
}


if __name__ == "__main__":
    # use [thop](https://pypi.org/project/thop/) to show mac and param count of models
    from thop import profile

    for arch, img_size in [
        ("revbifpn_s0", 224), ("revbifpn_s1", 256), ("revbifpn_s2", 256),
        ("revbifpn_s3", 288), ("revbifpn_s4", 320), ("revbifpn_s5", 352), ("revbifpn_s6", 352)
    ]:
        # alternative methods to generate the model
        # use a nn.Module vs model gen function
        for gen_model in [RevBiFPN_S, model_fns[arch]]:
            # different arg examples
            args = {}
            if gen_model == RevBiFPN_S: args["arch"] = arch
            # args["pretrained"] = True
            # args["strict"] = False          # if neck or head is removed, cant load strict ckpt
            # args["head_channels"] = None    # remove neck
            # args["classes"] = None          # remove head
            # args["norm"] = nn.BatchNorm2d   # alt: nn.SyncBatchNorm, nn.InstanceNorm2d
            # args["act_fn"] = nn.SiLU        # use alt activation
            # args["se_sig_act"] = nn.Sigmoid # use alt activation for squeeze-excite

            # get RevBiFPN model
            model = gen_model(**args)

            # profile network MAC and param count
            macs, params = profile(
                model, inputs=(torch.randn(1, 3, img_size, img_size), ), verbose=False,)
            total_params = sum(p.numel() for p in model.parameters())
            assert params == total_params, f"total_params: {total_params}, profiled params: {params}"
            print(f"Model uses {int(macs)} MACs and has {int(params)} parameters.")

            # Vitaliy - for me this outputs:
            # Note: revbifpn_s0 is pretrained using an input image size of 224.
            # Model uses 308039716 MACs and has 3419196 parameters.
            # Note: revbifpn_s0 is pretrained using an input image size of 224.
            # Model uses 308039716 MACs and has 3419196 parameters.
            # Note: revbifpn_s1 is pretrained using an input image size of 256.
            # Model uses 616781508 MACs and has 5112300 parameters.
            # Note: revbifpn_s1 is pretrained using an input image size of 256.
            # Model uses 616781508 MACs and has 5112300 parameters.
            # Note: revbifpn_s2 is pretrained using an input image size of 256.
            # Model uses 1370329000 MACs and has 10576528 parameters.
            # Note: revbifpn_s2 is pretrained using an input image size of 256.
            # Model uses 1370329000 MACs and has 10576528 parameters.
            # Note: revbifpn_s3 is pretrained using an input image size of 288.
            # Model uses 3330873312 MACs and has 19574184 parameters.
            # Note: revbifpn_s3 is pretrained using an input image size of 288.
            # Model uses 3330873312 MACs and has 19574184 parameters.
            # Note: revbifpn_s4 is pretrained using an input image size of 320.
            # Model uses 10632222080 MACs and has 48687336 parameters.
            # Note: revbifpn_s4 is pretrained using an input image size of 320.
            # Model uses 10632222080 MACs and has 48687336 parameters.
            # Note: revbifpn_s5 is pretrained using an input image size of 352.
            # Model uses 21848995912 MACs and has 81998976 parameters.
            # Note: revbifpn_s5 is pretrained using an input image size of 352.
            # Model uses 21848995912 MACs and has 81998976 parameters.
            # Note: revbifpn_s6 is pretrained using an input image size of 352.
            # Model uses 38084434748 MACs and has 142307972 parameters.
            # Note: revbifpn_s6 is pretrained using an input image size of 352.
            # Model uses 38084434748 MACs and has 142307972 parameters.
