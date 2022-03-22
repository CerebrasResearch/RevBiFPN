"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.

MIT License
Copyright (c) 2018 Jörn Jacobsen
From: https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py
"""

import torch
import torch.nn as nn

from .rev_op import RevOp


class RevSpatialDownsample(RevOp):
    def __init__(
        self,
        block_size,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def inverse(self, x):
        with torch.no_grad():
            if isinstance(x, (list, tuple)):
                assert len(x) == 1, "should be the right length"
                x = x[0]
            bl, bl_sq = self.block_size, self.block_size_sq
            bs, new_d, h, w = x.shape[0], x.shape[1] // bl_sq, x.shape[2], x.shape[3]
            return x.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)

    def forward(self, x):
        with torch.set_grad_enabled(not self._rev):
            if isinstance(x, (list, tuple)):
                assert len(x) == 1, "should be the right length"
                x = x[0]
            bl, bl_sq = self.block_size, self.block_size_sq
            bs, d, new_h, new_w = x.shape[0], x.shape[1], x.shape[2] // bl, x.shape[3] // bl
            return x.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)


    def backward_pass(self, y, dy):
        with torch.set_grad_enabled(not self._rev):
            return self.inverse(y), self.inverse(dy)
