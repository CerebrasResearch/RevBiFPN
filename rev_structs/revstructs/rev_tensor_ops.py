"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.
"""

import torch
import torch.nn as nn

from .rev_op import RevOp


class RevChunk(RevOp):
    def __init__(
        self,
        chunks,
        dim=0,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )
        self.chunks = chunks
        self.dim = dim

    def inverse(self, x):
        return torch.cat(x, dim=self.dim)

    def forward(self, *x):
        with torch.set_grad_enabled(not self._rev):
            if isinstance(x, (list, tuple)):
                assert len(x) == 1, "should be the right length"
                x = x[0]
            assert isinstance(x, torch.Tensor), "should be a tensor"
            return tuple(torch.chunk(x, self.chunks, dim=self.dim))

    def backward_pass(self, y, dy):
        return self.inverse(y), self.inverse(dy)


class RevCat(RevChunk):
    def __init__(
        self,
        dim=0,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            chunks=None,
            dim=dim,
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )

    def forward(self, *x):
        with torch.set_grad_enabled(not self._rev):
            if self.chunks is None:
                self.chunks = len(x)
            else:
                assert self.chunks == len(x)
            return torch.cat(x, dim=self.dim)

    def inverse(self, x):
        if not isinstance(x, torch.Tensor):
            assert len(x) == 1
            x = x[0]
        return torch.chunk(x, self.chunks, dim=self.dim)
