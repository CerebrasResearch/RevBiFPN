"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.
"""
import torch
import torch.nn as nn

from .rev_op import RevOp


class RecomputeOp(RevOp):
    r"""Wrapper class for recompute operations ie reverse checkpoint.

    Designed to be used as last operation in stack of reversible operations in a
    RevSequential neural network.
    """
    def __init__(
        self,
        xform,
        disable_recomp=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_recomp,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )
        self._recomp = not disable_recomp
        self.xform = xform
        self.num_inputs = 1

    def extra_repr(self):
        return f"recomp={self._recomp}"

    def forward(self, *x):
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, "should be the right length"
            x = x[0]
            
        with torch.set_grad_enabled(not self._recomp):
            y = self.launch_op_fwd(self.xfrom, x)

        return y

    def backward_pass(self, x, dy):
        assert self._recomp, "should only be called in _recomp mode"
        assert (not dy.requires_grad), "dy must not require grad"

        with torch.enable_grad():
            x.requires_grad = True
            y = self.launch_op_bwd(self.xfrom, x)
            y.backward(dy)

        del y, dy

        dx = x.grad
        dx.detach()
        dx.requires_grad = False
        x.requires_grad = False

        return x, dx


class RecomputeSilo(RevOp):
    def __init__(
        self,
        recomp_transforms,
        disable_recomp=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_recomp,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )
        self._recomp = not disable_recomp
        if recomp_transforms:
            if not isinstance(recomp_transforms, (list, tuple)):
                recomp_transforms = [recomp_transforms]

            self.num_inputs = len(recomp_transforms)
            
            for idx, tform in enumerate(recomp_transforms):
                setattr(self, "recomp_transform_" + str(idx), tform)
            self.xforms = recomp_transforms

    def extra_repr(self):
        return f"recomp={self._recomp}"

    def forward(self, *x):
        assert self.num_inputs == len(x)

        y = []
        with torch.set_grad_enabled(not self._recomp):
            for x_i, xform_i in zip(x, self.xforms):
                if xform_i is not None:
                    y += [self.launch_op_fwd(xform_i, x_i)]
                else:
                    y += [x_i]

        return tuple(y)

    def backward_pass(self, x, dy):
        assert self._recomp, "should only be called in _recomp mode"
        for dy_i in dy:
            assert (not dy_i.requires_grad), "dy_i must not require grad"

        with torch.enable_grad():
            dx = []
            for xform_i, x_i, dy_i in zip(self.xforms, x, dy):
                if xform_i is not None:
                    x_i.requires_grad = True
                    # create y_i and do bwd with dy_i, we dont actually need y_i
                    # xform_i(x_i).backward(dy_i)
                    self.launch_op_bwd(xform_i, x_i).backward(dy_i)

                    dx += [x_i.grad]
                else:
                    dx += [dy_i]

        for x_i, dx_i in zip(x, dx):
            x_i.detach()
            x_i.grad = None
            x_i.requires_grad = False
            dx_i.detach()
            dx_i.requires_grad = False

        del dy

        return x, dx
