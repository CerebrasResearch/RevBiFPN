"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.
"""

import torch
import torch.nn as nn

from .rev_op import RevOp


class RevAdditiveCouplingSilo(RevOp):
    def __init__(
        self,
        transforms,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )
        assert isinstance(transforms, (list, tuple))
        for xforms in transforms:
            assert isinstance(xforms, (list, tuple))

        self.num_outputs = len(transforms[0]) + 1

        for i, xforms in enumerate(transforms):
            assert i + len(xforms) + 1 == self.num_outputs
            for j, xform in enumerate(xforms):
                setattr(self, "xform_" + str(i) + "_" + str(j + i + 1), xform)
                assert j + i + 1 < self.num_outputs
        self.xforms = transforms

    def forward(self, *x):
        if self.num_inputs is None:
            self.num_inputs = len(x)
            assert len(self.xforms) in [self.num_inputs - 1, self.num_inputs, self.num_inputs + 1]
            assert self.num_outputs in [self.num_inputs, self.num_inputs + 1]
        assert len(x) == self.num_inputs

        with torch.set_grad_enabled(not self._rev):
            y = list(x)
            y += (self.num_outputs - self.num_inputs) * [None]

            for i in range(self.num_inputs):
                for j in range(i + 1, self.num_outputs):
                    xform = getattr(self, "xform_" + str(i) + "_" + str(j))
                    if xform is not None:
                        fx_ij = self.launch_op_fwd(xform, x[i])
                        y[j] = fx_ij if y[j] is None else y[j] + fx_ij

        return tuple(y)

    def backward_pass(self, y, dy):
        assert self._rev, "should only be called in _rev mode"
        for y_i in y:
            assert (not y_i.requires_grad), "y_i must already be detached"
        for dy_i in dy:
            assert (not dy_i.requires_grad), "dy_i must not require grad"

        with torch.no_grad():
            x = list(y[:self.num_inputs])
            # Note dx is stored and accumulated into x[i].grad
            # explicitly creating dx and manually accumulating into it uses
            # extra compute and memory allocation
            for i in range(self.num_inputs):
                x[i].requires_grad = True
                # x[i].grad will store the accumulated grad of dx[i]
                x[i].grad = dy[i].detach().clone()

        for i in range(self.num_inputs):
            for j in range(i + 1, self.num_outputs):
                xform = getattr(self, "xform_" + str(i) + "_" + str(j))
                if xform is not None:
                    with torch.enable_grad():
                        # recompute and grad
                        fx_ij = self.launch_op_bwd(xform, x[i])
                        # NOTE: since x[i].requires_grad = True, the gradient
                        # of x[i] is accumulated after this backward call.
                        fx_ij.backward(dy[j])

                    with torch.no_grad():
                        if j < self.num_inputs:
                            # must be an inplace operation to preserve x[j].grad
                            # also saves us a malloc
                            x[j] -= fx_ij

        with torch.no_grad():
            dx = [x_i.grad for x_i in x]

        for x_i, dx_i in zip(x, dx):
            x_i.detach()
            x_i.grad = None
            x_i.requires_grad = False
            dx_i.detach()
            dx_i.requires_grad = False

        del y, dy

        return x, dx


class G_RevAdditiveCouplingSilo(RevAdditiveCouplingSilo):
    def __init__(
        self,
        transforms,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        transforms.reverse()
        for xforms in transforms:
            xforms.reverse()
        super().__init__(
            transforms,
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )

    def forward(self, *x):
        x = x[::-1]
        y = super().forward(*x)
        return tuple(y[::-1])

    def backward_pass(self, y, dy):
        y, dy = y[::-1], dy[::-1]
        x, dx = super().backward_pass(y, dy)
        return x[::-1], dx[::-1]


class RevResidualSilo(RevOp):
    def __init__(
        self,
        f_transform,
        g_transform,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )
        self.f = RevAdditiveCouplingSilo(f_transform, disable_rev=disable_rev)
        self.g = G_RevAdditiveCouplingSilo(g_transform, disable_rev=disable_rev)

    def forward(self, *x):
        x = self.f(x) if isinstance(x, torch.Tensor) else self.f(*x)
        return self.g(x) if isinstance(x, torch.Tensor) else self.g(*x)

    def backward_pass(self, y, dy):
        return self.f.backward_pass(*self.g.backward_pass(y, dy))


class RevLimAdditiveCouplingSilo(RevAdditiveCouplingSilo):
    def __init__(
        self,
        transforms,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]
        transforms.reverse()
        xforms = []
        for n, xform in enumerate(transforms):
            xforms += [[xform] + n * [None]]
        xforms.reverse()
        super().__init__(
            xforms,
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )


class G_RevLimAdditiveCouplingSilo(RevLimAdditiveCouplingSilo):
    def __init__(
        self,
        transforms,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]
        transforms.reverse()
        super().__init__(
            transforms,
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )

    def forward(self, *x):
        x = x[::-1]
        y = super().forward(*x)
        return tuple(y[::-1])

    def backward_pass(self, y, dy):
        y, dy = y[::-1], dy[::-1]
        x, dx = super().backward_pass(y, dy)
        return x[::-1], dx[::-1]


class RevLimResidualSilo(RevOp):
    def __init__(
        self,
        f_transform,
        g_transform,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )
        self.f = RevLimAdditiveCouplingSilo(f_transform, disable_rev=disable_rev)
        self.g = G_RevLimAdditiveCouplingSilo(g_transform, disable_rev=disable_rev)
        assert len(self.f.xforms) == len(self.g.xforms)

    def forward(self, *x):
        x = self.f(x) if isinstance(x, torch.Tensor) else self.f(*x)
        return self.g(x) if isinstance(x, torch.Tensor) else self.g(*x)

    def backward_pass(self, y, dy):
        return self.f.backward_pass(*self.g.backward_pass(y, dy))


class RevSilo(RevOp):
    def __init__(
        self,
        rev_transforms,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )
        if rev_transforms:
            if not isinstance(rev_transforms, (list, tuple)):
                rev_transforms = [rev_transforms]
            
            for idx, tform in enumerate(rev_transforms):
                setattr(self, "rev_transform_" + str(idx), tform)
            self.xforms = rev_transforms

    def forward(self, *x):
        if self.num_inputs is None:
            self.num_inputs = len(x)
            assert len(self.xforms) == self.num_inputs
        else:
            assert self.num_inputs == len(x)

        y = []
        with torch.set_grad_enabled(not self._rev):
            for idx in range(self.num_inputs):
                if self.xforms[idx] is not None:
                    y += [self.launch_op_fwd(self.xforms[idx], x[idx])]
                else:
                    y += [x[idx]]
        return tuple(y)

    def backward_pass(self, y, dy):
        assert self._rev, "should only be called in _rev mode"
        for y_i in y:
            assert (not y_i.requires_grad), "y_i must already be detached"
        for dy_i in dy:
            assert (not dy_i.requires_grad), "dy_i must not require grad"

        x, dx = [], []

        for idx in range(self.num_inputs):
            if self.xforms[idx] is not None:
                x_i, dx_i = self.launch_op_bwd(self.xforms[idx], y[idx], dy[idx])
                x += [x_i]
                dx += [dx_i]
            else:
                x += [y[idx]]
                dx += [dy[idx]]

        for x_i, dx_i in zip(x, dx):
            x_i.detach()
            x_i.requires_grad = False
            dx_i.detach()
            dx_i.requires_grad = False

        del y, dy

        return x, dx
