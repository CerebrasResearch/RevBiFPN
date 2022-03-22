"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.
"""
import warnings

import torch
import torch.nn as nn

from .rev_op import RevOp
from .rev_silo import RevLimResidualSilo


class RevResidualBlock(RevLimResidualSilo):
    """
    Note: RevResidualBlock is a special case of the RevLimResidualSilo

    RevResidualBlock defined in
    `The Reversible Residual Network: Backpropagation Without Storing Activations
    <https://proceedings.neurips.cc/paper/2017/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html>`_
    """
    
    def forward(self, x):
        with torch.set_grad_enabled(not self._rev):
            if isinstance(x, (list, tuple)):
                assert len(x) == 1, "should be the right length"
                x = x[0]
            x = torch.chunk(x, 2, dim=1)

        y = super().forward(*x)

        with torch.set_grad_enabled(not self._rev):
            return torch.cat(y, dim=1)

    def backward_pass(self, y, dy):
        assert self._rev, "should only be called in _rev mode"
        with torch.no_grad():
            if isinstance(y, (list, tuple)):
                assert len(y) == 1, "should be the right length"
                y = y[0]
            if isinstance(dy, (list, tuple)):
                assert len(dy) == 1, "should be the right length"
                dy = dy[0]
            y = torch.chunk(y, 2, dim=1)
            dy = torch.chunk(dy, 2, dim=1)
        
        x, dx = super().backward_pass(y, dy)

        with torch.no_grad():
            x = torch.cat(x, dim=1)
            dx = torch.cat(dx, dim=1)

        return x, dx


class RevElementWiseAffine(RevOp):
    def __init__(
        self,
        num_features,
        eps=1e-16,
        weight=True,
        bias=True,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )

        self.num_features = num_features
        self.weight = None
        if weight:
            self.weight = nn.parameter.Parameter(
                torch.ones(num_features), requires_grad=True
            )
        self.bias = None
        if bias:
            self.bias = nn.parameter.Parameter(
                torch.zeros(num_features), requires_grad=True
            )
        self.eps = eps
        self.view_s = None
        self.grad_reduce_dims = None
        self.num_inputs = 1

    @property
    def clipped_weight(self):
        if self.weight is not None:
            w = self.weight.clone().view(self.view_s)
            w[(0 <= w) & (w < self.eps)] = self.eps
            w[(-self.eps < w) & (w < 0)] = -self.eps
            return w
        return None

    @property
    def inference_weight(self):
        if self.weight is not None:
            return self.clipped_weight
        return None

    def extra_repr(self):
        repr_str = f"rev={self._rev}, "
        repr_str += f"num_features={self.num_features}, "
        repr_str += f"weight={self.weight is not None}, "
        repr_str += f"bias={self.bias is not None}"
        return repr_str

    def forward(self, *x):
        with torch.set_grad_enabled(not self._rev):
            if isinstance(x, (list, tuple)):
                assert len(x) == 1, "should be the right length"
                x = x[0]
            if self.view_s is None:
                self.view_s = [1 for _ in x.shape]
                self.view_s[1] = self.num_features

            if self.weight is not None:
                x = self.clipped_weight * x

            if self.bias is not None:
                return x + self.bias.view(self.view_s)
            return x

    def backward_pass(self, y, dy):
        assert self._rev, "should only be called in _rev mode"
        with torch.no_grad():
            if isinstance(y, (list, tuple)):
                assert len(y) == 1, "should be the right length"
                y = y[0]
            if isinstance(dy, (list, tuple)):
                assert len(dy) == 1, "should be the right length"
                dy = dy[0]

            if self.grad_reduce_dims is None:
                self.grad_reduce_dims = list(range(len(dy.shape)))
                self.grad_reduce_dims.pop(1)

            x = y
            dx = dy
            if self.bias is not None:
                # Compute bias input
                x = x - self.bias.view(self.view_s)

                # compute grad_bias
                self.bias.grad = dy.sum(self.grad_reduce_dims)


            if self.weight is not None:
                w = self.clipped_weight

                # Compute input
                x = x / w
        
                # compute grad_weight
                grad_w = (x * dy).sum(self.grad_reduce_dims)
                grad_w[(-self.eps < self.weight) & (self.weight < self.eps)] = 0
                self.weight.grad = grad_w

                # backprop - compute grad_input
                dx = dx * w

            # clean-up
            x.grad = None
            x.detach()
            x.requires_grad = False
            dx.grad = None
            dx.detach()
            dx.requires_grad = False

            del y, dy

            return x, dx


class RevElementWiseAffineDiv(RevOp):
    def __init__(
        self,
        num_features,
        eps=1e-16,
        weight=True,
        bias=True,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )

        self.num_features = num_features
        self.weight = None
        if weight:
            self.weight = nn.parameter.Parameter(
                torch.ones(num_features), requires_grad=True
            )
        self.bias = None
        if bias:
            self.bias = nn.parameter.Parameter(
                torch.zeros(num_features), requires_grad=True
            )
        self.eps = eps
        self.view_s = None
        self.grad_reduce_dims = None
        self.num_inputs = 1

    @property
    def clipped_weight(self):
        if self.weight is not None:
            w = self.weight.clone().view(self.view_s)
            w[(0 <= w) & (w < self.eps)] = self.eps
            w[(-self.eps < w) & (w < 0)] = -self.eps
            return w
        return None

    @property
    def inference_weight(self):
        if self.weight is not None:
            return 1 / self.clipped_weight
        return None

    def extra_repr(self):
        repr_str = f"rev={self._rev}, "
        repr_str += f"num_features={self.num_features}, "
        repr_str += f"weight={self.weight is not None}, "
        repr_str += f"bias={self.bias is not None}"
        return repr_str

    def forward(self, *x):
        with torch.set_grad_enabled(not self._rev):
            if isinstance(x, (list, tuple)):
                assert len(x) == 1, "should be the right length"
                x = x[0]
            if self.view_s is None:
                self.view_s = [1 for _ in x.shape]
                self.view_s[1] = self.num_features

            if self.weight is not None:
                x = x / self.clipped_weight

            if self.bias is not None:
                return x + self.bias.view(self.view_s)
            return x

    def backward_pass(self, y, dy):
        assert self._rev, "should only be called in _rev mode"
        with torch.no_grad():
            if isinstance(y, (list, tuple)):
                assert len(y) == 1, "should be the right length"
                y = y[0]
            if isinstance(dy, (list, tuple)):
                assert len(dy) == 1, "should be the right length"
                dy = dy[0]

            if self.grad_reduce_dims is None:
                self.grad_reduce_dims = list(range(len(dy.shape)))
                self.grad_reduce_dims.pop(1)

            x = y
            dx = dy
            if self.bias is not None:
                # Compute bias input
                x = x - self.bias.view(self.view_s)

                # compute grad_bias
                self.bias.grad = dy.sum(self.grad_reduce_dims)


            if self.weight is not None:
                w = self.clipped_weight

                # Compute input
                x = x * w
        
                # compute grad_weight
                grad_w = (x * dy).sum(self.grad_reduce_dims)
                grad_w = - grad_w / (w.squeeze() ** 2)
                grad_w[(-self.eps < self.weight) & (self.weight < self.eps)] = 0
                self.weight.grad = grad_w

                # backprop - compute grad_input
                dx = dx / w

            # clean-up
            x.grad = None
            x.detach()
            x.requires_grad = False
            dx.grad = None
            dx.detach()
            dx.requires_grad = False

            del y, dy

            return x, dx


class RevNorm(RevOp):
    def __init__(
        self,
        num_features,
        eps=1e-8,
        momentum=0.1,
        affine=True,
        div=False,
        weight=True,
        affine_eps=1e-16,
        bias=True,
        disable_rev=False,
        preserve_rng_state=True,
        preserve_op_state=True,
    ):
        super().__init__(
            disable_rev=disable_rev,
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state,
        )
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features

        self.register_buffer(
            "r_mean", torch.zeros((num_features), dtype=torch.float)
        )
        self.register_buffer(
            "r_var", torch.ones((num_features), dtype=torch.float)
        )

        AffineClass = None
        self.affine = None
        if affine:
            AffineClass = RevElementWiseAffine
        if div:
            AffineClass = RevElementWiseAffineDiv

        if AffineClass is not None:
            self.affine = AffineClass(
                num_features,
                eps=affine_eps,
                weight=weight,
                bias=bias,
                disable_rev=disable_rev
            )

        self.reduce_dims = None
        self.view_s = None

        self.mean = None
        self.scale = None
        self.num_inputs = 1

        self.in_type = None

    def extra_repr(self):
        return f"rev={self._rev}, num_features={self.num_features}"

    def forward(self, *x):
        with torch.no_grad():
            if isinstance(x, (list, tuple)):
                assert len(x) == 1, "should be the right length"
                x = x[0]

        self.in_type = x.type()
        if self.in_type != self.r_mean.type():
            warnings.warn(f"Norm using type: {self.r_mean.type()}, "
                          f"input type: {self.in_type}")
            x = x.type(self.r_mean.type())

        with torch.no_grad():
            if self.reduce_dims is None:
                self.reduce_dims = list(range(len(x.shape)))
                self.reduce_dims.pop(1)

            if self.view_s is None:
                self.view_s = [1 for _ in x.shape]
                self.view_s[1] = self.num_features
                if self.affine is not None:
                    self.affine.view_s = self.view_s

        if not self.training:
            w, b = None, None
            if self.affine is not None:
                # if self.affine.weight is not None:
                w = self.affine.inference_weight
                # if self.affine.bias is not None:
                b = self.affine.bias
            out = nn.functional.batch_norm(
                x,
                self.r_mean,
                self.r_var,
                weight=w,
                bias=b,
                training=self.training,
                momentum=self.momentum,
                eps=self.eps
            )
            if self.in_type != self.r_mean.type():
                return out.type(self.in_type)
            return out

        with torch.set_grad_enabled(not self._rev):
            self.mean = x.mean(self.reduce_dims, keepdim=True)
            var = (x - self.mean).pow(2).mean(self.reduce_dims, keepdim=True)
            self.scale = (var + self.eps).sqrt()

            with torch.no_grad():
                updt = self.mean.squeeze() - self.r_mean
                self.r_mean += self.momentum * updt
                updt = var.squeeze() - self.r_var
                self.r_var += self.momentum * updt

            out = (x - self.mean) / self.scale

        if self.affine is not None:
            out = self.affine(out)

        if self.in_type != self.r_mean.type():
            return out.type(self.in_type)
        return out

    def backward_pass(self, y, dy):
        if self.in_type != self.r_mean.type():
            y, dy = y.type(self.r_mean.type()), dy.type(self.r_mean.type())

        if self.affine is not None:
            y, dy = self.affine.backward_pass(y, dy)

        assert self._rev, "should only be called in _rev mode"
        with torch.no_grad():
            if isinstance(y, (list, tuple)):
                assert len(y) == 1, "should be the right length"
                y = y[0]
            if isinstance(dy, (list, tuple)):
                assert len(dy) == 1, "should be the right length"
                dy = dy[0]

            # Compute input
            x = self.scale * y + self.mean

            # backprop - compute grad_input
            dx = dy - dy.mean(self.reduce_dims, keepdim=True)
            dx -= y * (y * dy).mean(self.reduce_dims, keepdim=True)
            dx /= self.scale

            # clean-up
            x.grad = None
            x.detach()
            x.requires_grad = False
            dx.grad = None
            dx.detach()
            dx.requires_grad = False
            
            del y, dy, self.scale
            self.scale = None

            if self.in_type != self.r_mean.type():
                return x.type(self.in_type), dx.type(self.in_type)
            return x, dx
