"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.

Tests the different reversible structures in RevStructs
"""
import os
import sys
import logging
import unittest
import numpy as np

import torch
from torch import nn
from torch.utils.checkpoint import (
    get_device_states,
    set_device_states,
)

from revstructs import (
    RevSequential,
    RevSpatialDownsample,
    RevChunk,
    RevCat,
    RevAdditiveCouplingSilo,
    G_RevAdditiveCouplingSilo,
    RevResidualSilo,
    RevLimAdditiveCouplingSilo,
    G_RevLimAdditiveCouplingSilo,
    RevLimResidualSilo,
    RevResidualBlock,
    RevSilo,
    RevElementWiseAffine,
    RevElementWiseAffineDiv,
    RevNorm,
    RecomputeOp,
    RecomputeSilo
)


torch.manual_seed(42)
torch.set_deterministic(True)
np.random.seed(42)


class TestRevStructs(unittest.TestCase):
    """
    These tests compare the gradient produced using regular backpropagation vs
    backpropagating with reversible re-computation of activations
    """
    def base_test(self, x, model, final_out=0, n=1, amp=False):
        for x_i in x: x_i.requires_grad = True

        # set up rng_state
        fwd_cpu_state = torch.get_rng_state()
        had_cuda_in_fwd = False
        if torch.cuda._initialized:
            had_cuda_in_fwd = True
            fwd_gpu_devices, fwd_gpu_states = get_device_states(x)

        grad_cache = {}
        for disable_rev in [False, True]:
            # set rng_state
            rng_devices = []
            if had_cuda_in_fwd:
                rng_devices = fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                torch.set_rng_state(fwd_cpu_state)
                if had_cuda_in_fwd:
                    set_device_states(fwd_gpu_devices, fwd_gpu_states)

                grad_cache[disable_rev] = {}

                for x_i in x: x_i.grad = None
                for m in model.modules():
                    if hasattr(m, "_rev"):
                        m._rev = not disable_rev
                    if hasattr(m, "_recomp"):
                        m._recomp = not disable_rev

                model.zero_grad()
                for p in model.parameters(): p.grad = None
                model.train()
                for _ in range(n):
                    with torch.cuda.amp.autocast(amp):
                        y = model(x)
                        # compute dummy loss
                        loss = sum(y[final_out:]).sum() if final_out else sum(y).sum()
                    # backprop thru model
                    loss.backward()

                # cache grad of parameters and the input
                grad_cache[disable_rev]["input"] = [
                    x_i.grad.detach().clone() for x_i in x
                ]
                grad_cache[disable_rev]["params"] = []
                for p in model.parameters():
                    grad_cache[disable_rev]["params"] += [p.grad.detach().clone()]

        for idx in range(len(x)):
            check_grad_input = grad_cache[True]["input"][idx].allclose(
                grad_cache[False]["input"][idx],
                rtol=1e-04, atol=1e-06
            )
            if not check_grad_input:
                print(grad_cache[True]["input"][idx], grad_cache[False]["input"][idx])
            assert check_grad_input
        for idx in range(len(grad_cache[True]["params"])):
            check_grad_param = grad_cache[True]["params"][idx].allclose(
                grad_cache[False]["params"][idx],
                rtol=1e-04, atol=1e-06
            )
            if not check_grad_param:
                print(grad_cache[True]["params"][idx], grad_cache[False]["params"][idx])
            assert check_grad_param

    def test010(self):
        x = [torch.randn(2, 2, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(2)]

        model = RevSequential([
            RevSpatialDownsample(2),
            RevChunk(2, 1),
            RevLimAdditiveCouplingSilo(layers[0]),
            RevLimAdditiveCouplingSilo(layers[1]),
        ])

        self.base_test(x, model)

    def test020(self):
        x = [torch.randn(2, 2, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(4)]

        model = RevSequential([
            RevSpatialDownsample(2),
            RevChunk(2, 1),
            RevLimAdditiveCouplingSilo(layers[0]),
            G_RevLimAdditiveCouplingSilo(layers[1]),
            RevLimAdditiveCouplingSilo(layers[2:]),
        ])

        self.base_test(x, model)

    def test030(self):
        x = [torch.randn(2, 2, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(6)]

        model = RevSequential([
            RevSpatialDownsample(2),
            RevChunk(2, 1),
            RevLimAdditiveCouplingSilo(layers[0]),
            G_RevLimAdditiveCouplingSilo(layers[1]),
            RevLimResidualSilo(layers[4], layers[5]),
            RevLimAdditiveCouplingSilo([layers[2], layers[3]]),
        ])

        self.base_test(x, model)

    def test040(self):
        x = [torch.randn(2, 2, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(6)]

        model = RevSequential([
            RevSpatialDownsample(2),
            RevChunk(2, 1),
            RevLimAdditiveCouplingSilo(layers[0]),
            G_RevLimAdditiveCouplingSilo(layers[1]),
            RevLimResidualSilo(layers[2:4], layers[4:]),
        ])

        self.base_test(x, model)

    def test050(self):
        x = [torch.randn(2, 2, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        )]
        layers += [nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, padding=1,
        )]

        model = RevSequential([
            RevSpatialDownsample(2),
            RevChunk(2, 1),
            RevLimAdditiveCouplingSilo(layers[0]),
            RevCat(1),
            G_RevLimAdditiveCouplingSilo(layers[1]),
        ])

        self.base_test(x, model)

    def test060(self):
        x = [torch.randn(2, 2, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(3)]
        
        model = RevSequential([
                    RevSpatialDownsample(2),
                    RevChunk(2, 1),
                    RevLimAdditiveCouplingSilo(layers[0]),
                    G_RevLimAdditiveCouplingSilo(layers[1]),
                    RevLimAdditiveCouplingSilo([None, layers[2]]),
                ])

        self.base_test(x, model)

    def test070(self):
        x = [torch.randn(2, 2, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(2)]
        
        model = RevSequential([
            RevSpatialDownsample(2),
            RevResidualBlock(layers[0], layers[1]),
            RevChunk(2, 1),
        ])

        self.base_test(x, model)

    def test071(self):
        x = [torch.randn(2, 2, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(2)]
        
        model = RevSequential([
            RevSpatialDownsample(2),
            RevSilo([RevResidualBlock(layers[0], layers[1]),],),
        ])

        self.base_test(x, model)

    def test080(self):
        x = [torch.randn(2, 4, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(2)]

        model = RevSequential([
            RevLimResidualSilo(layers[0], layers[1]),
        ])

        self.base_test(x, model)

    def test090(self):
        x = [torch.randn(2, 4, 2, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(12)]
        
        model = RevSequential([
                    RevLimAdditiveCouplingSilo(layers[0]),
                    G_RevLimAdditiveCouplingSilo(layers[1]),
                    RevLimAdditiveCouplingSilo(layers[2:4]),
                    RevLimResidualSilo(layers[4:7], [None] + layers[7:9]),
                    RevLimResidualSilo([None] + layers[9:11], [None, None] + [layers[11]]),
                ])

        self.base_test(x, model, final_out=2)

    def test100(self):
        x = [torch.randn(2, 4, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(2)]
        layers += [nn.Conv2d(
            in_channels=2, out_channels=2, kernel_size=3, padding=1
        ) for _ in range(4)]

        model = RevSequential([
            RevLimResidualSilo(layers[0], layers[1]),
            RevSilo([RevResidualBlock(*layers[2:4]), RevResidualBlock(*layers[4:])]),
        ])

        self.base_test(x, model)

    def test110(self):
        x = [torch.randn(2, 4, 6, 4)]

        layers = [nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, padding=1
        ) for _ in range(2)]

        model = RevSequential([
            RevLimResidualSilo(layers[0], layers[1]),
            RevSilo([RevElementWiseAffine(4), RevElementWiseAffine(4)]),
        ])

        self.base_test(x, model)

    def test120(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        tbn = nn.BatchNorm2d(C, eps=0)
        rbn = RevNorm(C, eps=0, disable_rev=False)

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = tbn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)
        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)

    def test130(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        bn = RevNorm(C, eps=0, disable_rev=True)
        rbn = RevNorm(C, eps=0, disable_rev=False)

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = bn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)
        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)

    def test131(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        bn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=False, bias=False, disable_rev=True)
        rbn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=False, bias=False, disable_rev=False)

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = bn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)
        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)

    def test132(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        bn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=True, bias=False, disable_rev=True)
        rbn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=True, bias=False, disable_rev=False)

        with torch.no_grad():
            bn.affine.weight.data = 2 * torch.randn(bn.affine.weight.shape) + .05
            rbn.affine.weight.data = bn.affine.weight.data.clone().detach()

        assert rbn.affine.weight.allclose(bn.affine.weight, rtol=rtol, atol=atol)

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = bn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)
        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)
        # compare grad weight
        assert rbn.affine.weight.grad.allclose(bn.affine.weight.grad, rtol=rtol, atol=atol)

    def test133(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        bn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=False, bias=True, disable_rev=True)
        rbn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=False, bias=True, disable_rev=False)

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = bn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)
        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)
        # compare grad weight
        assert rbn.affine.bias.grad.allclose(bn.affine.bias.grad, rtol=rtol, atol=atol)

    def test134(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        bn = RevNorm(C, eps=1e-16, affine=False, div=True, weight=False, bias=False, disable_rev=True)
        rbn = RevNorm(C, eps=1e-16, affine=False, div=True, weight=False, bias=False, disable_rev=False)

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = bn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)
        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)

    def test135(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        bn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=True, bias=True, disable_rev=True)
        rbn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=True, bias=True, disable_rev=False)

        with torch.no_grad():
            bn.affine.weight.data = 2 * torch.randn(bn.affine.weight.shape) + .05
            rbn.affine.weight.data = bn.affine.weight.data.clone().detach()

            bn.affine.bias.data = 2 * torch.randn(bn.affine.bias.shape)
            rbn.affine.bias.data = bn.affine.bias.data.clone().detach()

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = bn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)
        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)
        # compare grad weight
        assert rbn.affine.weight.grad.allclose(bn.affine.weight.grad, rtol=rtol, atol=atol)
        assert rbn.affine.bias.grad.allclose(bn.affine.bias.grad, rtol=rtol, atol=atol)

    def test136(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        bn = RevNorm(C, eps=1e-16, affine=False, div=True, weight=True, bias=True, disable_rev=True)
        rbn = RevNorm(C, eps=1e-16, affine=False, div=True, weight=True, bias=True, disable_rev=False)

        with torch.no_grad():
            bn.affine.weight.data = 2 * torch.randn(bn.affine.weight.shape) + .05
            rbn.affine.weight.data = bn.affine.weight.data.clone().detach()

            bn.affine.bias.data = 2 * torch.randn(bn.affine.bias.shape)
            rbn.affine.bias.data = bn.affine.bias.data.clone().detach()

        assert rbn.affine.weight.allclose(bn.affine.weight, rtol=rtol, atol=atol)
        assert rbn.affine.bias.allclose(bn.affine.bias, rtol=rtol, atol=atol)

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = bn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)
        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)
        # compare grad weight
        assert rbn.affine.weight.grad.allclose(bn.affine.weight.grad, rtol=rtol, atol=atol)
        assert rbn.affine.bias.grad.allclose(bn.affine.bias.grad, rtol=rtol, atol=atol)


    def test137(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        bn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=True, bias=True, disable_rev=True)
        rbn = RevNorm(C, eps=1e-16, affine=True, div=False, weight=True, bias=True, disable_rev=False)

        with torch.no_grad():
            bn.affine.weight.data = 2 * torch.randn(bn.affine.weight.shape) + .05
            rbn.affine.weight.data = bn.affine.weight.data.clone().detach()

            bn.affine.bias.data = 2 * torch.randn(bn.affine.bias.shape)
            rbn.affine.bias.data = bn.affine.bias.data.clone().detach()

        bn.double()
        rbn.double()

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = bn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)

        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)
        # compare grad weight
        assert rbn.affine.weight.grad.allclose(bn.affine.weight.grad, rtol=rtol, atol=atol)
        assert rbn.affine.bias.grad.allclose(bn.affine.bias.grad, rtol=rtol, atol=atol)

    def test138(self):
        N, C, H, W = 2, 3, 2, 2
        mu, std = 0.3, 1.5
        rtol, atol = 1e-04, 1e-07
        bn = RevNorm(C, eps=1e-16, affine=False, div=True, weight=True, bias=True, disable_rev=True)
        rbn = RevNorm(C, eps=1e-16, affine=False, div=True, weight=True, bias=True, disable_rev=False)

        with torch.no_grad():
            bn.affine.weight.data = 2 * torch.randn(bn.affine.weight.shape) + .05
            rbn.affine.weight.data = bn.affine.weight.data.clone().detach()

            bn.affine.bias.data = 2 * torch.randn(bn.affine.bias.shape)
            rbn.affine.bias.data = bn.affine.bias.data.clone().detach()

        assert rbn.affine.weight.allclose(bn.affine.weight, rtol=rtol, atol=atol)
        assert rbn.affine.bias.allclose(bn.affine.bias, rtol=rtol, atol=atol)

        bn.double()
        rbn.double()

        # gen input
        x = std * torch.randn(N, C, H, W) + mu
        rx = x.clone().detach()
        x.requires_grad = True
        rx.requires_grad = True

        # compare inputs
        assert x.allclose(rx, rtol=rtol, atol=atol)
        
        # compute output
        y = bn(x)
        ry = rbn(rx)
        # compare outputs
        assert y.allclose(ry, rtol=rtol, atol=atol)

        # gen delta
        dy = torch.randn(N, C, H, W)
        rdy = dy.clone().detach()

        # backprop
        y.backward(dy)
        _rx, rdx = rbn.backward_pass(ry, rdy)
        # check recomputed input
        assert rx.allclose(_rx, rtol=rtol, atol=atol)
        # compare grad
        assert x.grad.allclose(rdx, rtol=rtol, atol=atol)
        # compare grad weight
        assert rbn.affine.weight.grad.allclose(bn.affine.weight.grad, rtol=rtol, atol=atol)
        assert rbn.affine.bias.grad.allclose(bn.affine.bias.grad, rtol=rtol, atol=atol)

    def test140(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        )]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            RevAdditiveCouplingSilo([[None], []]),
        ])

        self.base_test(x, model)

    def test150(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(2)]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            RevAdditiveCouplingSilo([[layers[1]], []]),
        ])

        self.base_test(x, model)

    def test160(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(4)]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            RevAdditiveCouplingSilo([layers[1:3], [layers[3]]]),
        ])

        self.base_test(x, model)

    def test170(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(10)]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            RevAdditiveCouplingSilo([layers[1:3], [layers[3]]]),
            RevAdditiveCouplingSilo(
                [
                    layers[4:7],
                    layers[7:9],
                    [layers[9]]
                ]
            ),
        ])

        self.base_test(x, model)

    def test180(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(10)]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            RevAdditiveCouplingSilo([layers[1:3], [layers[3]]]),
            RevAdditiveCouplingSilo(
                [
                    [layers[4], None, layers[6]],
                    [None, layers[8]],
                    [layers[9]]
                ]
            ),
        ])

        self.base_test(x, model)

    def test190(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        )]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            G_RevAdditiveCouplingSilo([[], [None]]),
        ])

        self.base_test(x, model)

    def test200(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(2)]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            G_RevAdditiveCouplingSilo([[], [layers[1]]]),
        ])

        self.base_test(x, model)

    def test210(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(4)]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            G_RevAdditiveCouplingSilo([[layers[1]], layers[2:]]),
        ])

        self.base_test(x, model)

    def test220(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(10)]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            RevAdditiveCouplingSilo([layers[1:3], [layers[3]]]),
            G_RevAdditiveCouplingSilo(
                [
                    layers[4:5],
                    layers[5:7],
                    layers[7:]
                ]
            ),
        ])

        self.base_test(x, model)

    def test230(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(10)]

        model = RevSequential([
            RevLimAdditiveCouplingSilo(layers[0]),
            RevAdditiveCouplingSilo([layers[1:3], [layers[3]]]),
            G_RevAdditiveCouplingSilo(
                [
                    [layers[4]],
                    [None, layers[6]],
                    [None] + layers[8:]
                ]
            ),
        ])

        self.base_test(x, model)

    def test240(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(10)]

        recomp_op_layers = [nn.Conv2d(
            in_channels=o_c, out_channels=3, kernel_size=k, padding=1
        ) for _ in range(4)]

        model = RevSequential(
            rev_ops = [
                RevLimAdditiveCouplingSilo(layers[0]),
                RevAdditiveCouplingSilo([layers[1:3], [layers[3]]]),
                G_RevAdditiveCouplingSilo(
                    [
                        [layers[4]],
                        [None, layers[6]],
                        [None] + layers[8:]
                    ]
                ),
            ],
            recomp_op=RecomputeSilo(recomp_op_layers),
            )

        self.base_test(x, model, n=3)

    def test250(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(10)]

        recomp_op_layers = [nn.Conv2d(
            in_channels=o_c, out_channels=3, kernel_size=k, padding=1
        ) for _ in range(4)]

        model = RevSequential(
            rev_ops = [
                RevLimAdditiveCouplingSilo(layers[0]),
                RevAdditiveCouplingSilo([layers[1:3], [layers[3]]]),
                G_RevAdditiveCouplingSilo(
                    [
                        [layers[4]],
                        [None, nn.Sequential(layers[6], nn.Dropout(.1))],
                        [None] + layers[8:]
                    ]
                ),
            ],
            recomp_op=RecomputeSilo(recomp_op_layers),
            )

        self.base_test(x, model, n=3)

    def test300(self):
        b, i_c, i_h, i_w = 2, 4, 6, 4
        o_c, k = 4, 3
        x = [torch.randn(b, i_c, i_h, i_w)]

        layers = [nn.Conv2d(
            in_channels=i_c, out_channels=o_c, kernel_size=k, padding=1
        ) for _ in range(10)]

        recomp_op_layers = [nn.Conv2d(
            in_channels=o_c, out_channels=3, kernel_size=k, padding=1
        ) for _ in range(4)]

        model = RevSequential(
            rev_ops = [
                RevLimAdditiveCouplingSilo(layers[0]),
                RevAdditiveCouplingSilo([layers[1:3], [layers[3]]]),
                G_RevAdditiveCouplingSilo(
                    [
                        [layers[4]],
                        [None, nn.Sequential(layers[6], nn.Dropout(.1))],
                        [None] + layers[8:]
                    ]
                ),
            ],
            recomp_op=RecomputeSilo(recomp_op_layers),
            )

        self.base_test(x, model, n=3, amp=True)


if __name__ == "__main__":
    unittest.main()
