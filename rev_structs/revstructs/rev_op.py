"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    get_device_states,
    set_device_states,
)


class OpState:
    def __init__(self, preserve_rng_state=True, preserve_op_state=True):
        self.preserve_op_state = preserve_op_state
        self.preserve_rng_state = preserve_rng_state

        self.had_autocast_in_fwd = {}

        self.op_state = {}
        self.fwd_cpu_state = {}
        self.had_cuda_in_fwd = {}
        self.fwd_gpu_devices = {}
        self.fwd_gpu_states = {}


class RevOp(nn.Module):
    r"""Base class for all reversible operations.

    All reversible ops should subclass this class.

    Reversible operations can be stacked into a module list to create a
    RevSequential neural network.
    """
    def __init__(
        self, disable_rev=False, preserve_rng_state=True, preserve_op_state=True,
    ):
        super().__init__()
        self._rev = not disable_rev
        self.xforms = None
        self.num_inputs = None

        self.op_state = OpState(
            preserve_rng_state=preserve_rng_state,
            preserve_op_state=preserve_op_state
        )

    def extra_repr(self):
        return f"rev={self._rev}"

    def r_state_dict(self, op):
        # save state that was used in fwd pass for use in bwd pass
        assert self._rev, "should only be used in _rev mode"
        state = {}
        for m in op.modules():
            if hasattr(m, "recompute_state_dict"):
                state[m] = m.recompute_state_dict()
        self.op_state.op_state[op] = state

    def load_r_state_dict(self, op):
        # load state that was used in fwd pass for use in bwd pass
        assert self._rev, "should only be used in _rev mode"
        for m in op.modules():
            if hasattr(m, "load_recompute_state_dict"):
                m.load_recompute_state_dict(self.op_state.op_state[op][m])

    def launch_op_fwd(self, op, *args):
        if self._rev:
            # setting rng state and amp autocast from pytorch checkpoint
            # https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html#checkpoint
            self.op_state.had_autocast_in_fwd[op] = torch.is_autocast_enabled()
            if self.op_state.preserve_rng_state:
                self.op_state.fwd_cpu_state[op] = torch.get_rng_state()
                self.op_state.had_cuda_in_fwd[op] = False
                if torch.cuda._initialized:
                    self.op_state.had_cuda_in_fwd[op] = True
                    fwd_gpu_devices, fwd_gpu_states = get_device_states(*args)
                    self.op_state.fwd_gpu_devices[op] = fwd_gpu_devices
                    self.op_state.fwd_gpu_states[op] = fwd_gpu_states

            if self.op_state.preserve_op_state:
                self.r_state_dict(op)

            y = op(*args)

            if self.op_state.preserve_op_state:
                self.load_r_state_dict(op)
        else:
            return op(*args)
        return y

    def launch_op_bwd(self, op, *args):
        # setting rng state and amp autocast from pytorch checkpoint
        # https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html#checkpoint
        rng_devices = []
        if self.op_state.preserve_rng_state and self.op_state.had_cuda_in_fwd[op]:
            rng_devices = self.op_state.fwd_gpu_devices[op]
        with torch.random.fork_rng(devices=rng_devices, enabled=self.op_state.preserve_rng_state):
            if self.op_state.preserve_rng_state:
                torch.set_rng_state(self.op_state.fwd_cpu_state[op])
                if self.op_state.had_cuda_in_fwd[op]:
                    set_device_states(self.op_state.fwd_gpu_devices[op], self.op_state.fwd_gpu_states[op])
            with torch.enable_grad(), torch.cuda.amp.autocast(self.op_state.had_autocast_in_fwd[op]):
                if hasattr(op, "backward_pass"):
                    return op.backward_pass(*args)
                return op(*args)

    def forward(self, x):
        r"""Defines the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward_pass(self, y, dy):
        r"""Defines the computation performed at every backward call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
