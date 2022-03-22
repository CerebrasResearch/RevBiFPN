"""
Released under BSD 3-Clause License,
Copyright (c) 2022 Cerebras Systems Inc.
All rights reserved.
"""
import warnings
import torch
import torch.nn as nn

from .rev_op import RevOp
from .recomp_op import RecomputeSilo


class RevContainer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        The fwd call of any inheriting RevContainer should be implemented using
        a torch.autograd.function.Function which allows it to reverse backwards
        through the container.

        Note RevSequential's custom autograd is implemented in RevProp
        """
        raise NotImplementedError


class RevProp(torch.autograd.function.Function):
    """
    Integrates the reversible sequence into the autograd framework
    """

    @staticmethod
    def forward(ctx, rev_ops, recomp_op, *x):
        # Note: recomp_op can be None
        for op in rev_ops:
            x = op(x) if isinstance(x, torch.Tensor) else op(*x)

        ctx.y = [_x.detach() for _x in x]
        ctx.rev_ops = rev_ops
        ctx.recomp_op = recomp_op

        if recomp_op:
            if isinstance(x, torch.Tensor):
                return recomp_op(x)
            return recomp_op(*x)
        return x

    @staticmethod
    def backward(ctx, *dy):
        y = ctx.y
        del ctx.y

        if ctx.recomp_op:
            y, dy = ctx.recomp_op.backward_pass(x=y, dy=dy)
        del ctx.recomp_op

        for idx in range(len(ctx.rev_ops)):
            y, dy = ctx.rev_ops[-1 - idx].backward_pass(y, dy)
        del ctx.rev_ops
    
        if isinstance(dy, torch.Tensor):
            return None, None, dy
        else:
            return (None, None, ) + tuple(dy)



class RevSequential(nn.ModuleList, RevContainer):
    r"""A reversible sequential container.
    A subclass of ModulesList which defines the order of the sequential model.

    Only the last output activation is stored, ie. intermediate activations are
    NOT stored but recomputed using the reversible nature of the reversible
    operations (operations derived from RevOp).

    The auto-grad function for this container is defined in RevProp.

    Args:
        rev_ops (torch.nn.ModuleList): list of reversible operations.
            Order of list defines the sequential nature of the container.

    To make it easier to understand, here is a small example::

        # Example of using RevSequential
        model = RevSequential([
                  RevLimResidualSilo(nn.Conv(args), nn.Conv(args)),
                  RevLimResidualSilo(nn.Conv(args), nn.Conv(args)),
                ])

        model.append(RevLimResidualSilo(nn.Conv(args), nn.Conv(args)))

        model.extend([
                  RevLimResidualSilo(nn.Conv(args), nn.Conv(args)),
                  RevLimResidualSilo(nn.Conv(args), nn.Conv(args)),
                ])

        model.insert(3, RevLimResidualSilo(nn.Conv(args), nn.Conv(args)))
    """
    def __init__(self, rev_ops=[], recomp_op=None, disable_rev=False, disable_recomp=False):
        self._rev = not disable_rev
        self._recomp = not disable_recomp
        self.rev_ops = []  # added in super().__init__ using extend
        if rev_ops:
            for block in rev_ops:
                assert (isinstance(block, RevOp)), "needs to be a rev_op"
        if recomp_op:
            assert isinstance(recomp_op, RecomputeSilo)
        super().__init__(modules=rev_ops)
        self.recomp_op = recomp_op

    def extra_repr(self):
        return f'rev={self._rev}'

    def insert(self, index, module):
        r"""Extends ModuleList's insert method.
        Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        assert (isinstance(module, RevOp)), "needs to be a rev_op"
        self.rev_ops.insert(index, module)
        super(RevSequential, self).insert(index, module)

    def append(self, module):
        r"""Extends ModuleList's append method.
        Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        assert (isinstance(module, RevOp)), "needs to be a rev_op"
        self.rev_ops.append(module)
        return super(RevSequential, self).append(module)

    def extend(self, modules):
        r"""Extends ModuleList's extend method.
        Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        for module in modules:
            assert (isinstance(module, RevOp)), "needs to be a rev_op"
        self.rev_ops.extend(modules)
        return super(RevSequential, self).extend(modules)

    def recomp_op_fwd(self, x):
        warnings.warn(f"recomp_op without recomp")
        if self.recomp_op:
            if isinstance(x, torch.Tensor):
                return self.recomp_op(x)
            return self.recomp_op(*x)
        return x

    def fwd(self, x):
        for op in self.rev_ops:
            x = op(x) if isinstance(x, torch.Tensor) else op(*x)
        return self.recomp_op_fwd(x)

    def forward(self, x):
        if self._rev:
            x = [x] if isinstance(x, torch.Tensor) else x
            if self._recomp:
                return RevProp.apply(self.rev_ops, self.recomp_op, *x)
            x = RevProp.apply(self.rev_ops, None, *x)
            return self.recomp_op_fwd(x)

        else:
            return self.fwd(x)
