"""First identity-change supervision after a reliable same-bee reset anchor.

This module accepts constraints, never field data or hive IDs. A caller must
certify masks from SOURCE labels only. It does not infer a negative from equal
endpoint IDs: equal endpoints with an unknown middle supply no survival mask.

One row contains logits for consecutive native gaps. Each h_i is the hazard
of the FIRST change, conditional on survival to gap i; independence of the
underlying observations is not required. A reliable different-original-ID
bracket establishes at least one change, NOT its first exact date. An adjacent
different-ID bracket has exactly one gap and is precise supervision.

survival_mask: a continuous, individually confirmed same-bee segment.
event_mask: one continuous different-ID bracket. Optional survival constraints
must immediately precede it; they certify a same-bee prefix. Unconstrained
gaps have zero gradient. Unknown gaps INSIDE a positive bracket receive only
the weak interval gradient, not invented individual 0/1 labels.

For a prefix P and positive interval I the loss is
    -log S(P) - log(1-S(I)),  S(A)=product_{i in A}(1-h_i).
The native-order first-event logsumexp computes log(1-S(I)) stably, including
very negative logits where computing 1-product or log(-expm1(log S)) loses the
event probability to underflow. No probability clipping or sorting is used.

Self-test is opt-in and uses only tiny synthetic tensors. It has not been
invoked by writing/importing this file. No changes to the registered head.
"""
import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F


def _mask(value, logits, name):
    if value is None:
        return torch.zeros_like(logits, dtype=torch.bool)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError(name + ' must be a bool tensor')
    if value.shape != logits.shape or value.device != logits.device:
        raise ValueError(name + ' must have exactly the logits shape/device')
    return value


def _bounds(mask):
    positions = torch.arange(mask.shape[-1], device=mask.device)
    first = positions.expand_as(mask).masked_fill(~mask, mask.shape[-1]).amin(-1)
    last = positions.expand_as(mask).masked_fill(~mask, -1).amax(-1)
    count = mask.sum(-1)
    if bool(((count > 0) & (last-first+1 != count)).any()):
        raise ValueError('each constraint must be contiguous; split separate runs into separate rows')
    return first, last, count


def _log_at_least_one(logits, event_mask):
    """Native-order sum of disjoint first-event probabilities, in log space."""
    z = logits.masked_fill(~event_mask, 0.)
    log_h = F.logsigmoid(z)
    log_q = F.logsigmoid(-z).masked_fill(~event_mask, 0.)
    prefix = torch.cat((torch.zeros_like(log_q[..., :1]), log_q.cumsum(-1)[..., :-1]), -1)
    first_event = (log_h+prefix).masked_fill(~event_mask, -torch.inf)
    active = event_mask.any(-1)
    # An inactive row gets a constant dummy log-probability 0. This avoids
    # logsumexp(all -inf), whose backward is undefined even under a zero mask.
    first_position = torch.arange(logits.shape[-1], device=logits.device) == 0
    first_event = torch.where((~active).unsqueeze(-1) & first_position,
                              torch.zeros_like(first_event), first_event)
    return torch.logsumexp(first_event, -1)


def switch_interval_loss(logits, survival_mask=None, event_mask=None, reduction='mean'):
    """Return per-row loss, sum, or mean over rows with any known constraint.

    logits may have any leading batch dimensions and a nonempty final gap
    dimension. Supported dtypes are float32/float64. All logits must be finite.
    An all-unknown batch returns a differentiable zero, with zero gradients.
    At most one positive bracket is allowed per row: after its first change,
    further identity changes require a new reset/query, not this first hazard.
    """
    if not isinstance(logits, torch.Tensor) or logits.dtype not in (torch.float32, torch.float64):
        raise TypeError('logits must be a float32/float64 tensor')
    if logits.ndim < 1 or logits.shape[-1] == 0 or not bool(torch.isfinite(logits).all()):
        raise ValueError('finite logits and at least one gap required')
    if reduction not in ('none', 'sum', 'mean'):
        raise ValueError('reduction must be none/sum/mean')
    survival = _mask(survival_mask, logits, 'survival_mask')
    event = _mask(event_mask, logits, 'event_mask')
    _, survival_last, nsurvival = _bounds(survival)
    event_first, _, nevent = _bounds(event)
    if bool((survival & event).any()):
        raise ValueError('survival prefix and positive interval must be disjoint')
    if bool(((nsurvival > 0) & (nevent > 0) & (survival_last+1 != event_first)).any()):
        raise ValueError('a certified prefix must end immediately before its positive bracket')
    # Mask before nonlinear operations: unconstrained values cannot affect loss.
    z = logits.masked_fill(~survival, 0.)
    survival_loss = F.softplus(z).masked_fill(~survival, 0.).sum(-1)
    loss = survival_loss-_log_at_least_one(logits, event)
    if reduction == 'none':
        return loss
    total = loss.sum()
    if reduction == 'sum':
        return total
    active = ((nsurvival+nevent) > 0).sum().clamp_min(1)
    return total/active


class SwitchIntervalLoss(nn.Module):
    """Small nn.Module wrapper; it owns no model or parameters."""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, survival_mask=None, event_mask=None):
        return switch_interval_loss(logits, survival_mask, event_mask, self.reduction)


def demo_first_cut_map(logits):
    """Illustrative native-order MAP first boundary, or -1 for no change.

    This is a demo, not a replacement for any registered decoder. Exact ties
    favor the earliest boundary; no-change is the final alternative.
    """
    log_h = F.logsigmoid(logits)
    log_q = F.logsigmoid(-logits)
    prefix = torch.cat((torch.zeros_like(log_q[..., :1]), log_q.cumsum(-1)[..., :-1]), -1)
    all_choices = torch.cat((log_h+prefix, log_q.sum(-1, keepdim=True)), -1)
    choice = all_choices.argmax(-1)
    return torch.where(choice == logits.shape[-1], torch.full_like(choice, -1), choice)


def _toy_event_windows(first_cut):
    """A concrete retained native prefix: only keeping date 10 gives entry."""
    dates = (6, 7, 8, 9, 10, 11)
    centres = (-3., -2., -1., -.5, .5, 1.)
    kept = [(f, x) for f, x in zip(dates, centres) if f <= 9+first_cut]
    entries = [f for (_, previous), (f, current) in zip(kept, kept[1:])
               if previous < 0 <= current]
    return tuple(sum(lo <= f < hi for f in entries) for lo, hi in ((0, 10), (10, 20)))


def self_test():
    """Deferred tiny CPU checks; run only when the owner releases CPU."""
    torch.set_num_threads(1)
    dtype = torch.float64
    all_gaps = torch.ones(2, dtype=torch.bool)
    p = torch.tensor([.2, .3], dtype=dtype)
    z = torch.logit(p).requires_grad_()
    positive = switch_interval_loss(z, event_mask=all_gaps)
    assert torch.allclose(positive, torch.tensor(-math.log(.44), dtype=dtype), atol=1e-14, rtol=1e-14)
    gradient = torch.autograd.grad(positive, z, retain_graph=True)[0]
    assert torch.allclose(gradient, -p*(.56/.44), atol=1e-14, rtol=1e-14)
    negative = switch_interval_loss(z, survival_mask=all_gaps)
    assert torch.allclose(negative, torch.tensor(-math.log(.8*.7), dtype=dtype), atol=1e-14, rtol=1e-14)
    adjacent = switch_interval_loss(z[:1], event_mask=all_gaps[:1])
    assert torch.allclose(adjacent, F.softplus(-z[0]), atol=1e-14, rtol=1e-14)
    combined_z = torch.logit(torch.tensor([.1, .2, .3, .9], dtype=dtype)).requires_grad_()
    prefix = torch.tensor([True, False, False, False])
    interval = torch.tensor([False, True, True, False])
    combined = switch_interval_loss(combined_z, prefix, interval)
    assert torch.allclose(combined, torch.tensor(-math.log(.9)-math.log(.44), dtype=dtype), atol=1e-14, rtol=1e-14)
    combined.backward()
    assert torch.isfinite(combined_z.grad).all() and combined_z.grad[-1] == 0
    assert torch.allclose(combined_z.grad, torch.tensor([.1, -.2*(.56/.44), -.3*(.56/.44), 0.], dtype=dtype), atol=1e-14, rtol=1e-14)
    # Same endpoints, unknown middle: no constraint, no negative supervision.
    unknown = torch.tensor([[-1e4, 1e4, -2.], [4., -8., 2.]], dtype=dtype, requires_grad=True)
    zero = switch_interval_loss(unknown)
    zero.backward()
    assert zero.item() == 0 and torch.equal(unknown.grad, torch.zeros_like(unknown))
    for test_dtype in (torch.float32, torch.float64):
        for values in ((-1e4, -1e4), (1e4, 1e4), (-1e4, 1e4), (1e4, -1e4)):
            for kind in ('event', 'survival'):
                x = torch.tensor(values, dtype=test_dtype, requires_grad=True)
                kwargs = {'event_mask' if kind == 'event' else 'survival_mask': all_gaps}
                value = switch_interval_loss(x, **kwargs)
                value.backward()
                assert torch.isfinite(value) and torch.isfinite(x.grad).all()
    # Group reductions do not dilute supervision by adding unknown examples.
    batch = torch.stack((z.detach(), z.detach())).requires_grad_()
    event = torch.tensor([[True, True], [False, False]])
    losses = switch_interval_loss(batch, event_mask=event, reduction='none')
    assert losses[1] == 0 and torch.allclose(switch_interval_loss(batch, event_mask=event), losses[0])
    for survival, event in (([True, False, True], [False]*3),
                            ([True, False, False], [False, False, True]),
                            ([True, False, False], [True, False, False])):
        try:
            switch_interval_loss(torch.zeros(3, dtype=dtype), torch.tensor(survival), torch.tensor(event))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid/gapped/overlapping constraints accepted')
    # Identical positive bag hit and interval likelihood, different timing.
    a = torch.logit(torch.tensor([.8, .2], dtype=dtype))
    b = a.flip(0)
    la = switch_interval_loss(a, event_mask=all_gaps)
    lb = switch_interval_loss(b, event_mask=all_gaps)
    assert torch.allclose(la, lb, atol=1e-14, rtol=1e-14)
    assert bool((a.sigmoid() >= .5).any()) == bool((b.sigmoid() >= .5).any())
    ca, cb = int(demo_first_cut_map(a)), int(demo_first_cut_map(b))
    assert (ca, cb) == (0, 1)
    wa, wb = _toy_event_windows(ca), _toy_event_windows(cb)
    assert wa == (0, 0) and wb == (0, 1)
    return dict(status='PASS', synthetic_only=True,
                tests='hand values; adjacent exact; prefix+interval; extremes float32/64; finite gradients; unknown zero; reduction; mask validation',
                warning='bag hit and interval likelihood do not identify the first boundary or event windows',
                permutation_demo=dict(losses=[float(la), float(lb)], first_cuts=[ca, cb], event_windows=[wa, wb]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(self_test())
