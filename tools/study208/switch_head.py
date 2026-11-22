"""Shared single-track cut head used by study208.

Four actual crops are ordered [before2,before1,after1,after2]. A switch requires
confirmed DIFFERENT bee identities across the cut; normal turning is negative,
missing/ambiguous identity is UNKNOWN(-1). A duplicate that truly changes bee is
still positive: mechanism is a reporting stratum, never a target override.

Three registered-parameter-count-matched arms share geometry/candidate cuts:
geometry; geometry+raw_cross; geometry+relative_cross. Geometry must be frozen
and reversal-invariant (motion_features is provided). All arms retain the SAME
CNN and MLP parameter count; excluded appearance channels are zeroed. The
geometry encoder gets ZERO loss gradient, so effective trainable capacity and
training/compute budgets are NOT equivalent. This is an input ablation, not
evidence that appearance beats an equally trained/capable geometry model.
Training controls must use identical source splits/candidates, not different crops.
Crop spatial matching/candidate construction remain caller responsibilities.

Cut classification does not force any reconnection, reverse track or swap.
Separate link candidates may be within3 frames only, and always include NULL.
Caller must mask unreliable geometry/image pairs; all-masked/empty means NULL.
No source-benefit, calibration, real identity or field-performance claim.
"""
import torch
from torch import nn
from torch.nn import functional as F


def motion_features(centers, frames):
    """Six symmetric geometry features from centers[B,4,2], frames[B,4].

    Reversing both arrays (including the signed time direction) leaves features
    unchanged. No field-dependent normalizer: caller supplies body-scale xy.
    """
    if centers.ndim != 3 or centers.shape[1:] != (4, 2) or frames.shape != centers.shape[:2]:
        raise ValueError('expected centers[B,4,2], frames[B,4]')
    if not centers.is_floating_point() or centers.device != frames.device or not torch.isfinite(centers).all() or not torch.isfinite(frames).all():
        raise ValueError('finite floating centers and same-device finite frames required')
    dt = frames[:, 1:]-frames[:, :-1]
    if ((dt == 0).any() or ((dt > 0).any(1) & (dt < 0).any(1)).any()):
        raise ValueError('frame order must be strictly monotone')
    steps = (centers[:, 1:]-centers[:, :-1])/dt.abs()[..., None]
    speed = steps.norm(dim=-1)
    cosine = F.cosine_similarity(steps[:, :-1], steps[:, 1:], dim=-1)
    return torch.stack((speed[:, 1], (speed[:, 0]+speed[:, 2])/2,
                        (speed[:, 0]-speed[:, 2]).abs(), cosine.mean(1),
                        (centers[:, -1]-centers[:, 0]).norm(dim=-1),
                        dt[:, 1].abs()), dim=1)


class SwitchHead(nn.Module):
    """Time-reversal symmetric appearance features, one binary switch logit."""
    ARMS = ('geometry', 'raw_cross', 'relative_cross')

    def __init__(self, geometry_dim=6, channels=16, arm='relative_cross'):
        super().__init__()
        if arm not in self.ARMS or geometry_dim < 1:
            raise ValueError('invalid arm or geometry dimension')
        self.arm, self.geometry_dim = arm, geometry_dim
        # Same small grayscale architecture as the private return-query encoder.
        self.encoder = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(geometry_dim+1, 16), nn.ReLU(), nn.Linear(16, 1))

    def appearance_features(self, crops):
        if crops.ndim != 5 or crops.shape[1:3] != (4, 1) or not crops.is_floating_point() or not torch.isfinite(crops).all():
            raise ValueError('expected finite grayscale crops[B,4,1,H,W]')
        batch = crops.shape[0]
        embedding = self.encoder(crops.flatten(0, 1)).mean((-2, -1)).reshape(batch, 4, -1)
        embedding = F.normalize(embedding, dim=-1)
        left = (embedding[:, 0]-embedding[:, 1]).square().sum(-1)
        right = (embedding[:, 2]-embedding[:, 3]).square().sum(-1)
        # Average all four cross-side distances; no directed temporal slot bias.
        cross = (embedding[:, :2, None]-embedding[:, None, 2:]).square().sum(-1).mean((1, 2))
        within = (left+right)/2
        return dict(left_internal=left, right_internal=right, cross=cross,
                    internal=within, relative=cross-within)

    def forward(self, crops, geometry):
        if geometry.ndim != 2 or geometry.shape != (crops.shape[0], self.geometry_dim) or not geometry.is_floating_point() or not torch.isfinite(geometry).all():
            raise ValueError('geometry must be finite floating[B,geometry_dim]')
        if geometry.dtype != crops.dtype or geometry.device != crops.device:
            raise ValueError('geometry and crops must share dtype/device')
        features = self.appearance_features(crops)
        appearance = features['cross'] if self.arm == 'raw_cross' else features['relative']
        if self.arm == 'geometry':
            appearance = appearance*0
        logit = self.classifier(torch.cat((geometry, appearance[:, None]), dim=1)).squeeze(-1)
        return dict(logit=logit, **features)


def switch_loss(logit, identity_target):
    """Confirmed different bee=1, confirmed same=0, unknown=-1 only."""
    if identity_target.shape != logit.shape or identity_target.device != logit.device or not torch.isfinite(identity_target).all() or not ((identity_target == -1) | (identity_target == 0) | (identity_target == 1)).all():
        raise ValueError('switch target must be same-device -1/0/1 matching logits')
    known = identity_target >= 0
    if not known.any():
        return (logit*0).sum()
    return F.binary_cross_entropy_with_logits(logit[known], identity_target[known].to(logit.dtype))


def frozen_post_swap(model, crops, geometry, permutation, matched_groups, *, confirmed_different_bee):
    """Paired post-side intervention with one unchanged image head, no fitting.

    Caller pre-matches groups by motion/hive/brightness BEFORE seeing responses.
    Every row keeps its original geometry, before2/before1 and candidate cut;
    donor's after1/after2 are copied together IN ORIGINAL temporal order. Group
    membership is a caller's frozen matching declaration, not a statistical
    matching guarantee inferred here. Permutation must be a within-group bijection.
    confirmed_different_bee[B] records independent identity evidence that donor
    post bee differs from original post bee (False means not confirmed, not same).

    Both passes use the same weights and eval mode without gradients or updates.
    Return raw paired logits/deltas, not a promised meaningful change or accuracy.
    On real images swapping is an intervention, NOT a new ground-truth identity
    label: don't train on it or evaluate it against the original cut labels.
    Synthetic known identities may separately specify synthetic intervention truth.
    """
    batch = len(crops)
    if permutation.shape != (batch,) or permutation.dtype != torch.long or permutation.device != crops.device:
        raise ValueError('permutation must be same-device long[B]')
    if not torch.equal(permutation.sort().values, torch.arange(batch, device=crops.device)):
        raise ValueError('post donors must form a bijective permutation')
    if matched_groups.shape != (batch,) or matched_groups.device != crops.device or matched_groups.dtype != torch.long:
        raise ValueError('matched_groups must be caller-frozen same-device long[B]')
    if not torch.equal(matched_groups, matched_groups[permutation]):
        raise ValueError('post swap must stay within pre-matched groups')
    if confirmed_different_bee.shape != (batch,) or confirmed_different_bee.dtype != torch.bool or confirmed_different_bee.device != crops.device:
        raise ValueError('confirmed_different_bee must be independent same-device bool[B] evidence')
    swapped = crops.clone()
    swapped[:, 2:] = crops[permutation, 2:]
    assert torch.equal(swapped[:, :2], crops[:, :2])
    assert torch.equal(swapped[:, 2:], crops[permutation, 2:])
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            original = model(crops, geometry)['logit']
            intervention = model(swapped, geometry)['logit']
    finally:
        model.train(was_training)
    return dict(original_logit=original, swapped_logit=intervention,
                paired_delta=intervention-original, donor=permutation.clone(),
                matched_groups=matched_groups.clone(),
                confirmed_different_bee=confirmed_different_bee.clone())


def synthetic_post_swap_fixture():
    """Known A->A/B->B cuts with identical motion, hive and brightness strata.

    Both toy bees have the same stripe area/brightness, opposite orientation.
    Swap changes synthetic A->A into A->B and B->B into B->A without changing
    motion. Only this constructed fixture has a verified new identity target.
    """
    coordinates = torch.arange(8)
    a = ((coordinates[None, :] % 3) == 0).float().expand(8, 8)
    b = a.t().contiguous()
    crops = torch.stack((torch.stack((a, a, a, a)), torch.stack((b, b, b, b))))[:, :, None]
    centers = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [3., 0.]]]).expand(2, 4, 2)
    frames = torch.arange(4, dtype=torch.float32).expand(2, 4)
    return dict(crops=crops, geometry=motion_features(centers, frames),
                permutation=torch.tensor([1, 0]), matched_groups=torch.tensor([0, 0]),
                confirmed_different_bee=torch.tensor([True, True]),
                original_identity_target=torch.tensor([0, 0]),
                synthetic_only_intervention_target=torch.tensor([1, 1]))


class OptionalLinkHead(nn.Module):
    """Independent endpoint-pair scorer, not a switch/swap assignment head.

    pair_features[B,K,F] supplied by caller, same meaning for every candidate.
    elapsed_frames[B,K] must be1..3 to be eligible. reliable[B,K] explicitly
    confirms usable geometry/appearance, not ground-truth identity correctness.
    Last returned class is NULL; argmax may always choose it. No forced linkage.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(feature_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.null = nn.Parameter(torch.zeros(()))

    def forward(self, pair_features, elapsed_frames, reliable):
        if pair_features.ndim != 3 or not pair_features.is_floating_point() or not torch.isfinite(pair_features).all():
            raise ValueError('finite pair_features[B,K,F] required')
        if elapsed_frames.shape != pair_features.shape[:2] or reliable.shape != elapsed_frames.shape or reliable.dtype != torch.bool:
            raise ValueError('elapsed_frames and boolean reliable must have shape[B,K]')
        if elapsed_frames.device != pair_features.device or reliable.device != pair_features.device or not torch.isfinite(elapsed_frames).all():
            raise ValueError('pair inputs must be same-device and finite')
        eligible = reliable & (elapsed_frames >= 1) & (elapsed_frames <= 3) & (elapsed_frames == elapsed_frames.round())
        logits = self.score(pair_features).squeeze(-1).masked_fill(~eligible, -torch.inf)
        return torch.cat((logits, self.null.expand(len(pair_features), 1)), dim=1)


def link_loss(logits, target):
    """target in0..K-1=confirmed successor, K=confirmed no link, -1=unknown."""
    if target.shape != logits.shape[:1] or target.dtype != torch.long or target.device != logits.device or ((target < -1) | (target >= logits.shape[1])).any():
        raise ValueError('link target must be same-device long[B] in [-1,K]')
    known = target >= 0
    if not known.any():
        return (logits.masked_fill(~torch.isfinite(logits), 0)*0).sum()
    if not torch.isfinite(logits[known, target[known]]).all():
        raise ValueError('confirmed link is outside eligible candidate support')
    return F.cross_entropy(logits[known], target[known])


def synthetic_examples():
    """Deferred source fixture, not evidence of learned performance.

    Four metadata cases expose geometry shortcuts; pair rows are illustrative
    (distance, appearance disagreement, temporal continuity), not field features.
    Unknown uses duplicate mechanism too, to show missing labels stay excluded.
    """
    coords = torch.arange(8, dtype=torch.float32)
    a = ((coords[None, :] % 3) == 0).float().expand(8, 8)
    b = a.t().contiguous()
    crops = torch.stack((torch.stack((a, a, b, b)),
                         torch.stack((a, a, a, a)),
                         torch.stack((a, a, a, a)),
                         torch.stack((a, a, b, b)),
                         torch.stack((a, a, b, b))))[:, :, None]
    centers = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [3., 0.]],
                            [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
                            [[0., 0.], [1., 0.], [2., 0.], [3., 0.]],
                            [[0., 0.], [1., 0.], [2., 0.], [3., 0.]],
                            [[0., 0.], [1., 0.], [2., 0.], [3., 0.]]])
    frames = torch.arange(4, dtype=torch.float32).expand(5, 4)
    return dict(crops=crops, centers=centers, frames=frames,
                identity_target=torch.tensor([1, 0, 0, 1, -1]),
                mechanism=['wrong_trajectory_smoother_than_truth', 'same_bee_large_turn',
                           'track_ends_no_successor', 'duplicate_with_confirmed_bee_change', 'duplicate_identity_unknown'],
                pair_features=torch.tensor([[[.1, .8, 1.], [.4, .1, 1.]],
                                            [[.2, .1, 1.], [.3, .7, 1.]],
                                            [[0., 0., 0.], [0., 0., 0.]],
                                            [[.1, .8, 1.], [.2, .1, 1.]],
                                            [[.1, .2, 1.], [.2, .3, 1.]]]),
                elapsed_frames=torch.tensor([[1., 2.], [1., 2.], [0., 4.], [1., 3.], [1., 2.]]),
                reliable=torch.tensor([[True, True], [True, True], [False, False], [True, True], [False, False]]),
                link_target=torch.tensor([1, 0, 2, 1, -1]))


def deferred_selfcheck():
    """Optional future single-CPU assertions; NOT run while writing source."""
    torch.manual_seed(7)
    batch = synthetic_examples()
    geometry = motion_features(batch['centers'], batch['frames'])
    torch.testing.assert_close(geometry, motion_features(batch['centers'].flip(1), batch['frames'].flip(1)))
    counts = []
    for arm in SwitchHead.ARMS:
        model = SwitchHead(arm=arm)
        counts.append(sum(p.numel() for p in model.parameters()))
        output = model(batch['crops'], geometry)
        reverse = model(batch['crops'].flip(1), geometry)
        torch.testing.assert_close(output['logit'], reverse['logit'])
        torch.testing.assert_close(output['left_internal'], reverse['right_internal'])
        switch_loss(output['logit'], batch['identity_target']).backward()
        model.zero_grad(set_to_none=True)
        switch_loss(model(batch['crops'], geometry)['logit'], torch.full((5,), -1)).backward()
        assert all(p.grad is not None and torch.equal(p.grad, torch.zeros_like(p.grad)) for p in model.parameters())
    assert len(set(counts)) == 1
    # Equal REGISTERED counts above do not imply equal effective capacity: the
    # geometry arm blocks its encoder loss gradient by design.
    swap_case = synthetic_post_swap_fixture()
    control = frozen_post_swap(SwitchHead(), swap_case['crops'], swap_case['geometry'],
                               swap_case['permutation'], swap_case['matched_groups'],
                               confirmed_different_bee=swap_case['confirmed_different_bee'])
    assert control['paired_delta'].shape == (2,) and torch.isfinite(control['paired_delta']).all()
    link = OptionalLinkHead(3)
    logits = link(batch['pair_features'], batch['elapsed_frames'], batch['reliable'])
    assert logits[2].argmax() == 2
    assert torch.isfinite(link_loss(logits, batch['link_target']))
    empty = link(torch.empty(2, 0, 3), torch.empty(2, 0), torch.empty(2, 0, dtype=torch.bool))
    assert empty.shape == (2, 1) and torch.equal(empty.argmax(1), torch.zeros(2, dtype=torch.long))
    print('Synthetic interface assertions passed; no training or field-benefit claim.')


if __name__ == '__main__':
    deferred_selfcheck()
