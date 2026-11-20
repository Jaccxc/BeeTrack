"""CPU boundary-cell event classifier. Intervals and event frames are one-based."""
import numpy as np
import torch
from torch import nn

TAPS = np.array([-12, -6, -3, -1, 0, 1, 3, 6, 12])


def _validate(bank, start, end, arm):
    if bank.dtype != np.uint8 or bank.ndim != 4 or bank.shape[1:] != (32, 21, 21):
        raise ValueError("Expected uint8 bank (T,32,21,21)")
    if not 1 <= start <= end <= len(bank) or arm not in ("T", "S"):
        raise ValueError("Invalid inclusive interval or arm")


def _inputs(bank, frames, cells, start, end, arm):
    offsets = TAPS if arm == "T" else np.zeros(9, dtype=int)
    indices = np.clip(frames[:, None] + offsets, start - 1, end - 1)
    x = bank[indices, cells[:, None]].astype(np.float32)
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    std = x.std(axis=(1, 2, 3), keepdims=True).clip(8)
    return torch.from_numpy((x - mean) / std)


def _network():
    return nn.Sequential(
        nn.Conv2d(9, 12, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Linear(144, 2),
    )


def train_model(bank, events, start, end, arm, seed=179):
    """Fit fixed 12x16 batches; arms share seed-determined sampled coordinates."""
    _validate(bank, start, end, arm)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = np.zeros((end - start + 1, 32, 2), np.float32)
    for frame, cell, direction in events:
        if not (int(frame) == frame and int(cell) == cell and int(direction) == direction
                and 0 <= cell < 32 and direction in (0, 1)):
            raise ValueError("Invalid event (frame,cell,direction)")
        if start <= frame <= end:
            labels[int(frame) - start, int(cell), int(direction)] = 1
    positive = np.argwhere(labels.any(axis=2))
    excluded = np.zeros(labels.shape[:2], bool)
    for frame, cell in positive:
        excluded[max(0, frame - 2):min(len(labels), frame + 3),
                 [(cell - 1) % 32, cell, (cell + 1) % 32]] = True
    negative = np.argwhere(~excluded)
    if not len(positive) or not len(negative):
        raise ValueError("Training requires positive centers and truly negative cells")
    model = _network().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(12):
        for _ in range(16):
            coordinates = np.concatenate((positive[rng.integers(len(positive), size=32)],
                                          negative[rng.integers(len(negative), size=32)]))
            frame, cell = coordinates.T
            x = _inputs(bank, frame + start - 1, cell, start, end, arm)
            target = torch.from_numpy(labels[frame, cell])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
    return model


def predict_model(model, bank, start, end, arm):
    """Return natural-area probabilities without negative sampling or reweighting."""
    _validate(bank, start, end, arm)
    torch.set_num_threads(2)
    output = np.empty((end - start + 1, 32, 2), np.float32)
    model.eval()
    with torch.inference_mode():
        for first in range(start - 1, end, 8):
            stop = min(first + 8, end)
            frames = np.repeat(np.arange(first, stop), 32)
            cells = np.tile(np.arange(32), stop - first)
            values = model(_inputs(bank, frames, cells, start, end, arm)).sigmoid().numpy()
            output[first - start + 1:stop - start + 1] = values.reshape(-1, 32, 2)
    return output


def _smoketest():
    rng = np.random.default_rng(179)
    bank = rng.integers(95, 105, (24, 32, 21, 21), dtype=np.uint8)
    events = [(f, 4, 0) for f in range(5, 21)] + [(f, 16, 1) for f in range(5, 21)]
    bank[4:20, 4, 3:10, 3:10] = 240
    bank[4:20, 16, 11:18, 11:18] = 240
    model = train_model(bank, events, 3, 22, "S")
    prediction = predict_model(model, bank, 3, 22, "S")
    assert prediction.dtype == np.float32 and prediction.shape == (20, 32, 2)
    assert np.isfinite(prediction).all()
    assert prediction[2:18, 4, 0].mean() > .8
    assert prediction[2:18, 16, 1].mean() > .8
    assert prediction[:, 24].mean() < .2
    altered = bank.copy()
    altered[:2] = 0
    altered[22:] = 255
    for arm in ("T", "S"):
        assert torch.equal(_inputs(bank, np.array([2, 21]), np.array([4, 16]), 3, 22, arm),
                           _inputs(altered, np.array([2, 21]), np.array([4, 16]), 3, 22, arm))
    print("Synthetic CPU training passed: learned both directions; interval taps do not leak.")


if __name__ == "__main__":
    _smoketest()
