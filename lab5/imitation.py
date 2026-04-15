import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _load_demo_tensors(X, y):
    """Flatten vector-env batches from demos2.pkl into (N, obs_dim) and (N, act_dim)."""
    obs_rows = []
    act_rows = []
    for obs, act in zip(X, y):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        obs_rows.append(o)
        if torch.is_tensor(act):
            a = act.detach().cpu().float().reshape(-1)
        else:
            a = torch.as_tensor(np.asarray(act).reshape(-1), dtype=torch.float32)
        act_rows.append(a)
    X_t = torch.from_numpy(np.stack(obs_rows, axis=0))
    y_t = torch.stack(act_rows, dim=0)
    return X_t, y_t


class ImitationPolicy(nn.Module):
    """Same MLP as PPO actor default: 64-64 ReLU, linear output (no activation)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        out = self.net(x)
        return out.squeeze(0) if squeeze else out


def imitate(lr=0.001, epochs=500, batch_size=256, demos_path=None):
    if demos_path is None:
        demos_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos2.pkl")
    with open(demos_path, "rb") as file:
        X, y = pickle.load(file)

    X_t, y_t = _load_demo_tensors(X, y)
    obs_dim, act_dim = X_t.shape[1], y_t.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ImitationPolicy(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in tqdm(range(epochs), desc="imitation"):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    policy.cpu()
    policy.eval()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imitation_policy.pt")
    # Save state_dict only so eval_il can load without __main__.ImitationPolicy pickle issues.
    torch.save(
        {"state_dict": policy.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim},
        out_path,
    )


if __name__ == "__main__":
    imitate()
