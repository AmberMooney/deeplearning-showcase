# landscapes/loss_landscape_gif.py
# Make a GIF of a fixed 2D loss landscape with a moving dot showing training trajectory.
# Saves: outputs/loss_landscape.gif
#
# Usage:
#   pip install torch scikit-learn matplotlib imageio
#   python landscapes/loss_landscape_gif.py --epochs 500 --grid-res 41 --fps 2.0 --auto-span

import os, copy, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import imageio
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as mlp

# --------- Utils ---------
def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def set_params_from_flat(model: nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[idx: idx + n].view_as(p))
            idx += n

def normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / (vec.norm() + 1e-12)

def gram_schmidt_2(d1: torch.Tensor, d2: torch.Tensor):
    d1 = normalize(d1)
    d2 = d2 - torch.dot(d2, d1) * d1
    d2 = normalize(d2)
    return d1, d2

def loss_on(model, X, y):
    with torch.no_grad():
        return nn.CrossEntropyLoss()(model(X), y).item()

# --------- Model ---------
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--grid-res", type=int, default=41, help="Landscape resolution per axis")
    ap.add_argument("--span", type=float, default=1.0, help=" plus minus span along each direction (ignored if --auto-span)")
    ap.add_argument("--auto-span", action="store_true", help="Auto span from trajectory range")
    ap.add_argument("--fps", type=float, default=2.0, help="GIF frames per second (lower = slower)")
    ap.add_argument("--frame-stride", type=int, default=1, help="Use every n-th epoch as a frame")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    X, y = make_moons(n_samples=args.samples, noise=args.noise, random_state=args.seed)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=args.seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)

    # Model & training
    model = MLP()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr),
                                         batch_size=args.bs, shuffle=True)

    weights_over_time = []
    weights_over_time.append(flatten_params(model).clone())
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        weights_over_time.append(flatten_params(model).clone())

    # Reference = final weights; directions via Gram-Schmidt
    w_ref = weights_over_time[-1]
    d1 = torch.randn_like(w_ref)
    d2 = torch.randn_like(w_ref)
    d1, d2 = gram_schmidt_2(d1, d2)

    # Project trajectory onto (d1, d2) plane centered at w_ref
    traj = []
    for w in weights_over_time:
        delta = w - w_ref
        a = torch.dot(delta, d1).item()
        b = torch.dot(delta, d2).item()
        traj.append((a, b))

    # Determine span
    if args.auto_span:
        max_a = max(abs(a) for a, _ in traj)
        max_b = max(abs(b) for _, b in traj)
        span = max(1e-3, 1.4 * max(max_a, max_b))  # pad a bit
    else:
        span = args.span

    # Precompute landscape Z on the fixed plane (around w_ref)
    alphas = np.linspace(-span, span, args.grid_res)
    betas  = np.linspace(-span, span, args.grid_res)
    Agrid, Bgrid = np.meshgrid(alphas, betas)
    Z = np.zeros_like(Agrid)

    probe = copy.deepcopy(model)
    # ensure probe params align with w_ref
    set_params_from_flat(probe, w_ref.clone())

    for i in range(args.grid_res):
        for j in range(args.grid_res):
            a = float(Agrid[j, i]); b = float(Bgrid[j, i])
            w = w_ref + a * d1 + b * d2
            set_params_from_flat(probe, w)
            Z[j, i] = loss_on(probe, Xte, yte)

    # Build frames: fixed contour + moving dot
    os.makedirs("outputs", exist_ok=True)
    frames = []
    stride = max(1, args.frame_stride)
    xs = [ab[0] for ab in traj]
    ys = [ab[1] for ab in traj]

    # Draw with consistent limits
    xlim = (-span, span)
    ylim = (-span, span)

    for t in range(0, len(traj), stride):
        fig = plt.figure(figsize=(5, 4), dpi=160)
        ax = fig.add_subplot(111)
        #cs = ax.contourf(Agrid, Bgrid, Z, levels=30)
        zmin, zmax = Z.min(), Z.max()
        norm = mlp.colors.Normalize(vmin=float(zmin), vmax=float(zmax))

        cs = ax.contourf(
            Agrid, Bgrid, Z,
            levels=30,
            cmap="magma_r",          
            norm=norm
            )
        fig.colorbar(cs, ax=ax, label="Loss")
        ax.plot(xs[:t+1], ys[:t+1], '-', alpha=0.9)  # path so far
        ax.plot(xs[t], ys[t], 'o', markersize=5)     # current point
        ax.set_xlabel("alpha (dir 1)")
        ax.set_ylabel("beta (dir 2)")
        ax.set_title("Loss Landscape with Training Trajectory")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        fig.tight_layout()

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)

    # Save GIF
    gif_path = "outputs/loss_landscape.gif"
    imageio.mimsave(gif_path, frames, duration=1.0 / max(0.1, args.fps))
    print(f"Saved {gif_path}")

if __name__ == "__main__":
    main()
