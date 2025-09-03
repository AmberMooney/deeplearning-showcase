import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Config ---
SEED = 42
N_SAMPLES = 600
NOISE = 0.25
DATASET = "moons"   # "moons" or "circles"
HIDDEN = 32
LR = 1e-2
EPOCHS = 20
BATCH_SIZE = 128
GRID_RES = 140
FRAMES_PER_EPOCH = 1  # increase for smoother animation

np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Data ---
if DATASET == "moons":
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
else:
    X, y = make_circles(n_samples=N_SAMPLES, noise=NOISE, factor=0.5, random_state=SEED)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.long)

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# --- Simple MLP Model ---
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=HIDDEN, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cpu")
model = MLP().to(device)
opt = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# --- Plot utilities ---
def plot_decision_boundary(ax, model, X_plot, y_plot, title=""):
    # Create grid
    x_min, x_max = X_plot[:,0].min() - 0.5, X_plot[:,0].max() + 0.5
    y_min, y_max = X_plot[:,1].min() - 0.5, X_plot[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, GRID_RES),
                         np.linspace(y_min, y_max, GRID_RES))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        logits = model(torch.tensor(grid, dtype=torch.float32).to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    Z = pred.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.35)
    ax.scatter(X_plot[:,0], X_plot[:,1], c=y_plot, s=15, alpha=0.9)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

# --- Training and frame capture ---
os.makedirs("outputs", exist_ok=True)
frames = []

def save_frame(epoch, acc_val):
    fig = plt.figure(figsize=(4,4), dpi=150)
    ax = fig.add_subplot(111)
    plot_decision_boundary(ax, model, X_val.numpy(), y_val.numpy(), title=f"Epoch {epoch} | Val Acc: {acc_val:.2f}")
    fig.canvas.draw()
    # Convert to image array
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)
    plt.close(fig)

def accuracy(x, y):
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        return (preds == y).float().mean().item()

# Initial frame
save_frame(0, accuracy(X_val, y_val))

for epoch in range(1, EPOCHS+1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()

    if epoch % 1 == 0:
        acc = accuracy(X_val, y_val)
        for _ in range(FRAMES_PER_EPOCH):
            save_frame(epoch, acc)

# Save GIF
gif_path = "outputs/decision_boundary.gif"
imageio.mimsave(gif_path, frames, duration=1000)
print(f"Saved {gif_path}")
