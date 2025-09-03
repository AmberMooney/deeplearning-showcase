# attention_heatmap.py
# Visualizes multi-head self-attention on a toy token sequence.
# Saves:
#   - outputs/attention_heatmap.png  (head 0, labeled axes)
#   - outputs/attention_heads.png    (all heads grid)
#
# Usage:
#   pip install torch matplotlib
#   python attention_heatmap.py --seq-len 16 --heads 4 --d-model 32 --causal

import argparse, os, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def make_tokens(seq_len: int) -> torch.Tensor:
    """
    Make a simple repeating pattern of token IDs, shape (S, 1).
    """
    base = [0,1,2,3,2,1,0,4,5,4]
    toks = (base * ((seq_len + len(base) - 1)//len(base)))[:seq_len]
    return torch.tensor(toks, dtype=torch.long).unsqueeze(1)  # (S, N=1)

def causal_mask(S: int) -> torch.Tensor:
    """
    Upper-triangular mask (no peeking ahead). Zeros on/below diag, -inf above.
    PyTorch expects additive mask where masked positions are -inf.
    """
    mask = torch.triu(torch.ones(S, S), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask

def attention_entropy(A: torch.Tensor) -> torch.Tensor:
    """
    Row-wise entropy for attention matrix A (S, S), softmax probs per row.
    """
    eps = 1e-12
    return -(A * (A + eps).log()).sum(dim=-1)

def plot_single_head(A_np: np.ndarray, tok_ids, outpath: str, title: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=150)
    im = plt.imshow(A_np, aspect='auto', interpolation='nearest')
    plt.title(title)
    plt.xlabel("Keys (token id)")
    plt.ylabel("Queries (token id)")
    plt.xticks(range(len(tok_ids)), tok_ids, rotation=90)
    plt.yticks(range(len(tok_ids)), tok_ids)
    cbar = plt.colorbar(im)
    cbar.set_label("Attention weight")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

def plot_all_heads(weights: torch.Tensor, outpath: str):
    """
    weights: (H, 1, S, S) tensor of attention weights (probabilities).
    """
    H = weights.shape[0]
    S = weights.shape[2]
    cols = min(4, H)
    rows = math.ceil(H / cols)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # Common scale for comparability
    vmin, vmax = 0.0, float(weights.max().item())

    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.0*rows), dpi=150)
    axes = np.array(axes).reshape(-1) if H > 1 else [axes]

    last_im = None
    for h in range(H):
        A = weights[h, 0].detach().cpu().numpy()  # (S, S)
        ax = axes[h]
        im = ax.imshow(A, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
        ax.set_title(f"Head {h+1}")
        ax.set_xticks([]); ax.set_yticks([])
        last_im = im

    # Hide extra axes if any
    for k in range(H, len(axes)):
        axes[k].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:H].tolist(), shrink=0.7, label="Attention weight")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=16, help="Sequence length S")
    p.add_argument("--vocab", type=int, default=6, help="Toy vocab size")
    p.add_argument("--d-model", type=int, default=32, help="Embedding / model dim")
    p.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--causal", action="store_true", help="Use causal (decoder-style) mask")
    p.add_argument("--head", type=int, default=0, help="Which head to show in single-head figure")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Tokens & embeddings
    tokens = make_tokens(args.seq_len)             # (S, 1)
    emb = nn.Embedding(args.vocab, args.d_model)

    # Multihead Self-Attention
    attn = nn.MultiheadAttention(embed_dim=args.d_model, num_heads=args.heads, batch_first=False)
    x = emb(tokens).float()                        # (S, 1, D)

    mask = causal_mask(args.seq_len) if args.causal else None
    out, weights = attn(x, x, x,
                        need_weights=True,
                        average_attn_weights=False,
                        attn_mask=mask)
    # weights: (H, N=1, S, S)

    # --- Metrics on head 0 (INSERTS 3 & 4) ---
    h = max(0, min(args.head, args.heads - 1))
    head0 = weights[h, 0]                          # (S, S) torch tensor
    row_H = attention_entropy(head0)               # (S,)
    self_ratio = torch.diag(head0).mean().item()

    print(f"\n[Head {h}] Per-token attention entropy:")
    print([round(v, 4) for v in row_H.tolist()])
    print(f"[Head {h}] Mean self-attention (diagonal avg): {self_ratio:.4f}")
    print(f"Masking: {'causal' if args.causal else 'bidirectional'}")

    # --- Save single-head labeled heatmap (head h) ---
    tok_ids = tokens.squeeze(1).tolist()
    plot_single_head(
        head0.detach().cpu().numpy(),
        tok_ids,
        outpath="outputs/attention_heatmap.png",
        title=f"Attention Heatmap (Head {h+1}) — {'Causal' if args.causal else 'Bidirectional'}"
    )

    # --- Save all-heads grid ---
    plot_all_heads(weights, outpath="outputs/attention_heads.png")

if __name__ == "__main__":
    main()
