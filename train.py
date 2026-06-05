# train_u_to_s_l2.py
import argparse, os, random
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import utils
import math

# ------------------ utils ------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 100
    seed: int = 42
    num_workers: int = 2
    val_ratio: float = 0.1
    hidden: Tuple[int, ...] = (512, 256, 128, 64)
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "./checkpoints"
    normalize_x: bool = False
    normalize_y: bool = False
    use_mse: bool = True  # if True, use L2^2 (MSE) instead of L2
    u_path: str = "./u_vec.csv"
    s_path: str = "./s.csv"

# ------------------ io ------------------
def load_any(path):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        with np.load(path) as z: return z[list(z.keys())[0]]
    if path.endswith(".csv") or path.endswith(".txt"):
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported file type: {path}")

def load_arrays(u_path, s_path):
     U = load_any(u_path)  # (d, N) columns = samples
     S = load_any(s_path)  # (k, N) or (N,)
     U = U.reshape(-1, 1)  
     S = S.reshape(-1, 1)
 
     if S.shape[-1] != U.shape[-1]:
        raise ValueError(f"Sample count mismatch: u has {U.shape[-1]}, s has {S.shape[-1]}")
    
     return U, S

def generate_index(n, data_size, args):
    I = []
    for _ in range(data_size):
        # random control vector length n+1
        if args.t_schedule == "random":
            idx = np.random.rand(n + 1) * args.t_end   # n+1 numbers in [0,1)
            if args.t_sorted: 
                order = idx.argsort()
                idx = idx[order]
        else: idx = np.linspace(0, args.t_end, num=args.n + 1, dtype=np.float32)

        I.append(idx)
    I = np.stack(I)  # shape (N, n+1)
    return I

def arrays_to_tensors(idx):
    idx = torch.from_numpy(idx.astype("float32"))  

    # x_mean = idx.mean(0, keepdim=True) if cfg.normalize_y else torch.zeros_like(idx[:1])
    # x_std  = idx.std(0, unbiased=False, keepdim=True) if cfg.normalize_y else torch.ones_like(idx[:1])
    # x_std = torch.clamp(x_std, 1e-6)
    # xn = (idx - x_mean) / x_std if cfg.normalize_y else idx

    meta = {"in_dim": idx.shape[1], "out_dim": idx.shape[1]}
    

    return idx,  meta


def make_loaders(idx, cfg: Config):
    ds = TensorDataset(idx)
    n_val = max(1, int(len(ds) * cfg.val_ratio))
    n_train = len(ds) - n_val
    tr, va = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    tr_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=True)
    va_loader = DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)
    return tr_loader, va_loader

# ------------------ model ------------------tried CNN, PINN
class MLP(nn.Module):
     def __init__(self, in_dim, out_dim, args, hidden: Tuple[int, ...]):
         super().__init__()
         layers = []
         prev = in_dim
         self.args = args
         for h in hidden:
             layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
             prev = h
         layers += [nn.Linear(prev, out_dim)]
         self.net = nn.Sequential(*layers)
     def forward(self, x): 
         z = self.net(x)              # unbounded
         # squashed to [0, 3] or positive
         if self.args.squash == "sigmoid":
            return 3 * torch.sigmoid(z)  # squashed to [0, 3]
         elif self.args.squash == "tanh":
            return 1.5 * (torch.tanh(z) + 1)
         elif self.args.squash == "Relu":
            return torch.relu(z) 
         elif self.args.squash == "Softplus": #decreasing but slow
             return torch.nn.functional.softplus(z)
         elif self.args.squash == "exp": #increasing
             return torch.exp(z)
         elif self.args.squash == "abs": # increasing
             return z.abs()  
         elif self.args.squash == "Soft_max": 
             probs = torch.softmax(z, dim=-1)         # 
             return 3 * probs
         else:
             return z

         

def init_positive_output_linear(model: nn.Module, bias_value: float = 1.0, w_scale: float = 1e-3):
    """
    - Hidden layers (Linear) get Kaiming init (for ReLU).
    - Final Linear gets tiny weights and a positive bias so outputs start > 0.
    """
    # collect all Linear layers in order of appearance
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) >= 1, "No Linear layers found."
    last = linears[-1]

    # init hidden linears
    for m in linears[:-1]:
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)

    # init final linear: tiny weights + positive bias
    nn.init.normal_(last.weight, mean=0.0, std=w_scale)
    with torch.no_grad():
        last.bias.fill_(bias_value)

# ------------------ loss ------------------
def l2_mean(pred, target):
    # mean over batch of vector L2 norms
    # shape: (B, k) -> (B,)
    norms = torch.norm(pred - target, p=2, dim=1)
    return norms.mean()

# ------------------ train/eval ------------------
def run_epoch(model, args, loader, opt, device, use_mse: bool, train=True):
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
         for (xb,) in loader:
            xb = xb.to(device)
            u_vec = model(xb)
            s = utils.Singular_RHS(u_vec, args.T, args.n, args.m, args.beta)
            loss = ( (s - u_vec).pow(2).mean() if use_mse else l2_mean(s, u_vec))
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward(); opt.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
    return tot / max(n,1)

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser(description="Train NN to minimize L2(u->s)")
    ap.add_argument("--n", type=int, default=1001)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--squash", type=str, default="None")
    ap.add_argument("--t_end", type=float, default=10) # default 10
    ap.add_argument("--t_schedule", type=str, default="random") #default Random, fixed has the similar results.
    ap.add_argument("--t_sorted", type=bool, default=False) #default False
    ap.add_argument("--m", type=int, default=21)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--data_size", type=float, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="512,256")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--normalize_x", action="store_true")
    ap.add_argument("--normalize_y", action="store_true")
    ap.add_argument("--mse", action="store_true", help="Use L2^2 (MSE) instead of L2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./checkpoints")
    args = ap.parse_args()

    cfg = Config(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, seed=args.seed,
                 val_ratio=args.val_ratio, weight_decay=args.weight_decay,
                 hidden=tuple(int(h) for h in args.hidden.split(",")) if args.hidden.strip() else (),
                 normalize_x=args.normalize_x, normalize_y=args.normalize_y,
                 use_mse=args.mse, out_dir=args.out_dir)
    set_seed(cfg.seed); os.makedirs(cfg.out_dir, exist_ok=True)

    # U, S = load_arrays(args.u_path, args.s_path)
    # X, y, meta = arrays_to_tensors(U, S, cfg)
    idx = generate_index(args.n, args.data_size, args)
    idx, meta = arrays_to_tensors(idx)
    tr_loader, va_loader = make_loaders(idx, cfg)

    device = torch.device(cfg.device)
    print(f"in_dim={meta['in_dim']} out_dim={meta['out_dim']} device={device} loss={'MSE' if cfg.use_mse else 'mean L2'}")

    model = MLP(meta['in_dim'], meta['out_dim'], args, cfg.hidden).to(device)
    init_positive_output_linear(model, bias_value=5.0, w_scale=1e-3) #set positive initials, 1.0, 3.0, works. the higher bias is, the higher loss initials and the later the testing value drop below 0.
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode="min",      # minimize val_loss
    factor=0.5,      # multiply LR by 0.5
    patience=2,      # wait 5 epochs without improvement
    min_lr=1e-6      # stop reducing below this
)


    best = float("inf"); best_path = os.path.join(cfg.out_dir, "best_u2s_l2.pt")
    for ep in range(1, cfg.epochs+1):
        tr_loss = run_epoch(model, args, tr_loader, opt, device, cfg.use_mse, train=True)
        va_loss = run_epoch(model, args, va_loader, opt, device, cfg.use_mse, train=False)
        print(f"[{ep:03d}] train {tr_loss:.6f} | val {va_loss:.6f}")
        # scheduler.step(va_loss)
        if va_loss < best:
            best = va_loss
            torch.save({"model_state": model.state_dict(),
                        "config": cfg.__dict__,
                        "meta": meta}, best_path)
    print(f"Saved best to: {best_path} (val {best:.6f})")
if __name__ == "__main__":
    main()
