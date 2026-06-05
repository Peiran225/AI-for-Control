import torch
import utils
import matplotlib.pyplot as plt
import train
import math
import torch.nn as nn
import numpy as np
import argparse, os, random
from dataclasses import dataclass
from typing import Tuple

checkpoint = torch.load("checkpoints/best_u2s_l2.pt", map_location="cpu")

state_dict = checkpoint["model_state"]
config = checkpoint["config"]



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




ap = argparse.ArgumentParser(description="Train NN to minimize L2(u->s)")
ap.add_argument("--n", type=int, default=1001)
ap.add_argument("--T", type=int, default=10)
ap.add_argument("--m", type=int, default=21)
ap.add_argument("--squash", type=str, default="None")
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



# idx = np.random.rand(args.n + 1) * K # K<= 5 will have positive u and the error between u and s increase as K increase, >=6 u will be negative.
idx = np.random.rand(args.n + 1) * 10
# idx = np.linspace(0, 10, num=args.n+1, dtype=np.float32)
idx = torch.from_numpy(idx.astype("float32")).unsqueeze(0)

in_dim = idx.shape[1]
out_dim = idx.shape[1]

model = MLP(in_dim, out_dim, args, hidden=tuple(config["hidden"]))
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    u_vec = model(idx)

s = utils.Singular_RHS(u_vec, args.T, args.n, args.m, args.beta)

loss = (s - u_vec).pow(2).mean() 
print("loss d%", loss)
# plt.plot(idx, u_vec, label="u vs t", color="blue", linestyle="-")   # solid blue line
# plt.plot(idx, s, label="s vs t", color="red",  linestyle="--")      # dashed red line
# plt.plot(idx, u_vec, label="u vs t", color="blue", marker='o') # line with markers at each point 
# plt.plot(idx, s, label="s vs t", color="red", marker="*")

# detach from graph, move to CPU, then to numpy
idx_np = idx.detach().cpu().numpy().squeeze()
u_np   = u_vec.detach().cpu().numpy().squeeze()
s_np   = s.detach().cpu().numpy().squeeze()

# sort by idx
order = idx_np.argsort()
idx_order = idx_np[order]
u_order = u_np[order]
s_order = s_np[order]
mask_u = u_order  >= 0
mask_s = s_order >= 0

plt.plot(idx_order[mask_u], u_order[mask_u], label="u vs t", color="blue")
plt.plot(idx_order[mask_s], s_order[mask_s], label="s vs t", color="red")



plt.xlabel("idx")
plt.ylabel("u/s")
plt.title("u/s vs t")
plt.legend()   
plt.show()




