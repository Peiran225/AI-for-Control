import numpy as np
try:
    from scipy.interpolate import PchipInterpolator
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import torch


def Singular_RHS(u_vec, T, n, m, beta):
    """
    If u_vec.shape == (n+1,), returns s with shape (m, n+1).
    If u_vec.shape == (B, n+1), returns s with shape (m, n+1, B).
    """
    # Grids & parameters (trait grid)
    x = np.linspace(0.0, 1.0, m)[:, None]                 # (m, 1) column
    r = 2.0 / (1.0 + 3.0 * x**4)                          # (m, 1)
    phi1 = 1.0 / (1.0 + x**2)                             # (m, 1)
    M_diag = np.full((m, 1), 0.5)  

    # Run growth simulation (handles batched u)
    N, G_N = simulateTumorGrowth(u_vec, T, n, m)  # N: (m, n+1[, B]), G_N: ( [B] )

   

    if N.ndim == 2:
        # Shapes to broadcast across time
        # r, phi1, M_diag are (m,1) → broadcast to (m, n+1)
        beta = np.asarray(beta).reshape(-1, 1)   # (m,1)

        s1 = (beta * (r - G_N * M_diag)) @ N
        s2 = (beta * phi1) @ N
        s = np.divide(s1, s2, out=np.zeros_like(s1), where=(s2 != 0))
        return s
    else:
        # --- normalize shapes ---
        beta  = np.asarray(beta).reshape(-1)     # (m,)
        r     = np.asarray(r).reshape(-1)        # (m,)
        M_diag= np.asarray(M_diag).reshape(-1)   # (m,)
        phi1  = np.asarray(phi1).reshape(-1)
        # N: (m, n1, B) -> (B, m, n1)
        N_bmn = np.moveaxis(N, 2, 0)
        B, _, n1 = N_bmn.shape

        G_N = np.asarray(G_N).reshape(B)         # (B,)

        # --- weights ---
        # vec1[b, i] = r[i] - G_N[b]*M_diag[i]
        vec1 = r[None, :] - G_N[:, None] * M_diag[None, :]   # (B, m)

        # w1[b, i] = beta[i] * vec1[b, i]
        w1 = beta[None, :] * vec1                             # (B, m)

        # w2[i] = beta[i] * phi1[i]; same across batch, then broadcast to (B, m)
        w2 = (beta * phi1)[None, :]                           # (1, m)
        w2 = np.broadcast_to(w2, (B, m))                      # (B, m)

        # --- batched contractions ---
        # (B,m) x (B,m,n1) -> (B,n1)
        s1 = np.einsum('bm,bmn->bn', w1, N_bmn)
        s2 = np.einsum('bm,bmn->bn', w2, N_bmn)

        s = np.divide(s1, s2, out=np.zeros_like(s1), where=(s2 != 0))
        s = torch.from_numpy(s).float()
        return s                      
        

def simulateTumorGrowth(u_vec, T, n, m):
    """
    Accepts:
      - u_vec: shape (n+1,) or (B, n+1)

    Returns:
      - If input is (n+1,):
          N: (m, n+1)
          G_N: scalar (float)
      - If input is (B, n+1):
          N: (m, n+1, B)
          G_N: (B,)  (last-step G_N for each batch)
    """
    # --- 1) Setup time/trait grid and parameters ---
    time = np.linspace(0.0, T, n + 1)                 # (n+1,)
    x = np.linspace(0.0, 1.0, m)[:, None]             # (m,1)
    r = 2.0 / (1.0 + 3.0 * x**4)                      # (m,1)
    M_diag = np.full((m, 1), 0.5)                     # (m,1)
    phi1 = 1.0 / (1.0 + x**2)                         # (m,1)
    e = np.ones((m, 1)) / m                           # (m,1)
    n0 = 10.0 * np.ones((m, 1))                       # (m,1)

    # --- 2) Normalize/validate u_vec and build u(t) ---
    # --- Convert u_vec to NumPy if it's a Torch tensor ---
    if isinstance(u_vec, torch.Tensor):
        u_arr = u_vec.detach().cpu().numpy()  # breaks gradient, safe for NumPy
    else:
        u_arr = np.asarray(u_vec, dtype=float)
    if u_arr.ndim == 1:
        if u_arr.shape[0] != n + 1:
            raise ValueError(f"Control vector must have length n+1 ({n+1}).")
        # Single control → reuse batched path with B=1 then squeeze later
        u_arr = u_arr[None, :]   # (1, n+1)
        squeeze_output = True
    elif u_arr.ndim == 2:
        if u_arr.shape[1] != n + 1:
            raise ValueError(f"Each control vector must have length n+1 ({n+1}).")
        squeeze_output = False
    else:
        raise ValueError("u_vec must be shape (n+1,) or (B, n+1).")

    B = u_arr.shape[0]  # batch size

    if _HAS_SCIPY:
        # PCHIP along the time axis (axis=0). y should be shape (n+1, B)
        pchip = PchipInterpolator(time, u_arr.T, axis=0, extrapolate=True)
        def u_of_t(t):
            # returns shape (B,)
            return np.asarray(pchip(t)).reshape(B)
    else:
        # Linear interp fallback for each batch
        def u_of_t(t):
            # vectorized across batch using list comprehension
            return np.array([np.interp(t, time, u_arr[b]) for b in range(B)])

    # --- 3) Forward Euler integration (batched) ---
    N, G_N = eulerMethod_batched(u_of_t, r, phi1, M_diag, e, time, n0, B)

    if squeeze_output:
        # Return to original shapes for single control vector
        return N[..., 0], float(G_N[0])
    else:
        return N, G_N


def eulerMethod_batched(u1, r, phi1, M_diag, e, time, n0, B):
    """
    Batched forward Euler.
    Inputs:
      - u1(t): returns shape (B,) control at time t
      - r, phi1, M_diag, e, n0 as before (trait-wise columns)
    Returns:
      - N: (m, n+1, B)
      - G_N: (B,)  (last-step values)
    """
    dt = float(time[1] - time[0]) if len(time) > 1 else 0.0
    m = n0.shape[0]

    N = np.zeros((m, len(time), B), dtype=float)       # (m, n+1, B)
    N[:, [0], :] = n0[:, [0]][:, None]                 # broadcast n0 to all batches

    # Broadcast helpers
    r_b = r[:, :, None]            # (m,1,1)
    phi1_b = phi1[:, :, None]      # (m,1,1)
    M_diag_b = M_diag[:, :, None]  # (m,1,1)
    e_b = e.T[:, :, None]          # (1,m,1) for matmul-like sum

    G_last = np.zeros((B,), dtype=float)

    for k in range(len(time) - 1):
        t = time[k]
        N_current = N[:, [k], :]                        # (m,1,B)

        # Total population per batch: sum over m with weights e
        # (1,m,1) * (m,1,B) → (1,1,B) then squeeze to (B,)
        N_total = (e_b * N_current).sum(axis=1).sum(axis=0)   # (B,)
        G_N = np.log(1.0 + N_total)                            # (B,)
        G_last = G_N                                           # keep last

        # Controls at time t (B,)
        u_t = u1(t).reshape(1, 1, B)                           # (1,1,B)

        # Growth per trait & batch: (m,1,1) - (m,1,1)*(1,1,B) - (m,1,1)*(1,1,B)
        growth = r_b - phi1_b * u_t - M_diag_b * G_N.reshape(1, 1, B)  # (m,1,B)

        dNdt = growth * N_current                               # (m,1,B)
        N[:, [k + 1], :] = N_current + dt * dNdt                # (m,1,B)

    return N, G_last
