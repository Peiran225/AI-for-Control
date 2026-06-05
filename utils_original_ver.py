import numpy as np
try:
    # Optional: use PCHIP to match MATLAB's 'pchip' interpolation
    from scipy.interpolate import PchipInterpolator
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def Singular_RHS(u_vec, T, n, m, beta):
    """
      s: (m, n+1) array
    """
    # Grids & parameters
    x = np.linspace(0.0, 1.0, m)[:, None]                 # (m, 1) column
    r = 2.0 / (1.0 + 3.0 * x**4)                          # (m, 1)
    phi1 = 1.0 / (1.0 + x**2)                             # (m, 1)
    M_diag = np.full((m, 1), 0.5)                         # (m, 1)

    N, G_N = simulateTumorGrowth(u_vec, T, n, m)


    # Ensure beta is a (m,1) column vector for broadcasting
    beta = np.asarray(beta).reshape(-1, 1)   # (m,1)

    s1 = beta * ((r - G_N * M_diag) * N)
    s2 = beta * (phi1 * N)
    # s11 = (beta * (r - G_N * M_diag)) @ N
    # s22 = (beta * phi1) @ N

    print(s1.shape)  
    print(s2.shape)  

    s = np.divide(s1, s2, out=np.zeros_like(s1), where=(s2 != 0))
    return s


def simulateTumorGrowth(u_vec, T, n, m):
    # --- 1) Setup time/trait grid and parameters ---
    time = np.linspace(0.0, T, n + 1)                     # (n+1,)
    x = np.linspace(0.0, 1.0, m)[:, None]                 # (m,1)
    r = 2.0 / (1.0 + 3.0 * x**4)                          # (m,1)
    M_diag = np.full((m, 1), 0.5)                         # (m,1)
    phi1 = 1.0 / (1.0 + x**2)                             # (m,1)
    e = np.ones((m, 1)) / m                               # (m,1)
    n0 = 10.0 * np.ones((m, 1))                           # (m,1)

    # --- 2) Validate and build u(t) ---
    if len(u_vec) != n + 1:
        raise ValueError(f"Control vector must have length n+1 ({n+1}).")

    u_vec = np.asarray(u_vec, dtype=float)
    if _HAS_SCIPY:
        u1_func = PchipInterpolator(time, u_vec, extrapolate=True)
        u_of_t = lambda t: float(u1_func(t))
    else:
        # Fallback to linear interpolation if SciPy isn't available
        u_of_t = lambda t: float(np.interp(t, time, u_vec))

    # --- 3) Forward Euler integration ---
    N, G_N = eulerMethod(u_of_t, r, phi1, M_diag, e, time, n0)
    return N, G_N


def eulerMethod(u1, r, phi1, M_diag, e, time, n0):
    dt = float(time[1] - time[0]) if len(time) > 1 else 0.0
    m = n0.shape[0]
    N = np.zeros((m, len(time)), dtype=float)
    print(N.shape)  
    N[:, [0]] = n0

    G_N = 0.0
    for k in range(len(time) - 1):
        t = time[k]
        N_current = N[:, [k]]                               # (m,1)
        N_total = float(e.T @ N_current)                    # scalar
        G_N = np.log(1.0 + N_total)                         # scalar

        # Growth factor per trait (m,1)
        # (R - Phi1*u(t) - M*G_N) acting on diag → elementwise multipliers
        growth = r - phi1 * u1(t) - M_diag * G_N            # (m,1)

        dNdt = growth * N_current                           # (m,1)
        N[:, [k + 1]] = N_current + dt * dNdt               # (m,1)

    return N, G_N
