"""
Pilot: multi-level optimization (MLO) of a two-level hierarchical world model
on a linear-quadratic control problem with objective mismatch at BOTH levels.

Levels (innermost first):
  L1  low-level world model   z = E x,  z' ~ A_L z + B_L u            (ridge LS on real data)
  L2  high-level world model  h = P z,  h_{t+k} ~ A_H h + B_H a        (ridge LS on k-step imagined
                                                                        rollouts of the L1 model)
      high-level cost model   c_H(h,a) = h'Q_H h + a'R_H a             (LS on real k-step costs)
  L3  high-level controller   K_H = LQR(A_H, B_H, Q_H, R_H)            (unrolled Riccati)
  L4  abstraction (E, P)      min  log J_real(policy u = -K_H P E x, held k steps)

Decoupled baseline: E, P chosen by per-level unsupervised/prediction criteria (PCA, or
reduced-rank regression), i.e. each level is trained for its own loss.
MLO: (E, P) updated with hypergradients through L3 -> L2 -> L1 (exact implicit
differentiation of the LS normal equations + unrolled Riccati). The outer gradient dJ/dK
is obtained either from (a) an oracle differentiable simulator, or (b) black-box SPSA on
real rollouts (no environment gradients), which is what a real system allows.
"""
import argparse, json, os, time
import jax, jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------- env
def make_env(key, n_rel=4, n_dis=8, m=2, dis_noise=3.0, rel_noise=0.1):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = n_rel + n_dis
    # relevant block: 2 slowly-unstable modes (need control, small action response) and
    # 2 stable modes with large action response (high variance but easy) -> L2 mismatch
    A_rel = jnp.diag(jnp.array([1.03, 1.02, 0.6, 0.5]))
    A_rel = A_rel.at[0, 1].set(0.05)
    B_rel = jnp.array([[0.25, 0.0], [0.0, 0.25], [2.0, 0.5], [0.5, 2.0]])
    # distractors: large noise, stable, irrelevant to cost -> L1 mismatch
    A_dis = 0.7 * jnp.linalg.qr(jax.random.normal(k1, (n_dis, n_dis)))[0]
    B_dis = 0.1 * jax.random.normal(k2, (n_dis, m))
    A0 = jax.scipy.linalg.block_diag(A_rel, A_dis)
    B0 = jnp.concatenate([B_rel, B_dis], 0)
    Q0 = jax.scipy.linalg.block_diag(jnp.eye(n_rel), jnp.zeros((n_dis, n_dis)))
    noise0 = jnp.concatenate([rel_noise * jnp.ones(n_rel), dis_noise * jnp.ones(n_dis)])
    # hide the block structure with a random rotation
    Rot = jnp.linalg.qr(jax.random.normal(k3, (n, n)))[0]
    return dict(A=Rot @ A0 @ Rot.T, B=Rot @ B0, Q=Rot @ Q0 @ Rot.T, R=0.1 * jnp.eye(m),
                noise_std=noise0, Rot=Rot, n=n, m=m)


def env_step(env, x, u, eps):
    w = env["Rot"] @ (env["noise_std"] * eps)
    return env["A"] @ x + env["B"] @ u + w


def collect_random_data(env, key, n_traj=200, T=30):
    """Exploration data: random actions from random initial states."""
    n, m = env["n"], env["m"]

    def traj(key):
        k0, ku, kw = jax.random.split(key, 3)
        x0 = env["Rot"] @ (env["noise_std"] * jax.random.normal(k0, (n,)))
        U = jax.random.normal(ku, (T, m))
        W = jax.random.normal(kw, (T, n))

        def f(x, uw):
            u, w = uw
            xn = env_step(env, x, u, w)
            return xn, (x, u, xn)
        _, (X, U_, Xn) = jax.lax.scan(f, x0, (U, W))
        return X, U_, Xn
    X, U, Xn = jax.vmap(traj)(jax.random.split(key, n_traj))
    return X, U, Xn  # (n_traj, T, .)


def rollout_cost(env, K_full, k, key, n_roll=64, H=120):
    """Real closed-loop cost of u = -K_full x, zero-order-hold for k steps. Differentiable in K_full."""
    n, m = env["n"], env["m"]

    def one(key):
        k0, kw = jax.random.split(key)
        x0 = env["Rot"] @ (env["noise_std"] * jax.random.normal(k0, (n,)))
        W = jax.random.normal(kw, (H, n))

        def f(carry, tw):
            x, u = carry
            t, w = tw
            u = jnp.where(t % k == 0, -K_full @ x, u)
            c = x @ env["Q"] @ x + u @ env["R"] @ u
            return (env_step(env, x, u, w), u), c
        _, cs = jax.lax.scan(f, (x0, jnp.zeros(m)), (jnp.arange(H), W))
        return jnp.mean(jnp.minimum(cs, 1e6))
    return jnp.mean(jax.vmap(one)(jax.random.split(key, n_roll)))


# ----------------------------------------------------------------------------- levels
def orth_rows(M):
    q, _ = jnp.linalg.qr(M.T)
    return q.T


def ridge(Phi, Y, lam=1e-3):
    return jnp.linalg.solve(Phi.T @ Phi + lam * jnp.eye(Phi.shape[1]), Phi.T @ Y)  # (p, q)


def level1_fit(E, X, U, Xn):
    """Low-level latent model on real data: E x' ~ A_L E x + B_L u."""
    Z, Zn = X @ E.T, Xn @ E.T
    W = ridge(jnp.concatenate([Z, U], 1), Zn)
    d = E.shape[0]
    return W[:d].T, W[d:].T  # A_L (d,d), B_L (d,m)


def level2_fit(P, A_L, B_L, E, X0, key, k, env):
    """High-level model trained on k-step imagination of the L1 model (macro-action held for k)."""
    m = B_L.shape[1]
    Z0 = X0 @ E.T
    Abar = jax.random.normal(key, (Z0.shape[0], m))

    def imagine(z, a):
        def f(z, _):
            return A_L @ z + B_L @ a, None
        zk, _ = jax.lax.scan(f, z, None, length=k)
        return zk
    Zk = jax.vmap(imagine)(Z0, Abar)
    H0, Hk = Z0 @ P.T, Zk @ P.T
    W = ridge(jnp.concatenate([H0, Abar], 1), Hk)
    dh = P.shape[0]
    return W[:dh].T, W[dh:].T, Abar


def cost_fit(P, E, X, U, env, k):
    """High-level cost model c_H(h) = h'Q_H h fit to real k-step accumulated state cost."""
    # real k-step windows from the exploration data
    Tn = X.shape[1] // k * k
    Xw = X[:, :Tn].reshape(X.shape[0], -1, k, X.shape[-1])
    c = jnp.einsum("btki,ij,btkj->bt", Xw, env["Q"], Xw).reshape(-1)
    Hs = (Xw[:, :, 0] @ E.T @ P.T).reshape(-1, P.shape[0])
    dh = P.shape[0]
    iu = jnp.triu_indices(dh)
    feats = (Hs[:, :, None] * Hs[:, None, :])[:, iu[0], iu[1]]
    q = ridge(feats, c[:, None], 1e-2)[:, 0]
    Qh = jnp.zeros((dh, dh)).at[iu].set(q)
    Qh = 0.5 * (Qh + Qh.T)
    # project to PSD (differentiable eig-clip) so Riccati is well posed
    w, V = jnp.linalg.eigh(Qh)
    return (V * jnp.maximum(w, 1e-3)) @ V.T


def lqr(A, B, Q, R, iters=400):
    def f(_, Pm):
        S = R + B.T @ Pm @ B
        return Q + A.T @ Pm @ A - A.T @ Pm @ B @ jnp.linalg.solve(S, B.T @ Pm @ A)
    Pm = jax.lax.fori_loop(0, iters, f, Q)
    return jnp.linalg.solve(R + B.T @ Pm @ B, B.T @ Pm @ A)


def hierarchy(params, data, env, k, key):
    """Run L1 -> L2 -> L3 for given abstraction params; return the full-state gain."""
    X, U, Xn = data
    E, P = orth_rows(params["E"]), orth_rows(params["P"])
    Xf, Uf, Xnf = X.reshape(-1, X.shape[-1]), U.reshape(-1, U.shape[-1]), Xn.reshape(-1, Xn.shape[-1])
    A_L, B_L = level1_fit(E, Xf, Uf, Xnf)
    A_H, B_H, _ = level2_fit(P, A_L, B_L, E, Xf[::3], key, k, env)
    Q_H = cost_fit(P, E, X, U, env, k)
    R_H = k * env["R"]
    K_H = lqr(A_H, B_H, Q_H, R_H)
    return K_H @ P @ E  # u = -K_H h, h = P E x


# ----------------------------------------------------------------------------- baselines
def pca_init(data, d_L, d_H):
    X = data[0].reshape(-1, data[0].shape[-1])
    _, _, Vt = jnp.linalg.svd(X - X.mean(0), full_matrices=False)
    E = Vt[:d_L]
    Z = X @ E.T
    _, _, Vt2 = jnp.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return dict(E=E, P=Vt2[:d_H])


def rrr_init(data, d_L, d_H):
    """Reduced-rank regression: subspace that best predicts the next state (prediction-optimal)."""
    X, U, Xn = (a.reshape(-1, a.shape[-1]) for a in data)
    Phi = jnp.concatenate([X, U], 1)
    W = ridge(Phi, Xn)
    Yhat = Phi @ W
    _, _, Vt = jnp.linalg.svd(Yhat, full_matrices=False)
    E = Vt[:d_L]
    Z, Zn = X @ E.T, Xn @ E.T
    W2 = ridge(jnp.concatenate([Z, U], 1), Zn)
    _, _, Vt2 = jnp.linalg.svd(jnp.concatenate([Z, U], 1) @ W2, full_matrices=False)
    return dict(E=E, P=Vt2[:d_H])


# ----------------------------------------------------------------------------- MLO outer loop
def run_mlo(params, data, env, k, key, steps, lr, outer_grad, spsa_pairs=8, spsa_c=0.05,
            log_every=25, train=("E", "P"), n_roll=64):
    # `train` selects which abstraction levels receive hypergradients (others frozen): ablation
    opt = optax.multi_transform({"on": optax.adam(lr), "off": optax.set_to_zero()},
                                {name: ("on" if name in train else "off") for name in params})
    state = opt.init(params)
    kfix = jax.random.PRNGKey(123)  # fixed imagination noise for L2 (common random numbers)

    def K_of(p):
        return hierarchy(p, data, env, k, kfix)

    def logJ(Kf, key):
        return jnp.log(rollout_cost(env, Kf, k, key, n_roll=n_roll))

    @jax.jit
    def step_oracle(params, state, key):
        # oracle: differentiate the real simulator too (upper bound on what MLO can do)
        g = jax.grad(lambda p: logJ(K_of(p), key))(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state

    @jax.jit
    def step_spsa(params, state, key):
        # black-box: dJ/dK from 2*spsa_pairs real rollout batches, then VJP through L3->L2->L1
        Kf, vjp = jax.vjp(K_of, params)
        ks, kr = jax.random.split(key)
        deltas = jax.random.rademacher(ks, (spsa_pairs,) + Kf.shape).astype(Kf.dtype)
        scale = spsa_c * (jnp.linalg.norm(Kf) / jnp.sqrt(Kf.size) + 1e-3)
        keys = jax.random.split(kr, spsa_pairs)

        def est(d, kk):
            jp = logJ(Kf + scale * d, kk)
            jm = logJ(Kf - scale * d, kk)
            return (jp - jm) / (2 * scale) * d
        gK = jnp.mean(jax.vmap(est)(deltas, keys), 0)
        (g,) = vjp(gK)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state

    step = step_oracle if outer_grad == "oracle" else step_spsa
    hist = []
    for i in range(steps + 1):
        if i % log_every == 0:
            hist.append((i, float(eval_cost(env, K_of(params), k))))
        if i < steps:
            key, sk = jax.random.split(key)
            params, state = step(params, state, sk)
    return params, hist


def run_model_free(K0, env, k, key, steps, lr, spsa_pairs=8, spsa_c=0.05, log_every=25, n_roll=64):
    """Same real-rollout budget as MLO-SPSA, but SPSA directly on the full-state gain (no model)."""
    opt = optax.adam(lr)
    state = opt.init(K0)

    @jax.jit
    def step(Kf, state, key):
        ks, kr = jax.random.split(key)
        deltas = jax.random.rademacher(ks, (spsa_pairs,) + Kf.shape).astype(Kf.dtype)
        scale = spsa_c * (jnp.linalg.norm(Kf) / jnp.sqrt(Kf.size) + 1e-3)

        def est(d, kk):
            jp = jnp.log(rollout_cost(env, Kf + scale * d, k, kk, n_roll=n_roll))
            jm = jnp.log(rollout_cost(env, Kf - scale * d, k, kk, n_roll=n_roll))
            return (jp - jm) / (2 * scale) * d
        g = jnp.mean(jax.vmap(est)(deltas, jax.random.split(kr, spsa_pairs)), 0)
        upd, state = opt.update(g, state, Kf)
        return optax.apply_updates(Kf, upd), state
    Kf, hist = K0, []
    for i in range(steps + 1):
        if i % log_every == 0:
            hist.append((i, float(eval_cost(env, Kf, k))))
        if i < steps:
            key, sk = jax.random.split(key)
            Kf, state = step(Kf, state, sk)
    return Kf, hist


def eval_cost(env, Kf, k, seed=999):
    return rollout_cost(env, Kf, k, jax.random.PRNGKey(seed), n_roll=256, H=200)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--d_L", type=int, default=4)
    ap.add_argument("--d_H", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--spsa_pairs", type=int, default=8)
    ap.add_argument("--outer_rollouts", type=int, default=64, help="real rollouts per SPSA evaluation")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {"args": vars(args), "runs": []}
    for seed in range(args.seeds):
        t0 = time.time()
        key = jax.random.PRNGKey(seed)
        ke, kd, ko, ks = jax.random.split(key, 4)
        env = make_env(ke)
        data = collect_random_data(env, kd)
        kfix = jax.random.PRNGKey(123)
        run = {"seed": seed}

        # references: true-model LQR at stride 1 and with the same k-step zero-order hold
        K1 = lqr(env["A"], env["B"], env["Q"], env["R"])
        run["oracle_lqr_stride1"] = float(eval_cost(env, K1, 1))
        Ak = jnp.linalg.matrix_power(env["A"], args.k)
        Bk = sum(jnp.linalg.matrix_power(env["A"], j) for j in range(args.k)) @ env["B"]
        Kk = lqr(Ak, Bk, args.k * env["Q"], args.k * env["R"])
        run["oracle_lqr_zoh_k"] = float(eval_cost(env, Kk, args.k))
        run["zero_control"] = float(eval_cost(env, jnp.zeros((env["m"], env["n"])), args.k))

        for name, init in [("pca", pca_init), ("rrr", rrr_init)]:
            p0 = init(data, args.d_L, args.d_H)
            run[f"decoupled_{name}"] = float(eval_cost(env, hierarchy(p0, data, env, args.k, kfix), args.k))

        p0 = rrr_init(data, args.d_L, args.d_H)
        variants = [("mlo_oracle", "oracle", ("E", "P")), ("mlo_spsa", "spsa", ("E", "P")),
                    ("mlo_spsa_E_only", "spsa", ("E",)), ("mlo_spsa_P_only", "spsa", ("P",))]
        for vname, og, train in variants:
            p, hist = run_mlo(p0, data, env, args.k, ko, args.steps, args.lr, og, train=train,
                              spsa_pairs=args.spsa_pairs, n_roll=args.outer_rollouts)
            run[vname] = hist[-1][1]
            run[f"{vname}_curve"] = hist
            if vname == "mlo_spsa":
                # relevance check: fraction of E's row-space inside the cost-relevant subspace
                E = orth_rows(p["E"]); P = orth_rows(p["P"])
                rel = env["Rot"][:, :4]
                run["mlo_spsa_E_relevance"] = float(jnp.sum((E @ rel) ** 2) / E.shape[0])
                unst = env["Rot"][:, :2]
                run["mlo_spsa_PE_unstable_alignment"] = float(jnp.sum((P @ E @ unst) ** 2) / P.shape[0])
        K0 = hierarchy(p0, data, env, args.k, kfix)
        for vname, lr_mf in [("model_free_spsa", args.lr), ("model_free_spsa_lr10x", 10 * args.lr)]:
            _, hist = run_model_free(K0, env, args.k, ko, args.steps, lr_mf,
                                     spsa_pairs=args.spsa_pairs, n_roll=args.outer_rollouts)
            run[vname] = hist[-1][1]
            run[f"{vname}_curve"] = hist
        # real-environment budget per outer step (identical for MLO-SPSA and model-free SPSA)
        run["real_steps_per_outer_step"] = 2 * args.spsa_pairs * args.outer_rollouts * 120
        run["exploration_data_steps"] = int(np.prod(data[0].shape[:2]))
        E0, P0 = orth_rows(p0["E"]), orth_rows(p0["P"])
        run["rrr_E_relevance"] = float(jnp.sum((E0 @ env["Rot"][:, :4]) ** 2) / E0.shape[0])
        run["rrr_PE_unstable_alignment"] = float(jnp.sum((P0 @ E0 @ env["Rot"][:, :2]) ** 2) / P0.shape[0])
        run["seconds"] = time.time() - t0
        print(json.dumps({k_: v for k_, v in run.items() if not k_.endswith("curve")}), flush=True)
        results["runs"].append(run)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
