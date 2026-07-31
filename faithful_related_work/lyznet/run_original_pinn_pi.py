#!/usr/bin/env python3
"""Faithful runner for the original LyZNet PINN-PI pendulum example.

There is deliberately no tumor mode.  The ICML 2024 method solves a
stationary, infinite-horizon, quadratic stabilization problem and uses the
analytic unconstrained policy improvement

    u_{i+1}(x) = -1/2 R^{-1} g(x)^T grad V_i(x).

The common tumor benchmark is finite-horizon, has a bounded affine control,
and penalizes the drug linearly.  Calling a tumor rewrite a faithful run of
the original method would therefore be scientifically inaccurate.

This file remains Python 3.6 compatible because the official dreal/dreal4
container ships its real Python binding for Python 3.6.
"""

import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[2]
LYZNET_ROOT = REPO_ROOT / "external" / "lyznet"
UPSTREAM_EXAMPLE = (
    LYZNET_ROOT / "examples" / "icml24-pinn-pi" / "pendulum" / "pendulum.py"
)

# These are exactly the explicit keyword arguments in the checked-out author
# example.  batch_size=32 is an upstream neural_pi default, not an override.
OFFICIAL_CALL_KWARGS = {
    "num_of_iters": 10,
    "lr": 0.001,
    "layer": 2,
    "width": 10,
    "num_colloc_pts": 300000,
    "max_epoch": 10,
}
OFFICIAL_EFFECTIVE_BUDGET = dict(OFFICIAL_CALL_KWARGS, batch_size=32)

PENDULUM_SPEC = {
    "state": ["x1", "x2"],
    "f": ["x2", "19.6*sin(x1) - 4.0*x2"],
    "g": ["0", "40.0"],
    "domain": [[-2.0, 2.0], [-2.0, 2.0]],
    "Q": [[1.0, 0.0], [0.0, 1.0]],
    "R": [[2.0]],
    "initial_u": "-(1/40)*x1 - (19.6/40)*sin(x1)",
}

SOURCE_FILES = {
    "official_example": UPSTREAM_EXAMPLE,
    "neural_pi": LYZNET_ROOT / "src" / "lyznet" / "neural_pi.py",
    "dynamical_system": LYZNET_ROOT / "src" / "lyznet" / "dynamical_system.py",
    "neural_learner": LYZNET_ROOT / "src" / "lyznet" / "neural_learner.py",
    "utils": LYZNET_ROOT / "src" / "lyznet" / "utils.py",
}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    return subprocess.check_output(
        ["git", "-C", str(LYZNET_ROOT)] + list(args),
        universal_newlines=True,
    ).strip()


def upstream_revision():
    return git_output("rev-parse", "HEAD")


def tracked_upstream_dirty():
    return bool(git_output("status", "--porcelain", "--untracked-files=no"))


def source_hashes():
    return {
        name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)}
        for name, path in sorted(SOURCE_FILES.items())
    }


def validate_upstream_contract():
    """Statically verify that the checked-out example has the claimed contract."""
    source = UPSTREAM_EXAMPLE.read_text(encoding="utf-8")
    compact = "".join(source.split())
    required_fragments = {
        "control_affine_system": "lyznet.ControlAffineSystem(f,g,domain,sys_name,R=R)",
        "dynamics_f": "f=[x2,19.6*sp.sin(x1)-4.0*x2]",
        "dynamics_g": "g=[0,40.0]",
        "domain": "domain=[[-2,2]]*2",
        "control_cost": "R=2.0*sp.eye(1)",
        "initial_policy": (
            "initial_u=sp.Matrix([-(1/40)*x1-(19.6/40)*sp.sin(x1)])"
        ),
        "neural_pi": "lyznet.neural_pi(system,initial_u=initial_u,",
        "official_budget": (
            "num_of_iters=10,lr=0.001,layer=2,width=10,"
            "num_colloc_pts=300000,max_epoch=10"
        ),
    }
    checks = {
        name: fragment in compact for name, fragment in required_fragments.items()
    }
    if not all(checks.values()):
        missing = sorted(name for name, ok in checks.items() if not ok)
        raise RuntimeError(
            "Checked-out LyZNet example no longer matches the audited original "
            "pendulum contract; failed checks: {}".format(", ".join(missing))
        )
    return checks


def require_real_dreal():
    """Import and exercise essential APIs of the real dReal Python binding."""
    try:
        module = importlib.import_module("dreal")
    except Exception as exc:
        raise RuntimeError(
            "The faithful LyZNet path requires the real dReal Python binding. "
            "No stub/fallback is permitted. Use run_in_docker.sh or install "
            "dreal4 with its Python binding. Import failed: {}".format(exc)
        )

    required = ["Variable", "Expression", "Config", "CheckSatisfiability"]
    missing = [name for name in required if not hasattr(module, name)]
    origin = getattr(module, "__file__", None)
    if missing or origin is None:
        raise RuntimeError(
            "Imported module named dreal is not the required binding; missing "
            "APIs: {}; origin: {}".format(missing, origin)
        )
    try:
        origin_path = Path(origin).resolve()
        origin_path.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "Refusing repository-local dReal module {}; a real external binding "
            "is required.".format(origin)
        )

    # This exercises compiled dReal types, rather than accepting import alone.
    x = module.Variable("faithful_preflight_x")
    expression = x * x + 1
    config = module.Config()
    if not str(expression) or config is None:
        raise RuntimeError("dReal compiled API smoke check failed")
    return module


def docker_state():
    state = {"client_available": False, "daemon_available": False}
    try:
        client = subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        state["error"] = str(exc)
        return state
    state["client_available"] = client.returncode == 0
    state["client_version"] = client.stdout.strip()
    try:
        daemon = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}} {{.OSType}}/{{.Architecture}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        state["error"] = str(exc)
        return state
    state["daemon_available"] = daemon.returncode == 0
    state["daemon"] = daemon.stdout.strip()
    return state


def preflight_report():
    checks = validate_upstream_contract()
    dreal_info = {"available": False}
    try:
        dreal_module = require_real_dreal()
    except RuntimeError as exc:
        dreal_info["error"] = str(exc)
    else:
        dreal_info = {
            "available": True,
            "origin": str(Path(dreal_module.__file__).resolve()),
        }
    docker = docker_state()
    return {
        "mode": "static-preflight",
        "timestamp_utc": utc_now(),
        "upstream_revision": upstream_revision(),
        "tracked_upstream_dirty": tracked_upstream_dirty(),
        "source_sha256": source_hashes(),
        "official_contract_checks": checks,
        "official_effective_budget": OFFICIAL_EFFECTIVE_BUDGET,
        "pendulum_spec": PENDULUM_SPEC,
        "local_dreal": dreal_info,
        "docker": docker,
        "local_ready": dreal_info["available"],
        "docker_ready": docker["daemon_available"],
        "tumor_comparable": False,
        "tumor_incompatibility": (
            "LyZNet PINN-PI is stationary infinite-horizon quadratic "
            "stabilization with an admissible stabilizing initial controller; "
            "the common tumor OCP is finite-horizon, bounded, and has linear "
            "control cost. It is excluded from the tumor-J ranking."
        ),
    }


def build_pendulum(sp, lyznet):
    x1, x2 = sp.symbols("x1 x2")
    dynamics_f = [x2, 19.6 * sp.sin(x1) - 4.0 * x2]
    dynamics_g = [0, 40.0]
    domain = [[-2, 2], [-2, 2]]
    control_cost = 2.0 * sp.eye(1)
    initial_u = sp.Matrix([-(1.0 / 40.0) * x1 - (19.6 / 40.0) * sp.sin(x1)])
    system = lyznet.ControlAffineSystem(
        dynamics_f,
        dynamics_g,
        domain,
        "pendulum",
        R=control_cost,
    )
    return system, initial_u


def artifact_inventory(out_dir):
    inventory = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            inventory.append(
                {
                    "path": str(path.relative_to(out_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return inventory


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def ensure_fresh_output(out_dir):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            "Output directory must be absent or empty so upstream neural_learner "
            "cannot silently load prior checkpoints: {}".format(out_dir)
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def execute(args):
    validate_upstream_contract()
    dreal_module = require_real_dreal()
    sys.path.insert(0, str(LYZNET_ROOT / "src"))

    import sympy as sp
    import torch
    import lyznet

    out_dir = args.out_dir.resolve()
    ensure_fresh_output(out_dir)
    manifest_path = out_dir / "manifest.json"

    if args.full:
        mode = "official-full"
        call_kwargs = dict(OFFICIAL_CALL_KWARGS)
        effective_budget = dict(OFFICIAL_EFFECTIVE_BUDGET)
    else:
        mode = "original-pendulum-smoke"
        call_kwargs = {
            "num_of_iters": 1,
            "lr": 0.001,
            "layer": 2,
            "width": 10,
            "num_colloc_pts": args.smoke_collocation,
            "max_epoch": 1,
        }
        effective_budget = dict(call_kwargs, batch_size=32)

    manifest = {
        "status": "running",
        "started_at_utc": utc_now(),
        "claim": "original-method reproduction",
        "paper": "Physics-Informed Neural Network Policy Iteration",
        "benchmark": "author pendulum example",
        "mode": mode,
        "upstream_revision": upstream_revision(),
        "tracked_upstream_dirty": tracked_upstream_dirty(),
        "source_sha256": source_hashes(),
        "seed": args.seed,
        "call_kwargs": call_kwargs,
        "effective_budget": effective_budget,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "dreal_origin": str(Path(dreal_module.__file__).resolve()),
        "runner_sha256": sha256(Path(__file__)),
        "tumor_comparable": False,
        "tumor_incompatibility": (
            "stationary infinite-horizon quadratic stabilization is not the "
            "finite-horizon bounded linear-control-cost tumor OCP"
        ),
        "artifacts": [],
    }
    write_manifest(manifest_path, manifest)

    original_cwd = Path.cwd()
    try:
        os.chdir(str(out_dir))
        lyznet.utils.set_random_seed(args.seed)
        system, initial_u = build_pendulum(sp, lyznet)
        # The full branch is the checked-out author's call, including the
        # upstream defaults for plotting, verification, and batch size.
        lyznet.neural_pi(system, initial_u=initial_u, **call_kwargs)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = utc_now()
        manifest["error"] = "{}: {}".format(type(exc).__name__, exc)
        manifest["traceback"] = traceback.format_exc()
        manifest["artifacts"] = artifact_inventory(out_dir)
        write_manifest(manifest_path, manifest)
        raise
    finally:
        os.chdir(str(original_cwd))

    manifest["status"] = "completed"
    manifest["finished_at_utc"] = utc_now()
    manifest["artifacts"] = artifact_inventory(out_dir)
    manifest["checkpoint_count"] = sum(
        item["path"].endswith(".pt") for item in manifest["artifacts"]
    )
    write_manifest(manifest_path, manifest)


def build_parser():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true", help="exact author budget")
    mode.add_argument(
        "--smoke", action="store_true", help="one-iteration pendulum pipeline smoke"
    )
    mode.add_argument(
        "--preflight", action="store_true", help="read-only contract/environment audit"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--smoke-collocation", type=int, default=64)
    return parser


def main():
    args = build_parser().parse_args()
    if args.preflight:
        print(json.dumps(preflight_report(), indent=2, sort_keys=True))
        return
    if args.out_dir is None:
        raise SystemExit("--out-dir is required for --full and --smoke")
    if args.smoke and args.smoke_collocation < 32:
        raise SystemExit("--smoke-collocation must be at least upstream batch_size=32")
    execute(args)


if __name__ == "__main__":
    main()
