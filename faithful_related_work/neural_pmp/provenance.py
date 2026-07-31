"""Auditable provenance for the dirty nested NeuralPMP2024 checkout."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


UPSTREAM_RELATIVE = Path("external/NeuralPMP2024")
ENV_PATH = "NeuralPMP/Env/Env.py"
SOLVER_PATH = "NeuralPMP/Solver/Solver.py"


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_record(repository: Path, relative: str) -> dict[str, Any]:
    head_bytes = _git(repository, "show", f"HEAD:{relative}")
    worktree_bytes = (repository / relative).read_bytes()
    diff_bytes = _git(repository, "diff", "--", relative)
    blob_oid = _git(repository, "rev-parse", f"HEAD:{relative}").decode().strip()
    numstat = _git(repository, "diff", "--numstat", "--", relative).decode().strip()
    return {
        "path": relative,
        "head_blob_oid": blob_oid,
        "head_blob_sha256": _sha256(head_bytes),
        "worktree_sha256": _sha256(worktree_bytes),
        "worktree_matches_head_bytes": worktree_bytes == head_bytes,
        "tracked_diff_sha256": _sha256(diff_bytes),
        "tracked_diff_numstat": numstat,
    }


def _verify_head_contract(repository: Path) -> dict[str, bool]:
    env_head = _git(repository, "show", f"HEAD:{ENV_PATH}").decode("utf-8")
    solver_head = _git(repository, "show", f"HEAD:{SOLVER_PATH}").decode("utf-8")
    checks = {
        "lqr_state_samples_minus5_plus10rand": (
            "x_cur = -5 * np.ones(self.state_dim) + 10 * np.random.rand(self.state_dim)" in env_head
        ),
        "lqr_action_samples_minus5_plus10rand": (
            "u_cur = -5 * np.ones(self.action_dim) + 10 * np.random.rand(self.action_dim)" in env_head
        ),
        "lqr_controller_lower_minus100000": "self.action_lower = -100000" in env_head,
        "lqr_controller_upper_plus100000": "self.action_upper = 100000" in env_head,
        "lqr_two_hidden_relu_network_width64": (
            "Networks.NNTypeB(input=self.state_dim+self.action_dim, output=self.state_dim, hidden=64)" in env_head
        ),
        "released_solver_clips_hamiltonian_gradient": (
            "H = torch.where(H > action_upper" in solver_head
            and "H = torch.where(H < action_lower" in solver_head
        ),
        "released_solver_then_uses_clipped_gradient": "u = u - lr * H" in solver_head,
    }
    missing = [name for name, passed in checks.items() if not passed]
    if missing:
        raise RuntimeError(f"nested HEAD no longer satisfies audited Neural-PMP contract: {missing}")
    return checks


def collect_upstream_provenance(repo_root: Path) -> dict[str, Any]:
    """Hash HEAD blobs and worktree files without trusting the dirty checkout."""
    repository = Path(repo_root) / UPSTREAM_RELATIVE
    head = _git(repository, "rev-parse", "HEAD").decode().strip()
    tracked_dirty_files = [
        line for line in _git(repository, "diff", "--name-only", "HEAD").decode().splitlines() if line
    ]
    untracked_files = [
        line[3:]
        for line in _git(repository, "status", "--short").decode().splitlines()
        if line.startswith("?? ")
    ]
    return {
        "upstream": "NeuralPMP2024",
        "nested_head": head,
        "tracked_dirty": bool(tracked_dirty_files),
        "tracked_dirty_files": tracked_dirty_files,
        "untracked_file_count": len(untracked_files),
        "files": {
            "env": _file_record(repository, ENV_PATH),
            "solver": _file_record(repository, SOLVER_PATH),
        },
        "head_contract_checks": _verify_head_contract(repository),
        "source_policy": {
            "lqr_sampling_architecture_and_constraints": "audited nested HEAD Env.py blob",
            "controller_update": "paper Eqs. 32-33: raw Hamiltonian gradient step, then action projection",
            "worktree_execution": False,
            "released_solver_defect": (
                "HEAD Solver clips dH/du using action bounds before the update; the faithful independent "
                "runner deliberately corrects this paper/code contradiction and never imports that Solver"
            ),
        },
        "tracked_diff_summary": {
            "env": (
                "worktree adds import os and a tumor environment after the original LQR block; newline "
                "normalization broadens the textual diff; LQR claims are read from the HEAD blob"
            ),
            "solver": (
                "worktree adds initial controls, output returns, and action projection, but retains the "
                "released pre-update Hamiltonian-gradient clipping defect"
            ),
        },
    }
