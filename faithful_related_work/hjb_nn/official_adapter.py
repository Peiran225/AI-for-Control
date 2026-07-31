"""Thin instrumentation around the audited author HJBnet compatibility port."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from .bvp import AdaptiveDataController


def checkpoint_payload(model) -> dict[str, Any]:
    weights, biases = model.export_model()
    return {
        "weights": weights,
        "biases": biases,
        "lb": model.lb,
        "ub": model.ub,
        "A_lb": model.A_lb,
        "A_ub": model.A_ub,
        "U_lb": model.U_lb,
        "U_ub": model.U_ub,
        "V_min": model.V_min,
        "V_max": model.V_max,
    }


def make_instrumented_hjbnet(official_class):
    """Add artifacts without further changing the compatibility-port logic."""

    class InstrumentedHJBnet(official_class):
        def __init__(
            self,
            problem,
            scaling,
            config,
            parameters=None,
            *,
            checkpoint_dir: Path,
            adaptive_controller: AdaptiveDataController | None = None,
        ):
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.adaptive_controller = adaptive_controller
            self.convergence_history: list[dict[str, Any]] = []
            self.optimizer_history: list[dict[str, Any]] = []
            self.checkpoint_paths: list[Path] = []
            self._completed_rounds = 0
            super().__init__(problem, scaling, config, parameters)

        def generate_data(self, desired_points, candidates_per_selection):
            if self.adaptive_controller is None:
                return super().generate_data(desired_points, candidates_per_selection)
            return self.adaptive_controller.generate(
                self, int(desired_points), int(candidates_per_selection)
            )

        def convergence_test(self, tf_dict, sample_grad, conv_tol, Ns_sub, s):
            before = int(self.Ns)
            converged = super().convergence_test(
                tf_dict, sample_grad, conv_tol, Ns_sub, s
            )
            self.convergence_history.append(
                {
                    "round": self._completed_rounds,
                    "converged": bool(converged),
                    "sample_gradient_l1": float(np.linalg.norm(sample_grad, ord=1)),
                    "subsample_size": int(Ns_sub),
                    "sample_size_before": before,
                    "sample_size_after": int(self.Ns),
                    "tolerance": float(conv_tol),
                    "growth_limit": float(s),
                }
            )
            return converged

        def _train_L_BFGS_B(self, *args, **kwargs):
            optimizer = super()._train_L_BFGS_B(*args, **kwargs)
            result = getattr(optimizer, "result", None)
            if result is not None:
                self.optimizer_history.append(
                    {
                        "round": self._completed_rounds + 1,
                        "success": bool(result.success),
                        "status": int(result.status),
                        "message": str(result.message),
                        "iterations": int(result.nit),
                        "function_evaluations": int(result.nfev),
                        "gradient_evaluations": int(result.njev),
                        "final_objective": float(result.fun),
                        "final_gradient_inf_norm": float(
                            np.linalg.norm(result.jac, ord=np.inf)
                        ),
                    }
                )
            self._completed_rounds += 1
            path = self.checkpoint_dir / f"checkpoint_round_{self._completed_rounds:02d}.mat"
            payload = checkpoint_payload(self)
            payload["completed_round"] = np.array([[self._completed_rounds]], dtype=np.int64)
            scipy.io.savemat(path, payload)
            self.checkpoint_paths.append(path)
            return optimizer

    InstrumentedHJBnet.__name__ = f"Instrumented{official_class.__name__}"
    return InstrumentedHJBnet
