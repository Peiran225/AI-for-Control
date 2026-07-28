#!/usr/bin/env python3
"""Stage every checkpoint-dependent result needed by the AAAI experiment paper.

The harness is intentionally non-destructive: a new output directory is
required unless ``--resume`` is used, and individual tasks never overwrite a
completed artifact.  It composes the repository's established evaluators
rather than reimplementing the optimal-control calculations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else ROOT / expanded).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_label_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw_path = text.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("LABEL must be nonempty")
    return label, resolve(Path(raw_path))


def artifact_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
    }
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            record["columns"] = next(reader, [])
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            record["top_level_fields"] = sorted(payload)
        else:
            record["json_type"] = type(payload).__name__
    return record


@dataclass(frozen=True)
class Task:
    name: str
    command: list[str]
    sentinel: Path
    cpu_only: bool = False


def task_record(task: Task) -> dict[str, Any]:
    return {
        "name": task.name,
        "command": task.command,
        "shell": shlex.join(task.command),
        "sentinel": str(task.sentinel.resolve()),
        "cpu_only": task.cpu_only,
    }


def diagnostic_tasks(args: argparse.Namespace, stage: Path) -> list[Task]:
    tasks: list[Task] = []
    settings = (("m16", 12801), ("m32", 25601))
    for resolution, points in settings:
        time_dir = stage / "diagnostics" / resolution / "time_only"
        tasks.append(
            Task(
                f"{resolution}_time_only",
                [
                    args.python,
                    str(SCRIPTS / "evaluate_time_only_two_state_offgrid_scalar.py"),
                    "--checkpoint",
                    str(args.time_checkpoint),
                    "--out-dir",
                    str(time_dir),
                    "--dense-points",
                    str(points),
                    "--refinement-multiplier",
                    str(args.time_refinement_multiplier),
                    "--interior-start",
                    "1.5",
                    "--interior-end",
                    "8.0",
                    "--query-batch-size",
                    str(args.query_batch_size),
                    "--report-scale-factor",
                    str(args.report_scale_factor),
                    "--torch-threads",
                    str(args.torch_threads),
                    "--device",
                    args.device,
                ],
                time_dir / "summary.json",
            )
        )
        for case, checkpoint in (
            ("feedback_cf", args.cf_checkpoint),
            ("feedback_der", args.der_checkpoint),
        ):
            out_dir = stage / "diagnostics" / resolution / case
            tasks.append(
                Task(
                    f"{resolution}_{case}",
                    [
                        args.python,
                        str(SCRIPTS / "evaluate_single_feedback_offgrid_scalar.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--out-dir",
                        str(out_dir),
                        "--dense-points",
                        str(points),
                        "--refinement-multiplier",
                        str(args.feedback_refinement_multiplier),
                        "--interior-start",
                        "1.5",
                        "--interior-end",
                        "8.0",
                        "--query-batch-size",
                        str(args.query_batch_size),
                        "--report-scale-factor",
                        str(args.report_scale_factor),
                        "--torch-threads",
                        str(args.torch_threads),
                        "--device",
                        args.device,
                    ],
                    out_dir / "summary.json",
                )
            )
    return tasks


def heldout_tasks(args: argparse.Namespace, stage: Path) -> list[Task]:
    tasks: list[Task] = []
    for short, checkpoint in (
        ("cf", args.cf_checkpoint),
        ("der", args.der_checkpoint),
    ):
        out_dir = stage / "heldout" / short
        tasks.append(
            Task(
                f"heldout_128_{short}",
                [
                    args.python,
                    str(SCRIPTS / "evaluate_feedback_refinement.py"),
                    "--pre",
                    str(checkpoint),
                    "--post",
                    str(checkpoint),
                    "--out_dir",
                    str(out_dir),
                    "--n",
                    "800",
                    "--radii",
                    args.heldout_radii,
                    "--samples",
                    "128",
                    "--direction_families",
                    "random",
                    "--seed",
                    str(args.heldout_seed),
                    "--substeps",
                    str(args.heldout_substeps),
                    "--pg_count",
                    str(args.heldout_pg_count),
                    "--bootstrap_repeats",
                    str(args.bootstrap_repeats),
                ],
                out_dir / "summary.json",
                cpu_only=True,
            )
        )
    return tasks


def timing_tasks(args: argparse.Namespace, stage: Path) -> list[Task]:
    time_path = stage / "timing" / "time_only.json"
    feedback_dir = stage / "timing" / "feedback"
    return [
        Task(
            "time_only_loaded_inference",
            [
                args.python,
                str(SCRIPTS / "benchmark_final_time_only_inference.py"),
                "--checkpoint",
                str(args.time_checkpoint),
                "--out",
                str(time_path),
                "--warmup",
                str(args.timing_query_warmups),
                "--repeats",
                str(args.timing_query_repeats),
                "--threads",
                "1",
            ],
            time_path,
            cpu_only=True,
        ),
        Task(
            "feedback_common_n800_objective_and_timing",
            [
                args.python,
                str(SCRIPTS / "compare_feedback_related_work.py"),
                "ours",
                "--output-dir",
                str(feedback_dir),
                "--case1",
                str(args.cf_checkpoint),
                "--case2",
                str(args.der_checkpoint),
                "--evaluation-intervals",
                "800",
                "--physical-alpha",
                "1",
                "--physical-beta",
                "40",
                "--physical-gamma",
                "8000",
                "--query-warmups",
                str(args.timing_query_warmups),
                "--query-repeats",
                str(args.timing_query_repeats),
                "--rollout-warmups",
                str(args.timing_rollout_warmups),
                "--rollout-repeats",
                str(args.timing_rollout_repeats),
            ],
            feedback_dir / "per_run_ours.csv",
            cpu_only=True,
        ),
    ]


def aggregate_task(args: argparse.Namespace, stage: Path) -> Task:
    return Task(
        "aggregate_final_tables",
        [
            args.python,
            str(SCRIPTS / "aggregate_final_paper_recompute.py"),
            "--stage-dir",
            str(stage),
            "--direct-nominal",
            str(args.direct_nominal),
            "--direct-resistant",
            str(args.direct_resistant),
            "--report-scale-factor",
            str(args.report_scale_factor),
            "--feedback-refinement-multiplier",
            str(args.feedback_refinement_multiplier),
        ],
        stage / "tables" / "summary.json",
        cpu_only=True,
    )


def figure_config(stage: Path) -> dict[str, Any]:
    weights = {"alpha": 1.0, "beta": 40.0, "gamma": 8000.0}
    cases = []
    for case, label in (
        ("time_only", "PMP/KKT time-only"),
        ("feedback_cf", "PMP/KKT-CF"),
        ("feedback_der", "PMP/KKT-DER"),
    ):
        cases.append(
            {
                "id": case,
                "label": label,
                "path": str(
                    (
                        stage
                        / "diagnostics"
                        / "m32"
                        / case
                        / "timeseries.npz"
                    ).resolve()
                ),
                "input_scale": "physical",
                "source_weights": weights,
                "nominal_prefix": "nominal",
                "resistant_heavy_prefix": "resistant_heavy",
            }
        )
    return {"target_weights": weights, "cases": cases}


def figure_task(args: argparse.Namespace, stage: Path) -> Task:
    config = stage / "figures" / "three_case_config.json"
    prefix = stage / "figures" / "three_case_continuous"
    return Task(
        "build_three_case_figures",
        [
            args.python,
            str(SCRIPTS / "build_three_case_dense_scalar_figure.py"),
            "--config",
            str(config),
            "--out-prefix",
            str(prefix),
            "--expected-points",
            "25601",
            "--zoom-start",
            "1.5",
            "--zoom-end",
            "8.0",
        ],
        prefix.with_name(prefix.name + "_manifest.json"),
        cpu_only=True,
    )


def field_mapping() -> dict[str, Any]:
    return {
        "main_key_policy_table": {
            "source": "tables/continuous_scalar_metrics.csv",
            "filter": (
                "resolution=m32, time_window=1.5<=t<8.0, "
                "sample_class=held_out_from_checkpoint_refinement"
            ),
            "fields": ["rms", "mean_abs", "max_abs"],
        },
        "m32_held_out_time_queries": {
            "source": "tables/continuous_scalar_metrics.csv",
            "filter": (
                "resolution=m32, "
                "sample_class=held_out_from_checkpoint_refinement"
            ),
            "definition": (
                "m32 coordinates absent from the scalar-refinement grid "
                "recorded in each checkpoint"
            ),
        },
        "continuous_resolution_check": {
            "source": "tables/continuous_resolution_convergence.csv",
            "purpose": "m16 versus m32 numerical stability",
        },
        "two_state_objectives": {
            "continuous_policy": "tables/continuous_objectives.csv",
            "common_n800": "tables/related_work_ours_rows.csv",
        },
        "control_adaptation": {
            "source": "tables/control_adaptation.csv",
            "fields": [
                "max_abs_delta_u",
                "mean_abs_delta_u",
                "rms_delta_u",
            ],
        },
        "held_out_128": {
            "source": "tables/heldout_128_summary.csv",
            "raw": "heldout/{cf,der}/per_sample.csv",
            "definition": (
                "feedback advantage against the same checkpoint's nominal-state "
                "schedule frozen and replayed"
            ),
        },
        "related_work_ours": {
            "source": "tables/related_work_ours_rows.csv",
            "baseline_join_rule": (
                "replace only ours/time_only/CF/DER rows; retain external "
                "baseline rows whose dependency hashes are unchanged"
            ),
        },
        "timing": {
            "source": "tables/timing.csv",
            "scope": (
                "loaded inference and deployment rollout only; do not compare "
                "offline construction times from unlike protocols"
            ),
        },
        "training_traces": {
            "source": "provenance/training_traces.json",
            "rule": "plot only recorded histories; never synthesize missing steps",
        },
        "main_figure": "figures/three_case_continuous_main.pdf",
        "supplement_figure": "figures/three_case_continuous_supplement.pdf",
    }


def run_task(task: Task, stage: Path, resume: bool) -> None:
    if task.sentinel.is_file():
        if resume:
            print(f"[skip] {task.name}: {task.sentinel}", flush=True)
            return
        raise FileExistsError(task.sentinel)
    task.sentinel.parent.mkdir(parents=True, exist_ok=True)
    log_path = stage / "logs" / f"{task.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if task.cpu_only:
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
    started = time.time()
    print(f"[run] {task.name}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(task.command)}\n")
        log.flush()
        completed = subprocess.run(
            task.command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(
            f"\nexit_code={completed.returncode} "
            f"elapsed_seconds={time.time() - started:.6f}\n"
        )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, task.command)
    if not task.sentinel.is_file():
        raise RuntimeError(
            f"{task.name} exited successfully but did not create {task.sentinel}"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--time-checkpoint", type=Path, required=True)
    result.add_argument("--cf-checkpoint", type=Path, required=True)
    result.add_argument("--der-checkpoint", type=Path, required=True)
    result.add_argument("--direct-nominal", type=Path, required=True)
    result.add_argument("--direct-resistant", type=Path, required=True)
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument(
        "--trace",
        action="append",
        type=parse_label_path,
        default=[],
        help="repeat LABEL=PATH for recorded time/CF/DER histories",
    )
    result.add_argument(
        "--baseline-artifact",
        action="append",
        type=parse_label_path,
        default=[],
        help="repeat LABEL=PATH for retained external-baseline/ablation artifacts",
    )
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--device", default="auto")
    result.add_argument("--torch-threads", type=int, default=8)
    result.add_argument("--query-batch-size", type=int, default=16)
    result.add_argument("--report-scale-factor", type=float, default=400.0)
    result.add_argument("--time-refinement-multiplier", type=int, default=8)
    result.add_argument(
        "--feedback-refinement-multiplier", type=int, default=16
    )
    result.add_argument("--heldout-radii", default="0.05,0.10,0.20")
    result.add_argument("--heldout-seed", type=int, default=20260720)
    result.add_argument("--heldout-substeps", type=int, default=4)
    result.add_argument("--heldout-pg-count", type=int, default=16)
    result.add_argument("--bootstrap-repeats", type=int, default=20000)
    result.add_argument("--timing-query-warmups", type=int, default=20)
    result.add_argument("--timing-query-repeats", type=int, default=200)
    result.add_argument("--timing-rollout-warmups", type=int, default=2)
    result.add_argument("--timing-rollout-repeats", type=int, default=5)
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.report_scale_factor <= 0.0:
        raise ValueError("--report-scale-factor must be positive")
    if (
        args.time_refinement_multiplier < 1
        or args.feedback_refinement_multiplier < 1
    ):
        raise ValueError("refinement multipliers must be positive")
    for field in (
        "time_checkpoint",
        "cf_checkpoint",
        "der_checkpoint",
        "direct_nominal",
        "direct_resistant",
        "out_dir",
    ):
        setattr(args, field, resolve(getattr(args, field)))
    dependencies = {
        "time_checkpoint": artifact_record(args.time_checkpoint),
        "cf_checkpoint": artifact_record(args.cf_checkpoint),
        "der_checkpoint": artifact_record(args.der_checkpoint),
        "direct_nominal": artifact_record(args.direct_nominal),
        "direct_resistant": artifact_record(args.direct_resistant),
        "scripts": {
            path.name: artifact_record(path)
            for path in (
                SCRIPTS / "evaluate_time_only_two_state_offgrid_scalar.py",
                SCRIPTS / "evaluate_single_feedback_offgrid_scalar.py",
                SCRIPTS / "evaluate_feedback_refinement.py",
                SCRIPTS / "compare_feedback_related_work.py",
                SCRIPTS / "benchmark_final_time_only_inference.py",
                SCRIPTS / "aggregate_final_paper_recompute.py",
                SCRIPTS / "build_three_case_dense_scalar_figure.py",
            )
        },
        "retained_baselines": {
            label: artifact_record(path) for label, path in args.baseline_artifact
        },
    }
    trace_manifest = {
        label: artifact_record(path) for label, path in args.trace
    }
    stage = args.out_dir
    if stage.exists() and any(stage.iterdir()) and not args.resume:
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory: {stage}"
        )
    stage.mkdir(parents=True, exist_ok=True)
    provenance = stage / "provenance"
    provenance.mkdir(exist_ok=True)

    tasks = (
        diagnostic_tasks(args, stage)
        + heldout_tasks(args, stage)
        + timing_tasks(args, stage)
        + [aggregate_task(args, stage)]
    )
    figure_config_path = stage / "figures" / "three_case_config.json"
    figure_config_path.parent.mkdir(parents=True, exist_ok=True)
    if not figure_config_path.exists():
        figure_config_path.write_text(
            json.dumps(figure_config(stage), indent=2) + "\n",
            encoding="utf-8",
        )
    tasks.append(figure_task(args, stage))

    plan = {
        "schema": "final-paper-recompute-plan-v1",
        "root": str(ROOT),
        "stage": str(stage),
        "physical_objective_weights": [1.0, 40.0, 8000.0],
        "training_to_physical_scale": args.report_scale_factor,
        "continuous_resolutions": {
            "m16": 12801,
            "m32": 25601,
            "primary": "m32",
        },
        "interior": [1.5, 8.0],
        "heldout": {
            "states": 128,
            "direction_family": "random",
            "seed": args.heldout_seed,
            "radii": args.heldout_radii,
        },
        "feedback_refinement_multiplier": args.feedback_refinement_multiplier,
        "tasks": [task_record(task) for task in tasks],
    }
    current_signature = {
        key: dependencies[key]["sha256"]
        for key in (
            "time_checkpoint",
            "cf_checkpoint",
            "der_checkpoint",
            "direct_nominal",
            "direct_resistant",
        )
    }
    signature_path = provenance / "input_signature.json"
    if args.resume and signature_path.is_file():
        previous = load_json(signature_path)
        if previous != current_signature:
            raise ValueError(
                "resume refused because checkpoint/direct-reference hashes changed"
            )
    signature_path.write_text(
        json.dumps(current_signature, indent=2) + "\n", encoding="utf-8"
    )
    (provenance / "dependencies.json").write_text(
        json.dumps(dependencies, indent=2) + "\n", encoding="utf-8"
    )
    (provenance / "training_traces.json").write_text(
        json.dumps(trace_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (stage / "field_mapping.json").write_text(
        json.dumps(field_mapping(), indent=2) + "\n", encoding="utf-8"
    )
    (stage / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    (stage / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(shlex.join(task.command) for task in tasks)
        + "\n",
        encoding="utf-8",
    )

    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    for task in tasks:
        run_task(task, stage, args.resume)
    print(f"Final-paper recomputation staged at {stage}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
