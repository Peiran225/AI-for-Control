#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ROOT}/.venv/bin/python"

mkdir -p "${ROOT}/paper_runs/related_work_corrected"
echo "[1/5] HJB_NN [7] isolated full tumor sweep"
HJB_OUT="${ROOT}/paper_runs/faithful_related_work/hjb_nn_tumor_full"
IFS=',' read -r -a HJB_TAUS <<< "${HJB_NN_TUMOR_TAUS:-10,5,2}"
IFS=',' read -r -a HJB_SEEDS <<< "${HJB_NN_TUMOR_SEEDS:-0,1,2}"
for tau in "${HJB_TAUS[@]}"; do
  tau_name="$(printf '%g' "${tau}")"
  for seed in "${HJB_SEEDS[@]}"; do
    PYTHONPATH="${ROOT}" "${PY}" -m faithful_related_work.hjb_nn.runner run-tumor-one \
      --tau "${tau}" --seed "${seed}" --tau-start 10 \
      --train-trajectories 16 --validation-trajectories 16 \
      --width 96 --hidden-layers 3 --max-rounds 3 --min-rounds 2 \
      --maxiter 5000 --convergence-tolerance 0.5 \
      --candidates 32 --max-batch-points 8192 \
      --bvp-tolerance 1e-3 --bvp-max-nodes 100000 \
      --adaptive-bvp-max-nodes 20000 \
      --time-march-steps 16 --initial-mesh-nodes 41 \
      --evaluation-intervals 200 --adaptive-max-failures 256 \
      --out-dir "${HJB_OUT}/tau_${tau_name}/seed_${seed}"
  done
done
PYTHONPATH="${ROOT}" "${PY}" -m faithful_related_work.hjb_nn.runner aggregate-tumor \
  --taus "${HJB_NN_TUMOR_TAUS:-10,5,2}" \
  --seeds "${HJB_NN_TUMOR_SEEDS:-0,1,2}" \
  --out-dir "${HJB_OUT}"

echo "[2/5] Neural-PMP [3] official-code tumor adaptation"
for init in zero mid front random; do
  (
    cd "${ROOT}/external/NeuralPMP2024/NeuralPMP"
    suffix="${init}"
    if [[ "${init}" == "random" ]]; then
      suffix="random1"
    fi
    NEURAL_PMP_TUMOR_N="${NEURAL_PMP_TUMOR_N:-200}" \
    NEURAL_PMP_TUMOR_INIT="${init}" \
    NEURAL_PMP_TUMOR_SEED="${NEURAL_PMP_TUMOR_SEED:-1}" \
    NEURAL_PMP_TUMOR_SAVE="${ROOT}/paper_runs/related_work_corrected/neural_pmp_tumor_official_n200_${suffix}.npz" \
    PYTHONPATH=. \
    "${PY}" main.py \
      --Env_Name Tumor \
      --Sample_Num "${NEURAL_PMP_SAMPLE_NUM:-2000}" \
      --Training_iters "${NEURAL_PMP_TRAINING_ITERS:-3000}" \
      --PMP_iters "${NEURAL_PMP_ITERS:-1000}" \
      --PMPlr "${NEURAL_PMP_LR:-0.001}"
  )
done

echo "[3/5] DeepBSDE [2] official-code tumor adaptation"
"${PY}" "${ROOT}/scripts/run_deepbsde_tumor_official.py" \
  --out_dir "${ROOT}/paper_runs/related_work_corrected" \
  --exp_name deepbsde_tumor_sigma005_seed1 \
  --policy_mode smooth \
  --policy_temp "${DEEPBSDE_TUMOR_POLICY_TEMP:-0.5}" \
  --num_iterations "${DEEPBSDE_TUMOR_ITERS:-300}" \
  --batch_size "${DEEPBSDE_TUMOR_BATCH:-64}" \
  --valid_size "${DEEPBSDE_TUMOR_VALID:-128}" \
  --logging_frequency "${DEEPBSDE_TUMOR_LOG_EVERY:-50}"

echo "[4/5] PINN-PI [6] official-code tumor adaptation"
"${PY}" "${ROOT}/external/lyznet/examples/icml24-pinn-pi/tumor/tumor_finite_horizon.py" \
  --out_dir "${ROOT}/paper_runs/related_work_corrected" \
  --outer "${LYZNET_TUMOR_OUTER:-3}" \
  --steps_per_outer "${LYZNET_TUMOR_STEPS:-500}" \
  --batch "${LYZNET_TUMOR_BATCH:-512}" \
  --terminal_batch "${LYZNET_TUMOR_TERMINAL_BATCH:-512}" \
  --width "${LYZNET_TUMOR_WIDTH:-96}" \
  --layers "${LYZNET_TUMOR_LAYERS:-3}" \
  --lr "${LYZNET_TUMOR_LR:-5e-4}"

echo "[5/5] PI-DeepONet [5] paper-algorithm tumor adaptation"
"${PY}" "${ROOT}/scripts/run_pi_deeponet_tumor_paper.py" \
  --outer "${PIDEONET_TUMOR_OUTER:-4}" \
  --steps_per_outer "${PIDEONET_TUMOR_STEPS:-700}" \
  --batch "${PIDEONET_TUMOR_BATCH:-384}" \
  --terminal_batch "${PIDEONET_TUMOR_TERMINAL_BATCH:-192}" \
  --sensors "${PIDEONET_TUMOR_SENSORS:-32}" \
  --width "${PIDEONET_TUMOR_WIDTH:-96}" \
  --branch_depth "${PIDEONET_TUMOR_BRANCH_DEPTH:-2}" \
  --trunk_depth "${PIDEONET_TUMOR_TRUNK_DEPTH:-3}" \
  --lr "${PIDEONET_TUMOR_LR:-3e-4}" \
  --policy_temp_start "${PIDEONET_TUMOR_POLICY_TEMP_START:-2.0}" \
  --policy_temp "${PIDEONET_TUMOR_POLICY_TEMP:-0.5}" \
  --initial_u "${PIDEONET_TUMOR_INITIAL_U:-1.0}" \
  --out_dir "${ROOT}/paper_runs/related_work_corrected/pi_deeponet_seed1"

"${PY}" "${ROOT}/scripts/summarize_official_related_work_adaptations.py"
