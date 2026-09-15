# NeMoGym DAPO-17k for Nemotron 3.5 Super VL

This directory provides a **text-only** NeMoGym GRPO reference configuration
for Nemotron 3.5 Super VL (`NemotronH_Omni_Reasoning_V3`) on NeMo RL's
AutoModel (DTensor) backend. It uses the same DAPO-Math-17K Hugging Face
conversion procedure as the Nemotron 3 Ultra NeMoGym guide, sends rollouts
through Gym's Responses API, and receives rewards from Gym's
`math_with_judge` resource server.

Use [`dapo_nemotron_3_5_super_vl_nemo_gym.yaml`](dapo_nemotron_3_5_super_vl_nemo_gym.yaml).
Complete the shared setup in [`../README.md`](../README.md) first.

> [!IMPORTANT]
> This recipe is text-only. Do not attach images or enable image RL: the Super
> VL AutoModel path does not yet support image RL, including through NeMoGym.

## Training goal and data

The goal is to improve mathematical reasoning by optimizing sampled responses
for verifiable final-answer correctness, rather than by supervised imitation of
a reference solution. The recipe converts 6,400 `DAPO-Math-17K` mathematics
problems into Gym requests for training and reserves the next 256 examples for
validation. Each converted record carries a problem and expected answer;
Gym's `math_with_judge` service evaluates generated answers against that
expected result.

## Shared-storage layout

Mount the host shared-storage directory at `/shared` inside the container. The
host-to-container mapping and layout are:

```text
</YOUR/SHARED/STORAGE> (login/head node)  -->  /shared (training container)
```

The container-side tree is therefore:

```text
/shared
|____code
|    |____RL                    <- NeMo RL, branch super-3.5-automodel
|    |____Nemotron              <- Cookbook repository
|____models
|    |____NVIDIA-Nemotron-3.5-Super-EA-09112026
|____runs
|____.cache/huggingface
```

From the login/head node, define the host-side paths used by this guide. Do not
use `/shared` on the login node; it is the container mount point.

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
export MODEL_DIR="${SHARED_ROOT}/models/NVIDIA-Nemotron-3.5-Super-EA-09112026"
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"
```

## Configuration overview

The reference configuration uses four nodes with four GPUs each. It starts four
TP=4/EP=4 Gym-backed vLLM groups, collects DAPO-17k rollouts through the
Responses API, processes `math_with_judge` rewards, and trains the colocated
16-GPU AutoModel policy.

The included profile matches the native Super-VL DAPO rollout batch: 32 prompts
x 16 generations, or 512 samples per policy update, with `max_new_tokens:
2048` and a 4,096-token total sequence limit. This allows multi-step math
reasoning while remaining substantially smaller than the native DAPO profile's
8,192-token response budget. The policy global batch is also 512. Adjust
sequence length, generation count, step count, validation, checkpointing, and
observability for the target training program.

## Model and topology requirements

The checked-in recipe encodes required Super VL settings. Do not change these
as a group without revalidating the path:

- `policy.is_vlm: false` keeps this DAPO math workflow text-only.
- `policy.hf_config_overrides.num_nextn_predict_layers: 0` removes the MTP
  head during training.
- vLLM runs asynchronously over HTTP with `skip_tokenizer_init: false`,
  `mamba_ssm_cache_dtype: float32`, and `skip_mm_profiling: true`.
- Both training EP and vLLM TP/EP are `4`, matching four GPUs per node. DeepEP
  expert parallelism must not cross nodes.
- Generation and training are colocated across all 16 GPUs. vLLM releases
  memory while FusedAdam initializes and trains; a two-node policy slice OOMs
  at the first update.
- The 512-sample rollout and policy batch (32 prompts x 16 generations) divides
  evenly across the 16 policy logprob shards.

Use the `super-3.5-automodel` NeMo RL branch and a compatible Super-VL image.
After changing the branch, submodules, or image contents, set
`NRL_FORCE_REBUILD_VENVS=true`; see the parent README for the shared storage,
checkpoint, and container setup.

## Prepare DAPO-17k for Gym

This guide uses the same converter and deterministic split as the Nemotron 3
Ultra NeMoGym guide: 6,400 training examples followed by 256 validation
examples from
[`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k).
The converter writes Gym rows with `responses_create_params`, `expected_answer`,
and the `math_with_judge_simple_agent` reference required by this recipe.

> [!IMPORTANT]
> Run every command in this section on the **login/head node**, outside the
> Slurm allocation and training container. Use only the host-side paths rooted
> at `${SHARED_ROOT}`—the login node does **not** have a `/shared` path. The
> training container sees that same host directory at `/shared`, as configured
> later by `MOUNTS="${SHARED_ROOT}:/shared"`. Set `SHARED_ROOT`, `NEMO_RL`, and
> `NEMOTRON_REPO` as described in the parent README before continuing.

```bash
# Host-side paths: do not substitute the container's /shared mount here.
export COOKBOOK_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-dapo-nemo-gym"
export PREP_SCRIPT="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo-nemo-gym/prepare_hf_dapo_data_for_nemo_gym.py"
export HF_HOME="${HF_HOME:-${SHARED_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_DATASET_ID=BytedTsinghua-SIA/DAPO-Math-17k

# Install once on the login/head node if `python -c 'import datasets'` fails.
python -m pip install --upgrade --user datasets

mkdir -p "${HF_DATASETS_CACHE}" "${COOKBOOK_DIR}/dapo17k"

python "${PREP_SCRIPT}" \
  --dataset "${HF_DATASET_ID}" \
  --split train \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --output "${COOKBOOK_DIR}/dapo17k/train.jsonl" \
  --limit 6400 \
  --strict

python "${PREP_SCRIPT}" \
  --dataset "${HF_DATASET_ID}" \
  --split train \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --skip 6400 \
  --output "${COOKBOOK_DIR}/dapo17k/validation.jsonl" \
  --limit 256 \
  --strict
```

The resulting `dapo17k/train.jsonl` and `dapo17k/validation.jsonl` contain
6,400 and 256 rows, respectively. The recipe requires both files even though
the initial profile disables validation callbacks. Do not commit these generated
data files to the cookbook repository.

## Four-node interactive reference run

Request four exclusive four-GPU nodes from the login node. `ray.sub` emits an
attach helper when the allocation starts; use that helper to enter the job,
not a direct `srun` attach.

```bash
export NUM_NODES=4
export GPUS_PER_NODE=4
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export CONTAINER=<SITE_ACCESSIBLE_SUPER_VL_NEMO_RL_IMAGE>
export MOUNTS="${SHARED_ROOT}:/shared"
export NRL_FORCE_REBUILD_VENVS=true
export UV_LOCK_TIMEOUT=3600
unset COMMAND

cd "${NEMO_RL}"
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=interactive-super-vl-nemo-gym \
  --time=04:00:00 \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  ray.sub
```

After Slurm creates `<jobid>-attach.sh`, attach to the head node:

```bash
cd "${NEMO_RL}"
bash ./<jobid>-attach.sh
```

Inside that attached container, use fresh log paths on every attempt:

```bash
export NEMO_RL=/shared/code/RL
export RECIPE=/shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-dapo-nemo-gym/dapo_nemotron_3_5_super_vl_nemo_gym.yaml
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-EA-09112026
export RUN_DIR=/shared/runs/nemotron-3.5-super-vl-nemo-gym-$(date +%Y%m%d-%H%M%S)
mkdir -p "${RUN_DIR}"

cd "${NEMO_RL}"
NRL_FORCE_REBUILD_VENVS=true UV_LOCK_TIMEOUT=3600 \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${RECIPE}" \
  cluster.num_nodes=4 \
  cluster.gpus_per_node=4 \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  env.nemo_gym.nemo_gym_log_dir="${RUN_DIR}/nemo_gym" \
  logger.log_dir="${RUN_DIR}/training"
```

The driver should reach `Epoch 1/1`, collect `512/512` rollouts, then print
`Processing rewards`, `Computing logprobs`, `Training policy`, and the normal
one-step completion message. Gym service logs are under `${RUN_DIR}/nemo_gym`;
the runner creates a numbered subdirectory under `${RUN_DIR}/training`.

If a failed attempt leaves Gym child services alive, inspect and stop only
processes from that job via `<jobid>-attach.sh` (use indexed helpers for worker
nodes). Release the allocation with `scancel <jobid>` when finished.

## Four-node batch run

For an unattended run, submit the driver as `COMMAND` from the **login/head
node**. Host paths are used before submission; the command itself runs in the
container and therefore uses `/shared` paths. Choose a new `RUN_NAME` for each
attempt so that logs and checkpoints are not mixed. A checkpoint can require
roughly 1.8 TB of shared storage.

```bash
# Run on the login/head node, not inside the training container.
export NUM_NODES=4
export GPUS_PER_NODE=4
export NUM_STEPS=<NUM_TRAINING_STEPS>
export RUN_NAME=nemotron-3.5-super-vl-nemo-gym-$(date +%Y%m%d-%H%M%S)
export HOST_RUN_DIR="${SHARED_ROOT}/runs/${RUN_NAME}"
mkdir -p "${HOST_RUN_DIR}/logs" "${HOST_RUN_DIR}/checkpoints"

export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export CONTAINER=<SITE_ACCESSIBLE_SUPER_VL_NEMO_RL_IMAGE>
export MOUNTS="${SHARED_ROOT}:/shared"
export BASE_LOG_DIR="${HOST_RUN_DIR}/slurm"
export NRL_FORCE_REBUILD_VENVS=true
export UV_LOCK_TIMEOUT=3600

# Optional: load WANDB_API_KEY from a login-node-only environment file.
# Do not place credentials in this guide, the recipe, or COMMAND.
if [ -f "${SHARED_ROOT}/.env" ]; then set -a && source "${SHARED_ROOT}/.env" && set +a; fi

export CONTAINER_NEMO_RL=/shared/code/RL
export CONTAINER_RECIPE=/shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-dapo-nemo-gym/dapo_nemotron_3_5_super_vl_nemo_gym.yaml
export CONTAINER_MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-EA-09112026
export CONTAINER_RUN_DIR="/shared/runs/${RUN_NAME}"

export COMMAND="cd ${CONTAINER_NEMO_RL} && \
NRL_FORCE_REBUILD_VENVS=true UV_LOCK_TIMEOUT=3600 \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config ${CONTAINER_RECIPE} \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  policy.model_name=${CONTAINER_MODEL_DIR} \
  policy.tokenizer.name=${CONTAINER_MODEL_DIR} \
  grpo.max_num_steps=${NUM_STEPS} \
  checkpointing.enabled=true \
  checkpointing.checkpoint_dir=${CONTAINER_RUN_DIR}/checkpoints \
  env.nemo_gym.nemo_gym_log_dir=${CONTAINER_RUN_DIR}/nemo_gym \
  logger.log_dir=${CONTAINER_RUN_DIR}/logs \
  logger.wandb_enabled=true \
  logger.tensorboard_enabled=true \
  logger.wandb.project=nemo-rl-super-3.5 \
  logger.wandb.name=${RUN_NAME}"

cd "${NEMO_RL}"
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name="${RUN_NAME}" \
  --time=04:00:00 \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  ray.sub
```

Monitor the submitted job from the login/head node with `squeue -j <jobid>` and
inspect `${HOST_RUN_DIR}/slurm`, `${HOST_RUN_DIR}/logs`, and
`${HOST_RUN_DIR}/nemo_gym` as the job progresses. Cancel it with
`scancel <jobid>` if needed.

## Scope and next steps

For a sustained training campaign, choose a run-specific checkpoint directory,
enable validation, checkpointing, and external observability, and provision
roughly 1.8 TB per checkpoint. Keep the required Super-VL topology. For the
native DAPO reward path and the 16-node layout, see
[`../grpo-dapo/grpo_training_cookbook.md`](../grpo-dapo/grpo_training_cookbook.md).
