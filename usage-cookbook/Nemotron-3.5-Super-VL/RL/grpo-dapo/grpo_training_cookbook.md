# DAPO/GRPO for Nemotron 3.5 Super VL

This guide runs the supported, text-only Nemotron 3.5 Super VL DAPO/GRPO
recipe on NeMo RL's AutoModel (DTensor) backend. It uses FSDP2 training,
DeepEP expert parallelism, and colocated vLLM generation.

Use [`dapo_nemotron_3_5_super_vl.yaml`](dapo_nemotron_3_5_super_vl.yaml) as
the cookbook configuration. Complete the common setup in
[`../README.md`](../README.md) first.

> [!IMPORTANT]
> This recipe is **text-only**. Image RL is not supported for Super VL on the
> AutoModel path. The included configuration is a validated starting point for
> distributed text-only training; establish task-specific quality metrics before
> scaling a training campaign.

## Training goal and data

The goal is to improve mathematical reasoning by optimizing sampled responses
for verifiable final-answer correctness, rather than by supervised imitation of
a reference solution. Training uses `DAPOMath17K`, a collection of mathematics
problems with expected answers; the native `dapo_math_verify` reward function
checks each generated answer against that expected result. `DAPOMathAIME2024`
is used for validation, providing a held-out competition-math accuracy signal.

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
|    |____NVIDIA-Nemotron-3.5-Super-VL-09212026
|____data
|    |____dapo17k
|____runs
|____.cache/huggingface
```

From the login/head node, define the host-side paths used by this guide. Do not
use `/shared` on the login node; it is the container mount point.

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
export MODEL_DIR="${SHARED_ROOT}/models/NVIDIA-Nemotron-3.5-Super-VL-09212026"
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"
```

## Topology

Use the validated GB200 reference topology: eight nodes x four GPUs (32 GPUs
total), with training EP=4 and vLLM TP/EP=4.
This matches the interactive and batch examples below. Keep the topology
settings aligned when adapting the recipe:

```text
cluster.num_nodes=8
cluster.gpus_per_node=4
policy.dtensor_cfg.expert_parallel_size=4
policy.generation.vllm_cfg.tensor_parallel_size=4
policy.generation.vllm_cfg.expert_parallel_size=4
```

Also keep `cluster.num_nodes` equal to the nodes requested from Slurm. The
DeepEP group must stay within one node, so training EP must not exceed GPUs per
node.

Four nodes can initialize the runtime and complete one update, but the native
recipe repeatedly OOMed during update 2 even with a 1,024-token generation
budget. Use 8x4 for sustained training. When scaling 4-GPU nodes, retain
EP/TP=4 and increase only `cluster.num_nodes`.

The default profile uses a 2,048-token generation budget and a 4,096-token
total sequence limit. This supports multi-step mathematical reasoning as a
practical starting point; increase the limits only after reassessing memory
capacity and task quality.

## Reference timing and memory

The following second-update measurement was collected on eight GB200 nodes (32
GPUs total) with the local Super-VL EA checkpoint, DAPO-17k, and the default
2,048-token generation budget. It excludes cluster allocation and model/service
startup. The native path generated 1,536 candidates through dynamic sampling
and retained 512 for the policy update.

| Metric | Observed value |
| --- | --- |
| End-to-end update | 351.45 s |
| Generation | 253.52 s |
| Policy training | 52.36 s |
| Policy/reference log-probs | 7.43 s |
| End-to-end throughput | 93.53 tokens/s/GPU |
| Generation throughput | 129.66 tokens/s/GPU |
| Sampled head-node memory peak | 150.2 GiB/GPU |

Treat these as an operational reference, not a convergence or capacity claim.
The measurement completed two updates. Treat 8x4 as the reference for a
production campaign; validate repeated-update memory on the exact container
and hardware before changing topology or token budgets.

## Checkpoints and outputs

The configuration defaults to the gated Hugging Face checkpoint. For a local
checkpoint, override both model and tokenizer paths. The interactive and batch
sections define their own recipe and run-output paths.

The default recipe creates checkpoints. Each checkpoint is about 1.8 TB. Set a
run-specific `checkpointing.checkpoint_dir` after confirming capacity.

## Eight-node interactive run

Use this interactive run as the recommended clean-pipeline check before a
production batch run. It provides a convenient environment for debugging the
distributed setup, generation, rewards, log-probability calculation, and policy
refit before committing to a longer campaign. The run executes one GRPO
optimizer step on eight 4-GPU nodes. Generation can take tens of minutes.

Run from the login/head node:

```bash
export NUM_NODES=8
export GPUS_PER_NODE=4
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export CONTAINER=<SITE_ACCESSIBLE_SUPER_VL_NEMO_RL_IMAGE>
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared"
export NRL_FORCE_REBUILD_VENVS=true
export UV_LOCK_TIMEOUT=3600
unset COMMAND

cd "${NEMO_RL}"
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=interactive-super-vl-dapo \
  --time=04:00:00 \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  ray.sub
```

After the job starts, attach using the helper emitted by `ray.sub`:

```bash
bash ./<jobid>-attach.sh
```

Inside the attached container, launch the reference configuration:

```bash
export NEMO_RL=/shared/code/RL
export RECIPE=/shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-dapo/dapo_nemotron_3_5_super_vl.yaml
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026

cd "${NEMO_RL}"

NRL_FORCE_REBUILD_VENVS=true UV_LOCK_TIMEOUT=3600 uv run examples/run_grpo.py \
  --config "${RECIPE}" \
  cluster.num_nodes=8 \
  cluster.gpus_per_node=4 \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  policy.dtensor_cfg.expert_parallel_size=4 \
  policy.generation.vllm_cfg.tensor_parallel_size=4 \
  policy.generation.vllm_cfg.expert_parallel_size=4 \
  grpo.max_num_epochs=1 \
  grpo.max_num_steps=1 \
  grpo.val_at_start=false \
  grpo.val_at_end=false \
  checkpointing.enabled=false \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=false
```

The log should reach `Epoch 1/1`, `Step 1/1`, complete generation, log-prob
calculation, and `Training policy`, then print `Max number of steps has been
reached`. This provides a fast confirmation of the complete distributed
training path before launching a longer campaign.

## Eight-node batch run

For batch run mode, submit the same recipe through
`ray.sub`. Checkpoints, validation, and W&B are enabled below; make sure the
checkpoint destination has multiple terabytes of available capacity.

```bash
# Run on the login/head node, not inside the training container.
export NUM_NODES=8
export GPUS_PER_NODE=4
export NUM_STEPS=<NUM_TRAINING_STEPS>
export RUN_NAME=nemotron-3.5-super-vl-dapo-4n4g
export HOST_RUN_DIR="${SHARED_ROOT}/runs/${RUN_NAME}"
mkdir -p "${HOST_RUN_DIR}/logs" "${HOST_RUN_DIR}/checkpoints"

export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export CONTAINER=<SITE_ACCESSIBLE_SUPER_VL_NEMO_RL_IMAGE>
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared"
export BASE_LOG_DIR="${HOST_RUN_DIR}/slurm"
export NRL_FORCE_REBUILD_VENVS=true
export UV_LOCK_TIMEOUT=3600

# Optional: load WANDB_API_KEY from a login-node-only environment file.
# Do not place credentials in this guide, the recipe, or COMMAND.
if [ -f "${SHARED_ROOT}/.env" ]; then set -a && source "${SHARED_ROOT}/.env" && set +a; fi

export CONTAINER_NEMO_RL=/shared/code/RL
export CONTAINER_RECIPE=/shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-dapo/dapo_nemotron_3_5_super_vl.yaml
export CONTAINER_MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export CONTAINER_RUN_DIR="/shared/runs/${RUN_NAME}"

export COMMAND="cd ${CONTAINER_NEMO_RL} && \
NRL_FORCE_REBUILD_VENVS=true UV_LOCK_TIMEOUT=3600 uv run examples/run_grpo.py \
  --config ${CONTAINER_RECIPE} \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  policy.model_name=${CONTAINER_MODEL_DIR} \
  policy.tokenizer.name=${CONTAINER_MODEL_DIR} \
  policy.dtensor_cfg.expert_parallel_size=4 \
  policy.generation.vllm_cfg.tensor_parallel_size=4 \
  policy.generation.vllm_cfg.expert_parallel_size=4 \
  grpo.max_num_steps=${NUM_STEPS} \
  checkpointing.enabled=true \
  checkpointing.checkpoint_dir=${CONTAINER_RUN_DIR}/checkpoints \
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

Set `NUM_STEPS` to the number of optimizer updates for this allocation. NeMo RL
resumes from the latest complete checkpoint in the configured directory, so
resubmit the same command with the same `RUN_NAME` after a time limit. The
recipe's timeout checkpoint setting reserves time for a final save under a
four-hour Slurm limit. To release an interactive allocation, run `scancel
<jobid>` from the login/head node.
