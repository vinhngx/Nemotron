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


## Topology

The cookbook configuration defaults to 16 nodes x 8 GPUs, with training EP=8
and vLLM TP/EP=8. For 4-GPU nodes, override all four topology settings
together:

```text
cluster.gpus_per_node=4
policy.dtensor_cfg.expert_parallel_size=4
policy.generation.vllm_cfg.tensor_parallel_size=4
policy.generation.vllm_cfg.expert_parallel_size=4
```

Also keep `cluster.num_nodes` equal to the nodes requested from Slurm. The
DeepEP group must stay within one node, so training EP must not exceed GPUs per
node.

Four 4-GPU GB200 nodes are the practical minimum for the reference
configuration. Use 16 4-GPU nodes for the 4-GPU-node production layout; a
single node can initialize parts of the runtime but will OOM at the first
refit.

## Local checkpoint and outputs

The configuration defaults to the gated Hugging Face checkpoint. For a local
checkpoint, override both model and tokenizer paths. The examples below use
the variables defined in the parent README:

```bash
export RECIPE="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-dapo/dapo_nemotron_3_5_super_vl.yaml"
export RUN_NAME=nemotron-3.5-super-vl-dapo
export RUN_DIR="${SHARED_ROOT}/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
```

The default recipe creates checkpoints. Each checkpoint is about 1.8 TB. Set a
run-specific `checkpointing.checkpoint_dir` after confirming capacity.

## Four-node interactive reference run

This reference run executes one GRPO optimizer step on four 4-GPU nodes. It
preserves the Super VL sequence length, batch, DAPO reward path, and dynamic
sampling settings. Generation can take tens of minutes. Configure validation,
checkpoints, and external logging for longer campaigns.

Run from the login/head node:

```bash
export NUM_NODES=4
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
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-EA-09112026

cd "${NEMO_RL}"

NRL_FORCE_REBUILD_VENVS=true UV_LOCK_TIMEOUT=3600 uv run examples/run_grpo.py \
  --config "${RECIPE}" \
  cluster.num_nodes=4 \
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

## Full 4-GPU-node run

For the intended 16-node x 4-GPU layout, submit the same recipe through
`ray.sub`. Checkpoints, validation, and W&B are enabled below; make sure the
checkpoint destination has multiple terabytes of available capacity.

```bash
export NUM_NODES=16
export GPUS_PER_NODE=4
export RUN_NAME=nemotron-3.5-super-vl-dapo-16n4g
export RUN_DIR="${SHARED_ROOT}/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints"

if [ -f "${SHARED_ROOT}/.env" ]; then set -a && source "${SHARED_ROOT}/.env" && set +a; fi

export COMMAND="cd ${NEMO_RL} && \
NRL_FORCE_REBUILD_VENVS=true UV_LOCK_TIMEOUT=3600 uv run examples/run_grpo.py \
  --config ${RECIPE} \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  policy.model_name=${MODEL_DIR} \
  policy.tokenizer.name=${MODEL_DIR} \
  policy.dtensor_cfg.expert_parallel_size=4 \
  policy.generation.vllm_cfg.tensor_parallel_size=4 \
  policy.generation.vllm_cfg.expert_parallel_size=4 \
  checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints \
  logger.log_dir=${RUN_DIR}/logs"

cd "${NEMO_RL}"
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name="${RUN_NAME}" \
  --time=04:00:00 \
  --gres=gpu:"${GPUS_PER_NODE}" \
  ray.sub
```

NeMo RL resumes from the latest complete checkpoint in the configured
directory. The recipe's timeout checkpoint setting reserves time for a final
save under a four-hour Slurm limit. To release an interactive allocation, run
`scancel <jobid>` from the login/head node.
