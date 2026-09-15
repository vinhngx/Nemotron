# NeMoGym DAPO-17k for Nemotron 3.5 Super VL

This directory provides a **text-only** NeMoGym GRPO reference configuration
for Nemotron 3.5 Super VL (`NemotronH_Omni_Reasoning_V3`) on
NeMo RL's AutoModel (DTensor) backend. It uses the same DAPO-17k source and
Ultra/NeMoGym preprocessing flow as the math workflow, sends rollouts through
Gym's Responses API, and receives rewards from Gym's `math_with_judge`
resource server.

Use [`dapo_nemotron_3_5_super_vl_nemo_gym.yaml`](dapo_nemotron_3_5_super_vl_nemo_gym.yaml).
Complete the shared setup in [`../README.md`](../README.md) first.

> [!IMPORTANT]
> This recipe is text-only. Do not attach images or enable image RL: the Super
> VL AutoModel path does not yet support image RL, including through NeMoGym.

## Configuration overview

The reference configuration uses four nodes with four GPUs each. It starts four
TP=4/EP=4 Gym-backed vLLM groups, collects DAPO-17k rollouts through the
Responses API, processes `math_with_judge` rewards, and trains the colocated
16-GPU AutoModel policy.

The included profile uses one prompt x 16 generations with `max_new_tokens:
256`. Adjust sequence length, generation count, step count, validation,
checkpointing, and observability for the target training program.

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
- The initial global batch is 16 (one prompt x 16 generations), matching the 16
  policy logprob shards.

Use the `super-3.5-automodel` NeMo RL branch and a compatible Super-VL image.
After changing the branch, submodules, or image contents, set
`NRL_FORCE_REBUILD_VENVS=true`; see the parent README for the shared storage,
checkpoint, and container setup.

## Prepare DAPO-17k for Gym

The checked-in [`dapo17k/`](dapo17k/) output was produced from the upstream
Ultra/NeMoGym preprocessing scripts, not from the direct GRPO JSONL. Regenerate
it after changing either Gym's source data or its `dapo17k.yaml` configuration:

```bash
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export GYM_ROOT="${NEMO_RL}/3rdparty/Gym-workspace/Gym"
export COOKBOOK_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-dapo-nemo-gym"

cd "${GYM_ROOT}/resources_servers/math_with_judge"
python prepare_dapo17k.py
python prepare_aime24.py

cd "${GYM_ROOT}"
gym dataset collate \
  --config resources_servers/math_with_judge/configs/dapo17k.yaml \
  --output-dir "${COOKBOOK_DIR}/dapo17k" \
  --mode train_preparation \
  ++math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
```

This creates the Gym-collated `dapo17k/train.jsonl` and the repeated AIME-2024
`dapo17k/validation.jsonl` (17,398 and 960 rows, respectively, in the tested
artifact). The recipe requires both files even though the initial profile
disables validation callbacks.

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

The driver should reach `Epoch 1/1`, collect `16/16` rollouts, then print
`Processing rewards`, `Computing logprobs`, `Training policy`, and the normal
one-step completion message. Gym service logs are under `${RUN_DIR}/nemo_gym`;
the runner creates a numbered subdirectory under `${RUN_DIR}/training`.

If a failed attempt leaves Gym child services alive, inspect and stop only
processes from that job via `<jobid>-attach.sh` (use indexed helpers for worker
nodes). Release the allocation with `scancel <jobid>` when finished.

## Scope and next steps

For a sustained training campaign, choose a run-specific checkpoint directory,
enable validation, checkpointing, and external observability, and provision
roughly 1.8 TB per checkpoint. Keep the required Super-VL topology. For the
native DAPO reward path and the 16-node layout, see
[`../grpo-dapo/grpo_training_cookbook.md`](../grpo-dapo/grpo_training_cookbook.md).
