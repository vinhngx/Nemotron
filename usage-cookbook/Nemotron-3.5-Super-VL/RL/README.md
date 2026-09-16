# Nemotron 3.5 Super VL RL Training Cookbook

This directory documents RL post-training for Nemotron 3.5 Super VL
(`NemotronH_Omni_Reasoning_V3`, 120B-A12B) on NeMo RL's AutoModel (DTensor)
backend. The supported workflow is **text-only** DAPO/GRPO with colocated vLLM
generation. Although the checkpoint includes a RADIO vision tower, image RL is
not supported on this backend yet.

Both cookbook paths have completed one text-only optimizer update on the
four-node GB200 reference topology. They are starting points, not convergence
or quality validation; establish task-specific metrics before scaling a
campaign. The NeMoGym path uses Gym-prepared DAPO-17k with the
`math_with_judge` resource server.

- `grpo-dapo/`: supported text-only DAPO/GRPO recipe and Slurm workflow.
- `grpo-dapo-nemo-gym/`: validated text-only NeMoGym DAPO-17k reference
  configuration and deployment guide.

## Runtime and hardware requirements

The `super-3.5-automodel` NeMo RL branch is required. It pins the AutoModel
source that recognizes the Super VL checkpoint and a Super-VL-aware vLLM fork.
Published NeMo RL containers do not include this runtime, so use a site image
built from this branch or a compatible image, then force Ray's worker virtual
environments to rebuild.

The cookbook recipes default to the validated GB200 test configuration: four
nodes x four GPUs, with training EP=4 and vLLM TP/EP=4. This is a validated
starting point, not a universal hardware requirement. For other GPU models or
memory capacities, adjust the node count, GPUs per node, and parallelism
settings together to fit available memory. When scaling to additional 4-GPU
nodes, retain EP/TP=4 and increase only the node count.

| Workload | Nodes x GPUs | Training EP | vLLM TP / EP |
| --- | --- | --- | --- |
| Default / validated GB200 test configuration | 4 x 4 | 4 | 4 / 4 |
| Scale-out on 4-GPU nodes | N x 4, N >= 4 | 4 | 4 / 4 |

Training EP must not span nodes. DeepEP's CUDA-IPC path requires
`expert_parallel_size <= gpus_per_node`.

## Shared-storage layout

Mount the shared root at `/shared` inside the container. The examples use this
layout:

```text
/shared
|____code
|    |____RL                    <- NeMo RL, branch super-3.5-automodel
|    |____Nemotron              <- Cookbook repository
|____models
|    |____NVIDIA-Nemotron-3.5-Super-EA-09112026
|____data
|    |____dapo17k
|____runs
|____.cache/huggingface
```

From the login/head node, define the paths used by the remaining guides:

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
export MODEL_DIR="${SHARED_ROOT}/models/NVIDIA-Nemotron-3.5-Super-EA-09112026"
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"
```

## Clone NeMo RL and initialize pinned sources

Clone the Super VL branch, then initialize its submodules. In particular, the
AutoModel submodule is pinned to a revision that includes the Super VL model
and its FSDP mixed-precision fix.

```bash
mkdir -p "${SHARED_ROOT}/code"
git clone --branch super-3.5-automodel --recursive \
  https://github.com/NVIDIA-NeMo/RL.git "${NEMO_RL}"
git clone https://github.com/NVIDIA-NeMo/Nemotron.git "${NEMOTRON_REPO}"

cd "${NEMO_RL}"
git submodule update --init --recursive
```

If a checkout already exists, verify that it is on `super-3.5-automodel` and
run the final submodule command before launching.

## Container and worker environments

Build an ARM64 release image from the checked-out `super-3.5-automodel` branch
for GB200 systems. Run this from the NeMo RL repository root after initializing
its submodules. The image tag includes the source revision so a submitted job
can be traced back to the exact NeMo RL checkout.

```bash
cd "${NEMO_RL}"
export NEMO_RL_REV="$(git rev-parse --short=12 HEAD)"

docker buildx build \
  --platform linux/arm64 \
  --progress=plain \
  --load \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --target release \
  --build-arg MAX_JOBS=8 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS="examples/nemo_gym/prefetch_super_all_envs.yaml" \
  -t "nemo-rl:super-3.5-automodel-${NEMO_RL_REV}-arm64" \
  .
```

Push or otherwise make the resulting image accessible to every Slurm node, then
set its site-accessible URI as `CONTAINER`. The image must expose the shared
filesystem and include Python/uv; dependencies are resolved from the pinned
sources in the checkout.

```bash
export CONTAINER=<SITE_ACCESSIBLE_URI_FOR_nemo-rl:super-3.5-automodel-${NEMO_RL_REV}-arm64>
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared"
export NRL_FORCE_REBUILD_VENVS=true
export UV_LOCK_TIMEOUT=3600
```

`NRL_FORCE_REBUILD_VENVS=true` is required after changing the branch,
submodules, or dependencies because Ray caches isolated worker environments.
On a multi-node launch, `UV_LOCK_TIMEOUT=3600` avoids failures while nodes
contend for the shared uv cache. If rebuilding inside a container image,
regenerate `/opt/nemo_rl_container_fingerprint` with
`python tools/generate_fingerprint.py` before reusing that image.

## Obtain the checkpoint

The recipe defaults to the gated Hugging Face checkpoint
`nvidia/nemotron-3.5-super-pre-ea-text-08282026`. The worked commands use the
local EA checkpoint shown in the storage layout; it is the checkpoint used for
the reference run. These are distinct artifacts: use one model directory and
the matching tokenizer consistently, rather than mixing their paths. If using
the recipe default, download it into a directory named for that artifact:

```bash
export HUB_MODEL_DIR="${SHARED_ROOT}/models/nemotron-3.5-super-pre-ea-text-08282026"
mkdir -p "${HUB_MODEL_DIR}" "${HF_HOME}"
hf download nvidia/nemotron-3.5-super-pre-ea-text-08282026 \
  --local-dir "${HUB_MODEL_DIR}"
```

The checkpoint contains remote model code and 63 BF16 safetensors shards
(about 232 GB). NeMo RL enables `trust_remote_code` for this model.

## Operational notes

- The recipe is text-only. It constructs the vision tower but sends no images.
- Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: it can break
  colocated trainer-to-vLLM IPC refit on GB200 clusters.
- Checkpoints include FP32 master weights and optimizer state and are about
  1.8 TB each. Plan shared-storage capacity before enabling checkpointing.
- Generation dominates step time. Begin with the included four-node reference
  configuration, then set run-specific step counts, checkpointing, validation,
  and observability for the target training program.

## Troubleshooting

| Symptom | Check and action |
| --- | --- |
| vLLM fails during initialization | Keep `max_num_batched_tokens: 4096`; Super-VL's Mamba cache requires more than the inherited 2,048-token limit. |
| OOM during policy refit | Use the colocated 4 nodes x 4 GPUs topology with EP/TP=4. Do not reduce the policy to two nodes. |
| Refit IPC failure | Remove `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and restart the job. |
| Environment setup stalls | Keep `NRL_FORCE_REBUILD_VENVS=true` after branch or image changes and use `UV_LOCK_TIMEOUT=3600`. |
| Gym services persist after a failure | Use the job's `<jobid>-attach.sh` helper to inspect matching processes, then `scancel <jobid>` when the allocation is no longer needed. |

## What to run next

Follow [the direct DAPO/GRPO guide](grpo-dapo/grpo_training_cookbook.md) for
the native reward path. Follow [the NeMoGym DAPO-17k
guide](grpo-dapo-nemo-gym/grpo_training_cookbook_nemo_gym.md) when the rollout
uses the Gym Responses API and `math_with_judge` verifier.
