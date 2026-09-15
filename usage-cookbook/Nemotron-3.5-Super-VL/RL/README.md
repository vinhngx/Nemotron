# Nemotron 3.5 Super VL RL Training Cookbook

This directory documents RL post-training for Nemotron 3.5 Super VL
(`NemotronH_Omni_Reasoning_V3`, 120B-A12B) on NeMo RL's AutoModel (DTensor)
backend. The supported workflow is **text-only** DAPO/GRPO with colocated vLLM
generation. Although the checkpoint includes a RADIO vision tower, image RL is
not supported on this backend yet.

Both cookbook paths provide a validated starting point for distributed Super VL
RL on the four-node GB200 topology. The NeMoGym path is text-only and uses
Gym-prepared DAPO-17k with the `math_with_judge` resource server. The reference
configuration has completed a real optimizer update end to end; scale-out,
checkpoint/resume, and learning-curve evaluation should be enabled according to
the operational requirements of the target deployment.

- `grpo-dapo/`: supported text-only DAPO/GRPO recipe and Slurm workflow.
- `grpo-dapo-nemo-gym/`: validated text-only NeMoGym DAPO-17k reference
  configuration and deployment guide.

## Runtime and hardware requirements

The `super-3.5-automodel` NeMo RL branch is required. It pins the AutoModel
source that recognizes the Super VL checkpoint and a Super-VL-aware vLLM fork.
Published NeMo RL containers do not include this runtime, so use a site image
built from this branch or a compatible image, then force Ray's worker virtual
environments to rebuild.

The production recipe defaults to 16 nodes x 8 GPUs. On 4-GPU nodes such as
GB200 NVL72, request 16 nodes x 4 GPUs and set training EP plus vLLM TP/EP to
4. Four 4-GPU nodes are the practical minimum for the reference configuration.

| Workload | Nodes x GPUs | Training EP | vLLM TP / EP |
| --- | --- | --- | --- |
| Default recipe | 16 x 8 | 8 | 8 / 8 |
| 4-GPU-node production layout | 16 x 4 | 4 | 4 / 4 |
| Four-node reference configuration | 4 x 4 | 4 | 4 / 4 |

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
`nvidia/nemotron-3.5-super-pre-ea-text-08282026`. For a local checkpoint,
override both `policy.model_name` and `policy.tokenizer.name` together. The
tested local EA checkpoint is shown in the layout above. If downloading the
recipe default, use a directory named for that artifact instead:

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

## What to run next

Follow [the direct DAPO/GRPO guide](grpo-dapo/grpo_training_cookbook.md) for
the native reward path and the 16-node production topology. Follow [the
NeMoGym DAPO-17k guide](grpo-dapo-nemo-gym/README.md) when the rollout must go
through the Gym Responses API and `math_with_judge` verifier.

## Status of the NeMoGym workflow

The NeMoGym cookbook has been ported and validated for one text-only Super VL
step on four nodes x four GPUs. It starts four TP=4/EP=4 vLLM model groups,
collects DAPO-17k Gym rollouts, routes rewards through `math_with_judge`, and
performs the colocated AutoModel optimizer update. Image RL remains unsupported
on the AutoModel path, including through NeMoGym.
