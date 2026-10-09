# Train Nemotron 3.5 Super VL for Workplace Tool Use with GRPO

A workplace assistant rarely succeeds with one model response. It may need to
find a colleague, inspect a message, update a project task, and confirm that it
changed the correct record. This guide trains Nemotron 3.5 Super VL on that
kind of multi-step behavior with full-weight GRPO, NeMo RL's Megatron backend,
colocated vLLM generation, and the NeMo Gym Workplace Assistant environment.

The included environment is a local simulation. It does not connect directly
to Jira, Confluence, or another production service. Its separate agent,
resource server, dataset, and verifier provide a reference design for plugging
in a resettable customer harness later in the workflow.

## How the task works

Each dataset row asks the assistant to perform a workplace operation, such as
replying to a particular email or updating a project task. NeMo Gym exposes 27
tools over five mutable databases plus a company-directory lookup. The tools
cover email, calendar, project management, customer relationship management,
and analytics.

A rollout proceeds through these components:

| Component | Role |
| --- | --- |
| Dataset | Supplies the user request, tool schemas, reference actions, and `agent_ref` |
| `workplace_assistant_simple_agent` | Alternates policy and tool calls for at most six steps |
| Policy model server | Runs the trainable model and preserves token IDs and generation log probabilities |
| Workplace Assistant resource server | Executes tools against isolated in-memory workplace state |
| Verifier | Replays predicted and reference actions in fresh environments and compares final database state |
| Reward | Returns `1.0` for an equivalent final state and `0.0` otherwise |

The state-based reward permits different valid tool sequences to receive the
same score. A tool error, malformed call, unfinished task, wrong mutation, or
unintended side effect can make the final state differ and produce reward
zero. The six-step loop is internal to the NeMo Gym agent; NeMo RL treats the
complete trajectory as one rollout.

The published dataset is hosted at
[`nvidia/Nemotron-RL-agent-workplace_assistant`](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant)
and uses the Apache 2.0 license. The snapshot used to validate this guide
contained 1,255 training tasks and 545 validation tasks. Published dataset
sizes can change over time.

## Reference configuration

The included
[`super_vl_3_5_workplace_assistant_megatron.yaml`](super_vl_3_5_workplace_assistant_megatron.yaml)
uses the following settings:

| Component | Setting |
| --- | --- |
| Compute | 4 nodes x 4 GPUs |
| Training | Full-weight BF16, TP=4, EP=16 |
| Generation | Four colocated vLLM groups, each TP=4 |
| GRPO batch | 32 tasks x 8 sampled trajectories |
| Schedule | 20 updates; validation before RL, every 5 updates, and after training |
| Agent horizon | Up to 6 model and tool steps |
| Sequence limit | 16,384 total tokens; up to 2,048 generated tokens per model call |
| Validation | Fixed category-balanced 40-task subset, one deterministic trajectory per task |
| Checkpoints | Disabled |

Each update generates 256 trajectories. The 20-update example samples 640
training tasks and produces 5,120 trajectories. Training data is shuffled, so
those tasks are sampled from the full training file. The 16,384-token limit
accommodates the tool schemas, reasoning, tool results, and multi-step history.
TP=4 shards dense and attention tensors within each node. EP=16 distributes
the model's 512 routed experts across all GPUs, with 32 experts per
expert-parallel rank.

## Prerequisites

Use storage visible to every allocated node and mount it at `/shared` inside
the container. The guide assumes this layout:

```text
</YOUR/SHARED/STORAGE> (Host):/shared (Container)
|____code
|    |____RL                    <- NeMo RL, branch super-v3.5-posttraining
|    |____Nemotron              <- this cookbook repository
|____models
|    |____NVIDIA-Nemotron-3.5-Super-VL-09212026
|____runs
|    |____super35-workplace-assistant
|         |____data            <- downloaded train and validation JSONL files
|         |____logs            <- training and validation logs
|         |____checkpoints     <- optional; disabled by default
|____.cache
     |____huggingface          <- Hugging Face and vLLM caches
     |____megatron_ckpt        <- converted Megatron checkpoint
```

Use a NeMo RL container built from the `super-v3.5-posttraining` branch, or a
compatible prebuilt image newer than v0.7. The mounted source and container
must use compatible revisions. This branch supplies the Super VL Megatron
model path, NeMo Gym integration, Qwen tool-call parser, and colocated weight
refit support required by the recipe.

The model uses about 235 GiB, its converted Megatron cache about 232 GiB, and
an ARM64 squashfs image about 73 GiB. Provision at least 650 GiB for these
artifacts, the Workplace Assistant data, logs, and working headroom.
Checkpointing is disabled. If enabled, allow about 227 GiB per weights-only
checkpoint or 1.4 TB per checkpoint that includes optimizer state.

The validated topology used nodes with at least 920 GiB of host memory. During
refit, NeMo RL temporarily moves optimizer state to CPU memory before loading
the current policy into vLLM. Request all memory on every node.

### Define host paths

**Run from any directory on the login or head node in the shell that will
submit Slurm jobs:**

```bash
export SHARED_ROOT=$(realpath </YOUR/SHARED/STORAGE>)
export NFS_ROOT=/lustre
export NEMO_RL="${SHARED_ROOT}/code/RL"
export NEMOTRON_REPO="${SHARED_ROOT}/code/Nemotron"
export MODEL_DIR="${SHARED_ROOT}/models/NVIDIA-Nemotron-3.5-Super-VL-09212026"
export CONTAINER=<NEMO_RL_CONTAINER_OR_SQUASHFS>
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export GPUS_PER_NODE=4
export MOUNTS="${NFS_ROOT}:${NFS_ROOT},${SHARED_ROOT}:/shared"
```

`NFS_ROOT` is the root of the site's shared filesystem namespace. The first
mount preserves host paths used by `ray.sub`; the second provides portable
`/shared` paths to commands running inside the container.

### Clone the source repositories

**Run from any directory on the login, head, or Docker build node:**

```bash
mkdir -p "${SHARED_ROOT}/code"
git clone --branch super-v3.5-posttraining --recursive \
  https://github.com/NVIDIA-NeMo/RL.git "${NEMO_RL}"
git clone https://github.com/NVIDIA-NeMo/Nemotron.git "${NEMOTRON_REPO}"

cd "${NEMO_RL}"
git submodule update --init --recursive
```

If the checkouts already exist, switch NeMo RL to
`super-v3.5-posttraining`, update it, and rerun the submodule command before
building the image.

### Build the container

No suitable post-v0.7 prebuilt image was available when this guide was
published. Build the image from the checkout that will be mounted into the
job.

**Run from the NeMo RL repository root on a Docker-capable ARM64 build node
with registry access:**

```bash
cd "${NEMO_RL}"
export IMAGE="<YOUR_REGISTRY>/nemo-rl:super-v3.5-posttraining-arm64"

docker buildx build \
  --platform linux/arm64 \
  --progress=plain \
  --push \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --target release \
  --build-arg MAX_JOBS=8 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS="examples/nemo_gym/prefetch_super35_all_envs.yaml" \
  -t "${IMAGE}" \
  .
```

`--build-context nemo-rl=.` makes the build use the current checkout. The
prefetch configuration creates the Workplace Assistant worker environments in
the image rather than during the first training job.

Clusters using Enroot or Pyxis can convert the registry image to squashfs.

**Run from the NeMo RL repository root on a node with Enroot and registry
access:**

```bash
cd "${NEMO_RL}"
export IMAGE="<YOUR_REGISTRY>/nemo-rl:super-v3.5-posttraining-arm64"
export CONTAINER="${SHARED_ROOT}/nemo-rl-super-v3.5-posttraining-arm64.sqsh"
enroot import -o "${CONTAINER}" "docker://${IMAGE}"
```

Use the registry URI directly when the cluster runtime supports it.

### Obtain the model

**Run from any directory on the login or head node:**

```bash
python3 -m pip install --user --upgrade "huggingface_hub[cli]"
export PATH="${HOME}/.local/bin:${PATH}"
hf auth login

mkdir -p "${MODEL_DIR}" "${SHARED_ROOT}/.cache/huggingface"
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-09212026 \
  --local-dir "${MODEL_DIR}"
```

Your Hugging Face account must have access to the model. Keep the model,
shared cache, and mounted NeMo RL checkout visible to every Ray worker.

## Interactive path

Use this path for the first run, when it is useful to inspect services and
training output inside the allocation.

### 1. Reserve four nodes

**Run from the NeMo RL repository root on the login or head node:**

```bash
cd "${NEMO_RL}"
unset COMMAND
sbatch \
  --nodes=4 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=super-vl-workplace \
  --time=06:00:00 \
  --gres=gpu:4 \
  --mem=0 \
  --exclusive \
  ray.sub
```

### 2. Attach to the Ray head

Wait until the allocation is running. `ray.sub` creates an attach helper in
the NeMo RL repository.

**Run from the NeMo RL repository root on the login or head node:**

```bash
cd "${NEMO_RL}"
bash ./<jobid>-attach.sh
```

### 3. Prepare data and train

The data command downloads the published rows, resolves the Workplace
Assistant agent and tool schemas, and writes NeMo Gym training JSONL. The
fixed validation subset bounds evaluation cost across dataset revisions.

**Run from the NeMo Gym repository inside the attached Ray-head container:**

```bash
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/super35-workplace-assistant
export DATA_DIR="${RUN_DIR}/data"
export CACHE_DIR=/shared/.cache
export GYM_ROOT="${NEMO_RL}/3rdparty/Gym-workspace/Gym"
export EXAMPLE_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-workplace-assistant-nemo-gym"

mkdir -p "${DATA_DIR}" "${RUN_DIR}/logs" \
  "${CACHE_DIR}/huggingface"/{modules,vllm} \
  "${CACHE_DIR}/megatron_ckpt/config_locks"

export HF_HOME="${CACHE_DIR}/huggingface"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_DIR}/megatron_ckpt"
export MEGATRON_CONFIG_LOCK_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}/config_locks"
export VLLM_CACHE_ROOT="${HF_HOME}/vllm"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

cd "${GYM_ROOT}"
uv run --extra dev gym dataset collate \
  --config responses_api_models/vllm_model/configs/vllm_model_for_training.yaml \
  --resources-server workplace_assistant \
  --output-dir "${DATA_DIR}" \
  --mode train_preparation \
  --download \
  +data_source=huggingface

python3 - "${DATA_DIR}/validation.jsonl" \
  "${DATA_DIR}/validation_balanced_40.jsonl" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
per_category = {}
selected = []
with open(source) as src:
    for line in src:
        row = json.loads(line)
        category = row["category"]
        if per_category.get(category, 0) < 8:
            selected.append(row)
            per_category[category] = per_category.get(category, 0) + 1

if len(selected) != 40 or any(count != 8 for count in per_category.values()):
    raise RuntimeError(f"expected 8 tasks from each of 5 categories: {per_category}")

with open(destination, "w") as dst:
    for row in selected:
        dst.write(json.dumps(row) + "\n")
PY

cd "${NEMO_RL}"
uv run --no-sync python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${EXAMPLE_DIR}/super_vl_3_5_workplace_assistant_megatron.yaml" \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  logger.log_dir="${RUN_DIR}/logs"
```

Append `logger.wandb_enabled=false` to the last command when external
experiment tracking is not desired. In an interactive shell, omit
`set -euo pipefail` so a failed exploratory command does not close the working
session.

## Batch path

For an unattended run, pass the same preparation and training work through
`COMMAND`. `ray.sub` writes the command into the job's log directory and runs
it on the Ray head.

**Run from the NeMo RL repository root on the login or head node in the shell
where the prerequisite variables are defined:**

```bash
cd "${NEMO_RL}"

read -r -d '' COMMAND <<'RUN' || true
set -euo pipefail
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/super35-workplace-assistant
export DATA_DIR="${RUN_DIR}/data"
export CACHE_DIR=/shared/.cache
export GYM_ROOT="${NEMO_RL}/3rdparty/Gym-workspace/Gym"
export EXAMPLE_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-workplace-assistant-nemo-gym"

mkdir -p "${DATA_DIR}" "${RUN_DIR}/logs" \
  "${CACHE_DIR}/huggingface"/{modules,vllm} \
  "${CACHE_DIR}/megatron_ckpt/config_locks"

export HF_HOME="${CACHE_DIR}/huggingface"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_DIR}/megatron_ckpt"
export MEGATRON_CONFIG_LOCK_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}/config_locks"
export VLLM_CACHE_ROOT="${HF_HOME}/vllm"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

cd "${GYM_ROOT}"
if [[ ! -s "${DATA_DIR}/train.jsonl" || ! -s "${DATA_DIR}/validation.jsonl" ]]; then
  uv run --extra dev gym dataset collate \
    --config responses_api_models/vllm_model/configs/vllm_model_for_training.yaml \
    --resources-server workplace_assistant \
    --output-dir "${DATA_DIR}" \
    --mode train_preparation \
    --download \
    +data_source=huggingface
fi
python3 - "${DATA_DIR}/validation.jsonl" \
  "${DATA_DIR}/validation_balanced_40.jsonl" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
per_category = {}
selected = []
with open(source) as src:
    for line in src:
        row = json.loads(line)
        category = row["category"]
        if per_category.get(category, 0) < 8:
            selected.append(row)
            per_category[category] = per_category.get(category, 0) + 1

if len(selected) != 40 or any(count != 8 for count in per_category.values()):
    raise RuntimeError(f"expected 8 tasks from each of 5 categories: {per_category}")

with open(destination, "w") as dst:
    for row in selected:
        dst.write(json.dumps(row) + "\n")
PY

cd "${NEMO_RL}"
exec uv run --no-sync python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${EXAMPLE_DIR}/super_vl_3_5_workplace_assistant_megatron.yaml" \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  logger.log_dir="${RUN_DIR}/logs"
RUN
export COMMAND

sbatch \
  --nodes=4 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=super-vl-workplace \
  --time=06:00:00 \
  --gres=gpu:4 \
  --mem=0 \
  --exclusive \
  ray.sub
```

### Monitor the batch job

**Run from the NeMo RL repository root on the login or head node:**

```bash
cd "${NEMO_RL}"
squeue -j <jobid> -o '%i %T %M %l %D %R'
tail -f <jobid>-logs/ray-driver.log
```

## Reading the result

The driver reports the mean binary reward across the fixed validation subset
before training, every five updates, and after step 20. It also reports rollout
reward, generation length, policy loss, step time, and refit timing. Because
reward is based on final state, validation reward is the fraction of tasks for
which the agent produced an equivalent workplace state.

A four-node convergence run completed all 20 full-weight updates with 32
prompts and eight generations per prompt. The same 40 held-out tasks were
evaluated at every measurement point:

| Policy | Correct | Accuracy | Change from baseline |
| --- | ---: | ---: | ---: |
| Before RL | 33/40 | 82.5% | — |
| Step 5 | 36/40 | 90.0% | +7.5 points |
| Step 10 | 36/40 | 90.0% | +7.5 points |
| Step 15 | 34/40 | 85.0% | +2.5 points |
| Step 20 | 35/40 | 87.5% | +5.0 points |

The best measured result was 36/40, while the final policy retained a modest
two-task improvement over baseline. With only 40 validation tasks, one answer
moves accuracy by 2.5 points, so the differences among later measurements are
noisy. The decline after step 10 also shows that higher rollout reward did not
consistently improve held-out accuracy. Select a stopping point with held-out
validation rather than training reward alone, and use a larger test set before
making a product-quality claim.

For predictable runtime, the published YAML uses one fixed 32-by-8 rollout
batch per update. The measured campaign used dynamic sampling during its first
five updates and fixed batches afterward, so treat the table as a sizing and
directional reference rather than an exact expected curve. The uninterrupted
step-5-to-step-20 segment completed in about 2 hours 43 minutes, including
model startup and three validations. Checkpointing was disabled for that
segment, so no step-20 weights were saved.

## Connect a Jira, Confluence, or custom harness

Workplace Assistant provides simulated project-management records; it does not
ship Jira or Confluence connectors. Use it as an architectural template:

1. Implement a NeMo Gym Resources Server that exposes the harness tools with
   JSON schemas and creates isolated state for each rollout.
2. Point a NeMo Gym agent configuration at that server and keep policy calls on
   the training policy model server. Calls made outside that path do not retain
   the token IDs and log probabilities needed for GRPO.
3. Create JSONL rows whose `responses_create_params` contain the request and
   tool schemas, whose `agent_ref` selects the new agent, and whose hidden
   fields contain the verifier inputs.
4. Implement a deterministic verifier that compares observable effects in a
   fresh environment. Score both requested changes and unintended side
   effects.
5. Replace the Workplace Assistant config path under
   `env.nemo_gym.config_paths` and update the data paths in the YAML.

For a real service, give every rollout a sandbox, disposable tenant, snapshot,
or deterministic reset. Return tool failures to the policy so it can recover,
and prevent the policy from reading reference actions or verifier-only state.
Start with tasks that have unambiguous outcomes, then add longer workflows,
error recovery, tool selection among similar APIs, and side-effect constraints.

A useful reward progression is:

- begin with final-state success as the primary binary reward;
- add narrowly defined penalties for invalid tool calls or forbidden effects;
- avoid rewarding a specific call sequence when several sequences are valid;
- measure success by task category and trajectory length, not only the global
  mean; and
- retain a held-out set of workflows and entities to detect memorization.

## Troubleshooting

| Symptom | Check and action |
| --- | --- |
| A Gym worker environment is missing | Build with `prefetch_super35_all_envs.yaml`, or set `NRL_FORCE_REBUILD_VENVS=true` for one launch. |
| Dataset collation cannot find Workplace Assistant | Run it from `${NEMO_RL}/3rdparty/Gym-workspace/Gym` and initialize the Gym submodule. |
| Tool calls appear as plain text | Keep `enable_auto_tools: true`, `tool_parser: qwen3_coder`, and the model chat template from the recipe. |
| A trajectory reaches the context limit | Keep the 16,384-token limit, reduce the agent horizon, or shorten tool results. Do not silently drop overlength samples. |
| Megatron workers cannot import `transformers_modules` | Put the shared `HF_MODULES_CACHE` on `PYTHONPATH` and launch from the mounted NeMo RL checkout. |
| Model conversion repeats | Point `NRL_MEGATRON_CHECKPOINT_DIR` to persistent shared storage visible to every node. |
| vLLM runs out of memory during refit | Preserve TP=4, optimizer offload during refit, and the recipe's colocated memory settings. |
| Reward is always zero | Inspect tool parsing, tool errors, agent horizon, hidden verifier inputs, and whether each rollout starts from the expected state. |
| GRPO loss is zero | Check reward diversity within each group of eight trajectories; leave-one-out advantages are zero when every reward matches. |
| Validation takes too long | Use the fixed `validation_balanced_40.jsonl` subset or choose a smaller explicit subset for smoke testing. |
