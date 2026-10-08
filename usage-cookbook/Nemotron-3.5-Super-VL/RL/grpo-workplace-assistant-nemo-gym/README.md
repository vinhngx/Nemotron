# Workplace Assistant GRPO with NeMo Gym

This recipe applies full-weight GRPO to Nemotron 3.5 Super VL on NeMo Gym's
Workplace Assistant environment. The environment presents 27 tools for email,
calendar, project management, CRM, analytics, and directory lookup. Its
`simple_agent` runs as many as six model and tool steps, and a deterministic
verifier compares the resulting workplace state with the requested outcome.

This is a useful starting point for agentic RL because the agent loop, tools,
state, and reward have separate interfaces. A later environment can replace
the simulated project-management tools with a resettable Jira or Confluence
harness while retaining the NeMo RL training path.

## Requirements

- NeMo RL branch `super-v3.5-posttraining`, including its Gym submodule
- A NeMo RL container built from that branch
- Nemotron 3.5 Super VL in Hugging Face format
- Four nodes with four GPUs and at least 920 GiB host memory per node
- Shared storage mounted at `/shared`

The recipe uses the same full-weight BF16, TP4/EP16, colocated vLLM topology as
the validated Super VL star-count cookbook. See the parent
[RL README](../README.md) for container, model, storage, and Slurm setup.

## Prepare the Workplace Assistant data

From the NeMo Gym checkout, download and prepare the public dataset. Set a
Hugging Face token when your environment requires authenticated downloads:

```bash
cd "${NEMO_RL}/3rdparty/Gym-workspace/Gym"

uv run --extra dev gym dataset collate \
  --config responses_api_models/vllm_model/configs/vllm_model_for_training.yaml \
  --resources-server workplace_assistant \
  --output-dir /shared/runs/super35-workplace-assistant/data \
  --mode train_preparation \
  --download \
  +data_source=huggingface
```

The command creates `train.jsonl` and `validation.jsonl`. Dataset sizes can
change as the published dataset is revised; the snapshot used during this
cookbook's initial validation contained 1,255 training tasks and 545
validation tasks.

## Run GRPO

```bash
cd "${NEMO_RL}"

python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config /shared/code/Nemotron/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-workplace-assistant-nemo-gym/super_vl_3_5_workplace_assistant_megatron.yaml
```

The default schedule uses eight tasks and eight sampled trajectories per
task, for 64 trajectories per optimizer update. It evaluates the complete
held-out split before training, every five updates, and after the final
update. The 16,384-token trajectory budget accommodates the 27 tool schemas,
reasoning, tool results, and as many as six agent steps. Checkpointing is
disabled because a full optimizer checkpoint is
approximately 1.4 TB; enable it only after provisioning sufficient storage.

## What the reward measures

The verifier extracts the agent's executed function calls, applies the
reference actions in a separate fresh environment, and compares all relevant
state tables. A correct final state receives reward `1`; any mismatch receives
reward `0`. Equivalent tool sequences can receive the same reward because the
comparison is based on the resulting state.

## Adapting the environment

Keep the policy calls routed through the NeMo Gym policy model server so each
trainable turn retains its token IDs and generation log probabilities. Replace
the Resources Server when connecting another tool harness, and preserve these
properties:

- create isolated state for every rollout;
- return tool failures to the model so it can recover;
- verify against a fresh reference state;
- prevent the agent from reading verifier-only metadata; and
- score requested effects and unintended side effects.

Use a sandbox, simulator, or resettable tenant for systems that can mutate
external state.
