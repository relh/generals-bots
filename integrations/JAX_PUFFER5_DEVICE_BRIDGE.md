# JAX Classic environment to Puffer5 GPU bridge

The full Classic JAX step is fast while state and actions stay on the B300:
job 16850 measured a 1.613 ms median over 1,300 turns and 4,096 games,
including a truncation and recycle. The current Puffer5 CPU environment path
takes 96.44 ms per 4,096-game profile step with native serialization, before
policy inference or optimization (job 16823). The clean trainer completed at
34,148 SPS including its final checkpoint (job 16788). These are distinct
measurements; only the last is end-to-end training SPS.

The pinned Puffer5 source at revision `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`
already calls `puf_step` after model inference in each GPU rollout step. Its
GPU path requires one vector buffer and passes device pointers for
observations, actions, rewards and terminals to `puf_vec_create`. The
action-mask buffer is allocated but is not passed to the GPU environment.
Generals needs that mask for legal move sampling.

Metta's native Fabric policy already wraps Puffer-owned CUDA buffers as
DLPack tensors in `native/fabric.cuh`, and `NativeFabricPolicy.forward_device`
imports them into JAX. A separate B300 probe (job 16874) confirmed that a
spawned JAX worker can donate and update a parent-owned 25,288,704-float
CUDA buffer through IPC and DLPack. This is a feasibility result, not an
integrated trainer.

The implementation branch is `relh/generals-jax-device-bridge` in the
separate metta worktree `/Users/relh/Code/worktrees/metta-generals-device-bridge`,
based on the tested native transport commit `f93c027e03`. Complete these
pieces together before claiming a bridge:

1. Add an explicit device-resident Python environment configuration. Permit
   it only for bounded CUDA training with compatible observation/action
   layouts; keep current CPU evaluation intact. Generate a `.cu` environment
   header that selects `PUF_GPU` and uses one vector buffer.
2. Extend the pinned Puffer5 driver for this build only to pass its action-mask
   device pointer to the GPU environment. Guard the source patch with the
   existing upstream hash and exact text checks.
3. Add a GPU environment adapter that passes Puffer's action buffer to JAX
   through DLPack and writes observations, legal masks, rewards and terminal
   flags into the corresponding Puffer device buffers. Synchronize the
   Puffer and JAX streams. Disable CUDA graph capture until a graph-safe
   callback is proven.
4. Add `reset_device` and `step_device` to the Generals JAX adapter using
   the existing `_advance_states` and `_observe_states` paths. Preserve full
   18–21-tile Classic rules, masks, per-agent rewards and recycle behavior.
5. Check reset/step/action-mask parity against the existing CPU bridge on
   fixed seeds. Then run a bounded B300 Puffer5 training pilot through a
   clean final checkpoint. Report warm completed-step SPS including model
   inference and updates, GPU use, environment count and batch settings.

Environment state checkpointing and restore need an explicit design for this
new backend. Do not report a training gate pass from a crashed or
policy-only incomplete run. Strong held-out play is a separate gate: the
latest pure PPO checkpoint scored .007568 against ExpanderHarvester and zero
against Sentinel, so a long run on that learning objective is not justified.
