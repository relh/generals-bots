"""Activate audited hooks in both Python launchers and Puffer's embedded Python."""
import os
import sys

if os.environ.get("METTA_MEMORYLESS_OPTIMIZATION") == "1":
    try:
        if os.environ.get("GENERALS_AUDIT_CUDA_RUNTIME") == "1":
            from integrations.cuda_runtime_binding import audit
            audit()
        from integrations.activate_memoryless_optimization import activate
        activate("integrations.direct_spatial_optimization")
        if os.environ.get("METTA_AUDIT_DEVICE_REWARDS") == "1":
            from integrations.environment_reward_audit import activate as activate_rewards
            activate_rewards()
        # PID binding prevents a child from trusting its parent's inherited marker.
        os.environ["METTA_SPATIAL_BOOTSTRAP_PID"] = str(os.getpid())
        print("SPATIAL_BOOTSTRAP_READY pid=" + str(os.getpid()), flush=True)
    except Exception as error:
        print("SPATIAL_BOOTSTRAP_FAILED " + type(error).__name__, file=sys.stderr, flush=True)
        # Python normally swallows sitecustomize exceptions. Training must not
        # continue with generic Fabric semantics after a failed adapter install.
        os._exit(78)
