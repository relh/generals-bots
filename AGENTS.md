# Generals training agent guidance

## GPU throughput gate

- Run training and evaluation Slurm jobs on B200 or B300 GPUs. Do not submit CPU-only Slurm jobs or use metta4.
- Before any new long training experiment, demonstrate at least **30,000 sustained environment steps per second (SPS)** for the proposed training setup on one allocated GPU. Measure completed Puffer agent steps divided by wall-clock time over a steady-state interval after JAX compilation, including rollout, host/device transfer, and model updates. Report the interval, step counts, GPU model, GPU utilization, environment count, and per-process and aggregate SPS.
- A short GPU smoke or profiling job may run below the gate to diagnose and improve throughput. Limit these jobs to a bounded step or wall-clock budget, preserve useful output, and release the allocation promptly. Do not start or release a dependent long job merely because GPU utilization is high; throughput must pass the gate.
- Before a long recurrent-policy run, make the GPU probe pass the step range where prior runs failed from nonfinite Fabric gradients (about 2.6 million steps per trainer). Disable core dumps in the container so a native abort releases the GPU promptly, and stop a run when `NonFiniteGradsError` appears. Do not count steps from a crashed trainer toward sustained SPS.
- Stop a GPU smoke promptly if it produces no training steps and the GPU stays effectively idle after the bounded startup window. Check for Docker containers and trainer processes left behind by `scancel`, and stop only those belonging to the canceled job.
- During a Puffer run, read completed steps from its console dashboard or native metrics file. The JSON run monitor can remain at zero until the subprocess exits; do not use that field alone to classify a live trainer as stalled.
- Profile rollout time separately from optimization and investigate JAX synchronization, numeric observation encoding, teacher targets, Python list conversion, and Puffer vector environment configuration. Increase parallel games and process count only when an end-to-end measurement confirms an improvement.
- Preserve and verify existing checkpoints before canceling or replacing a run. Keep dependent long jobs held until the 30K SPS gate passes. After throughput is fixed, continue training for tens of millions of steps and evaluate a selected policy on held-out 10×10 classic maps against mixed scripted opponents; success requires at least 0.60 performance.

## Cluster storage

- Check live Slurm reservations and physical GPU use before submission. Reserve only proportional CPU and memory, with a bounded time and identifiable job name.
- Move finished checkpoints and logs to private `s3://softmax-slurm-artifacts/<user>/<job>/` using [Metta's transfer procedure](https://github.com/Metta-AI/metta/blob/main/docs/ai/onboarding/services/slurm.md#move-files-in-and-out). Do not give jobs AWS credentials.
- Split presigned uploads above 5 GB into 4 GB parts. Verify S3 objects before removing local copies; the bucket expires objects after 90 days.
- Keep only active outputs on node-local disks. Never delete another investigator's data or protected agent history during cleanup.
