# Agent guidance

Training work must prioritize GPU execution. Follow the PufferLib throughput
paradigm: run many parallel environments and batch inference and optimization
so the accelerator stays fed. Do not accept or continue sustained training
below 30,000 environment steps per second (SPS). Treat lower throughput as a
performance bug: profile the bottleneck and tune parallelism, batching, or data
movement first. Report the hardware, environment count, batch settings, warmup,
and measured steady-state end-to-end training SPS.
