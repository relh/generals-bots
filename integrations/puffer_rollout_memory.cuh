// Exact agent-major minibatch view of a time-major rollout; no arithmetic on observations.
__host__ __device__ static inline int64_t metta_rollout_index(
    int64_t i, int T, int B, int C, int agent_offset) {
    int64_t feature = i % C;
    int64_t row = i / C;
    return ((row % T) * B + agent_offset + row / T) * C + feature;
}
#ifdef __CUDACC__
template <typename Source>
__global__ void metta_gather_rollout(precision_t* dst, const Source* src,
    int T, int B, int C, int agent_offset, int64_t count) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = (precision_t)src[metta_rollout_index(i, T, B, C, agent_offset)];
}
#endif
