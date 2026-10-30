// Match Safa's minibatch actor normalization after all advantage producers run.
__global__ void metta_standardize_ppo_advantages(precision_t *advantages, int count) {
  __shared__ float sums[256], squares[256], mean, scale;
  int thread = threadIdx.x;
  float sum = 0.0f, square = 0.0f;
  for (int index = thread; index < count; index += blockDim.x) {
    float value = to_float(advantages[index]);
    sum += value;
    square += value * value;
  }
  sums[thread] = sum;
  squares[thread] = square;
  __syncthreads();
  for (int width = blockDim.x / 2; width; width /= 2) {
    if (thread < width) {
      sums[thread] += sums[thread + width];
      squares[thread] += squares[thread + width];
    }
    __syncthreads();
  }
  if (thread == 0) {
    mean = sums[0] / count;
    float variance = fmaxf((squares[0] - count * mean * mean) / (count - 1), 0.0f);
    scale = sqrtf(variance) + 1e-8f;
  }
  __syncthreads();
  for (int index = thread; index < count; index += blockDim.x)
    advantages[index] = from_float((to_float(advantages[index]) - mean) / scale);
}
