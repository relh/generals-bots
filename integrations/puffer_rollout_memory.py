"""Remove duplicate full observation/mask transposes; retain float32 rollout storage."""
import hashlib
import json
from pathlib import Path

def install(source):
    path = source / "src/pufferl.cu"
    original = path.read_text()
    text = original
    replacements = [('int T, int B, int input_size, int num_atns, int mask_size) {',
      'int T, int B, int input_size, int num_atns, int mask_size, bool dense = true) {'),
     ('        &bufs->observations, &bufs->values, &bufs->logprobs,\n'
      '        &bufs->rewards, &bufs->terminals, &bufs->action_mask,',
      '        &bufs->values, &bufs->logprobs, &bufs->rewards, &bufs->terminals,'),
     ('    transpose_102<<<grid_size((int64_t)T * B * obs_size), BLOCK_SIZE, 0, stream>>>(\n'
      '        rollouts->observations.data, src.observations.data, T, B, obs_size);',
      ''),
     ('    transpose_102<<<grid_size((int64_t)T * B * mask_c), BLOCK_SIZE, 0, stream>>>(\n'
      '        rollouts->action_mask.data, src.action_mask.data, T, B, mask_c);',
      ''),
     ('static void train_epoch_gpu(', '#include "metta_rollout_memory.cuh"\nstatic void train_epoch_gpu('),
     ('        graph.mb_obs = slice_rows(rollouts->observations, dest_off, Nmb);',
      '        graph.mb_obs = pufferl->minibatch_observations;\n'
      '        metta_gather_rollout<<<grid_size((int64_t)Nmb * T * obs_size), BLOCK_SIZE, 0, stream>>>(\n'
      '            graph.mb_obs.data, src.observations.data, T, B, obs_size, dest_off, (int64_t)Nmb * T * '
      'obs_size);'),
     ('        graph.mb_action_mask = slice_rows(rollouts->action_mask, dest_off, Nmb);',
      '        graph.mb_action_mask = pufferl->minibatch_action_mask;\n'
      '        metta_gather_rollout<<<grid_size((int64_t)Nmb * T * mask_c), BLOCK_SIZE, 0, stream>>>(\n'
      '            graph.mb_action_mask.data, src.action_mask.data, T, B, mask_c, dest_off, (int64_t)Nmb * T * '
      'mask_c);'),
     ('    alloc_register(alloc, &bufs->actions);',
      '    alloc_register(alloc, &bufs->actions);\n'
      '    if (dense) {\n'
      '        alloc_register(alloc, &bufs->observations);\n'
      '        alloc_register(alloc, &bufs->action_mask);\n'
      '    }'),
     ('    EnvBuf env;\n    TrainGraph train_buf;',
      '    Prec minibatch_observations;\n'
      '    Prec minibatch_action_mask;\n'
      '    EnvBuf env;\n'
      '    TrainGraph train_buf;'),
     ('        acts, total_agents, horizon, input_size, num_action_heads, act_n);',
      '        acts, total_agents, horizon, input_size, num_action_heads, act_n, false);\n'
      '    if (hypers.async) { fprintf(stderr, "Minibatch gather requires synchronous PPO\\n"); abort(); }\n'
      '    if (hypers.cudagraphs) { fprintf(stderr, "Minibatch gather requires CUDA graphs disabled\\n"); '
      'abort(); }\n'
      '    if (USE_BF16) { fprintf(stderr, "Minibatch gather requires float32\\n"); abort(); }\n'
      '#ifdef METTA_FABRIC\n'
      '    if (metta_requires_reward_centering) { fprintf(stderr, "Minibatch gather cannot center rewards\\n"); '
      'abort(); }\n'
      '#endif\n'
      '    pufferl->minibatch_observations = {.shape = {minibatch_segments, horizon, input_size}};\n'
      '    pufferl->minibatch_action_mask = {.shape = {minibatch_segments, horizon, act_n}};\n'
      '    alloc_register(acts, &pufferl->minibatch_observations);\n'
      '    alloc_register(acts, &pufferl->minibatch_action_mask);')]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError("Minibatch gather source anchor changed: " + old[:90])
        text = text.replace(old, new)
    header = Path(__file__).with_suffix('.cuh').read_bytes()
    (source / 'src/metta_rollout_memory.cuh').write_bytes(header)
    path.write_text(text)
    receipt = dict(original_pufferl_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   patched_pufferl_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   gather_header_sha256=hashlib.sha256(header).hexdigest(),
                   installer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (source.parent / 'rollout-memory.json').write_text(json.dumps(receipt, indent=2)+'\n')


def validate(build):
    """Bind the gather installer, generated source and kernel to the retained build."""
    build = Path(build)
    receipt = json.loads((build / 'rollout-memory.json').read_text())
    if receipt['gather_header_sha256'] != hashlib.sha256(Path(__file__).with_suffix('.cuh').read_bytes()).hexdigest():
        raise ValueError('Minibatch gather kernel differs from current source')
    for key, path in (('patched_pufferl_sha256', build/'source/src/pufferl.cu'),
                      ('gather_header_sha256', build/'source/src/metta_rollout_memory.cuh'),
                      ('installer_sha256', Path(__file__))):
        if receipt[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Minibatch gather build receipt differs: ' + key)
