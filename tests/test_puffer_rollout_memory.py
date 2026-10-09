"""The production gather index must reproduce transpose-and-slice bitwise."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from integrations.puffer_rollout_memory import install


class MinibatchGatherTests(unittest.TestCase):
    def test_unknown_source_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            path = root / 'src/pufferl.cu'
            path.write_text('unknown upstream')
            with self.assertRaises(ValueError):
                install(root)
            self.assertEqual(path.read_text(), 'unknown upstream')
            self.assertFalse((root / 'src/metta_rollout_memory.cuh').exists())

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler unavailable')
    def test_real_index_observation_bits_binary_masks_and_replay_order(self):
        header = Path(__file__).resolve().parents[1] / 'integrations/puffer_rollout_memory.cuh'
        code = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __host__
#define __device__
#include "HEADER"
int main() {
  for (int T : {8, 128, 256}) for (int B : {64, 128, 4096}) for (int C : {1, 7, 3529, 7056}) {
    if (B > 128 && C > 7) continue;
    std::vector<uint32_t> observations((int64_t)T*B*C);
    std::vector<unsigned char> masks(observations.size());
    for (int64_t i=0;i<(int64_t)observations.size();++i) {
      observations[i] = (uint32_t)(i*2654435761ULL); // Includes arbitrary float bits.
      masks[i] = (unsigned char)(i%2);
    }
    int N=64;
    // Repeated minibatches cover destination offsets and replay wraparound.
    for (int mb=0;mb<2*B/N;++mb) {
      int off=(mb*N)%B;
      for (int64_t i=0;i<(int64_t)N*T*C;++i) {
        int a=off+i/(T*C), t=(i/C)%T, f=i%C;
        int64_t reference=((int64_t)t*B+a)*C+f;
        int64_t gathered=metta_rollout_index(i,T,B,C,off);
        assert(reference==gathered);
        float old_value,new_value;
        memcpy(&old_value,&observations[reference],4);
        memcpy(&new_value,&observations[gathered],4);
        assert(memcmp(&old_value,&new_value,4)==0);
        float old_mask=(float)masks[reference],new_mask=(float)masks[gathered];
        assert(memcmp(&old_mask,&new_mask,4)==0);
      }
    }
  }
  // Full rollout strides can exceed signed int; production helper uses int64.
  assert(metta_rollout_index(8191LL*256*7056+255LL*7056+7055,
                             256,8192,7056,0)==8192LL*256*7056-1);
}
'''.replace('HEADER', str(header))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'proof.cc'
            binary = Path(directory) / 'proof'
            source.write_text(code)
            subprocess.run(['c++', '-std=c++17', '-O2', str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
