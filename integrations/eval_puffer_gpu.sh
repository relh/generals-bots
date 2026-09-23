#!/usr/bin/env bash
set -euo pipefail

run_directory=${1:?run directory required}
build_directory=${2:?build directory required}
output_directory=${3:?output directory required}
evaluation_config=${4:?evaluation config required}

checkpoint=$(python - "$run_directory" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
print(run / json.loads((run / "completed.json").read_text())["final_checkpoint"])
PY
)

exec bash /work/source-gpu/integrations/run_puffer_gpu.sh evaluate \
    --config "$evaluation_config" \
    --build "$build_directory" \
    --run "$run_directory" \
    --checkpoint "$checkpoint" \
    --output "$output_directory"
