#!/usr/bin/env python
"""Run active learning experiments training a CNN each round.

This script is a thin wrapper around the full-loop simulation located in
``src/active_learning/simulate.py``.  It exposes the same command line
options but makes it easy to launch the incremental-CNN AL run from the
``scripts/`` directory (parallel to ``run_al_experiments.py``).

Example usage:

```powershell
python scripts/run_al_cnn.py \
    --rounds 20 --query-size 10 --strategy margin --diversity \
    --data-root data/synth_surface_defects \
    --metrics-dir artifacts/active_learning/myrun
```

The simulation will train a fresh ``SurfaceDefectNet`` model each
round (optionally warm‑started from a checkpoint) and save per‑round metrics,
query selections, and a run summary.
"""

import sys
from pathlib import Path

# ensure workspace root is on Python path (mirrors other top‑level scripts)
root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.active_learning import simulate


if __name__ == "__main__":
    simulate.main()
