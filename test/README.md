# Offline Block Detection Tests

This folder contains standalone detection experiments that run on saved images
from `logs/` before any method is wired into the ROS agent.

## Color-Independent Proposal Detector

This is the preferred experiment for the robot pipeline. It proposes all
tabletop block-like foreground objects without using a fixed target color.
The next step can ask a VLM to choose one proposal ID rather than asking it to
invent pixel coordinates.

```bash
python3 test/propose_blocks.py \
  --images "logs/vlm_agent/*/step_001_raw.jpg" \
  --out-dir test/output_proposals
```

## Deprecated Single-Color Baseline

```bash
python3 test/detect_block.py \
  --images "logs/vlm_agent/*/step_001_raw.jpg" \
  --color green \
  --out-dir test/output
```

This older script is only a small baseline for one selected color. It should
not be wired into the robot pipeline.
