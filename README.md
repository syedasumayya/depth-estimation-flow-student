# Flow-conditioned correction of Depth Anything V2 (VKITTI2)

A ~30K-parameter correction network that takes the output of Depth Anything V2 (Small) and optical flow and predicts a depth correction.
Evaluated with leave-one-scene-out testing on VKITTI2 (clone variation, left camera).

## Contents
- `src/` - dataset, masks, model, loss, evaluation, fold runner (`foldrun.py`), data restore script (`restore.py`).
  `train.py` is an early version, superseded by `foldrun.py`.
- `results/` - result tables (CSV) and figures. `results/folds/` holds one file per variant, held-out scene and seed.

## Notes
- The dataset and caches are not included. Download VKITTI2 from its official page and check its license.
  Depth Anything V2 is used through Hugging Face (`depth-anything/Depth-Anything-V2-Small-hf`); check its license too.
- Paths are hard-coded for Google Colab (`/content/drive/MyDrive/depth_project`).
- Some analysis steps (fair baselines, summary tables, charts, depth pictures) were run as notebook cells and are not in `src/` yet.
- Tested on Colab (T4 GPU, torch 2.11, opencv 5.0). Results are from one or two training runs per scene and are not tuned.
