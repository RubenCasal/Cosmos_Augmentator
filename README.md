# cosmos_augmentor

Automates semantic-segmentation dataset augmentation with NVIDIA Cosmos-Transfer2.5.

## Current Status (2026-03-18)

- Core pipeline is modular and operational (`config -> scan -> augment -> merge`).
- Per-image timing and run summaries are implemented.
- Per-image failures are isolated (the run continues and reports errors at the end).
- Semantic masks support both input modes:
  - colorized RGB masks (`colorize=true`)
  - class-id masks (`colorize=false`) with optional on-the-fly conversion to RGB for Cosmos compatibility

## Prerequisites

- Python 3.10+
- NVIDIA GPU with enough VRAM for your selected Cosmos model variant
- A working local Cosmos-Transfer2.5 checkout (for example `/home/upia/rcasal/cosmos-transfer2.5`)
- Cosmos examples already working in your environment (for example `python examples/inference.py -i ...`)

## Dataset layout

```text
dataset_ws/
  .
    images/
    labels/
```

Labels are assumed to be valid ground truth for augmented images (no re-segmentation step is performed).

## Configuration

Edit `config/augmentations.yaml`:

- `cosmos.repo_root`: existing local Cosmos repo path
- `cosmos.model` (optional, recommended): exact model id/name expected by your Cosmos version
- `cosmos.use_edge_control`, `cosmos.edge_control_weight`, `cosmos.seg_control_weight`: control setup to match Cosmos examples (`edge` + `seg`)
- `dataset.root`: workspace containing your dataset folders
- `augmentations`: list of profiles (`prompt`, `negative_prompt`, `fraction`, `seed_base`, `output_dir`)
- `logging` (optional):
  - `level`: log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
  - `file`: optional output file path for logs
- `dataset.segmentation` (optional):
  - `encoding`: `rgb` or `id`
  - `convert_ids_to_rgb_for_cosmos`: when `true` and `encoding=id`, converts id masks to RGB for Cosmos
  - `converted_cache_dir`: cache folder under `dataset.root` for converted masks

Example for class-id masks (`colorize=false`):

```yaml
dataset:
  root: "/path/to/dataset"
  original_dir: "."
  image_subdir: "images"
  label_subdir: "labels"
  image_ext: ".png"
  segmentation:
    encoding: "id"
    convert_ids_to_rgb_for_cosmos: true
    converted_cache_dir: ".cosmos_seg_rgb_cache"
```

## Install

```bash
pip install -e .
```

## Run

Run full pipeline (augment + merge):

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml run-all
```

Override log options from CLI:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml --log-level INFO --log-file logs/run.log run-all
```

Or helper script:

```bash
./scripts/run_augmentation.sh
```

Other commands:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml run-augmentations
python -m cosmos_augmentor.cli --config config/augmentations.yaml merge
```

## Output

Per-augmentation datasets are created under `dataset_<name>/images` and `dataset_<name>/labels`, then merged into:

```text
dataset_ws/
  complete_dataset/
    images/
    labels/
```

Merged filenames are prefixed safely (`orig_...`, `<augmentation_name>_...`).

## Timing Logs

During augmentation, the tool logs:

- per-image generation time, for example: `[img 12/340][sunset] 1.83s | frame_0012.png`
- per-augmentation summary with average and total time, for example: `[augmentation sunset] success=340 failed=0 avg=1.76s total=09m58s`
- final run summary with global totals and timing
- failed-image traces (without aborting the entire run)

## Notes

- Converted RGB masks are only used as Cosmos control input when needed.
- Original labels are preserved and copied unchanged to augmented datasets.
