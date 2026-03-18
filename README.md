# cosmos_augmentor

Refactored dataset augmentation pipeline for NVIDIA Cosmos-Transfer2.5 with fully configurable control modalities.

## What This Refactor Adds

- Independent control configuration for `seg`, `depth`, and `edge`
- Per-control mode selection: `disabled`, `external`, `on_the_fly`
- External control validation only when required
- Separate `input_root` and `output_root`
- Clear module separation:
  - config loading and validation
  - dataset scanning
  - job creation
  - payload creation
  - Cosmos execution
  - result materialization
- Per-image timing and fault-tolerant execution (failures do not abort the whole run)

## Dataset Layout

The input dataset may contain:

```text
<input_root>/<original_dir>/
  images/
  labels/
  depth/
  edges/
```

Only `images/` is always required.

`labels/`, `depth/`, and `edges/` are required only when the corresponding control mode is `external`.

## Configuration

Example structure:

```yaml
cosmos:
  repo_root: "/path/to/cosmos-transfer2.5"
  model: "seg"
  model_variant: "edge"
  model_distilled: false
  disable_guardrails: true
  resolution: "720"
  guidance: 2.0
  num_steps: 24
  max_frames: 1
  num_video_frames_per_chunk: 1

  controls:
    seg:
      mode: external      # disabled | external | on_the_fly
      weight: 1.0
      subdir: "labels"
      encoding: "rgb"     # rgb (colorize=true) | id (colorize=false)
    depth:
      mode: on_the_fly    # disabled | external | on_the_fly
      weight: 0.9
      subdir: "depth"
    edge:
      mode: disabled      # disabled | external | on_the_fly
      weight: 1.0
      subdir: "edges"

dataset:
  input_root: "/path/to/input_dataset"
  output_root: "/path/to/output_dataset"
  original_dir: "."
  image_subdir: "images"
  label_subdir: "labels"
  image_ext: ".png"
  cache_dir: ".cosmos_control_cache"

augmentations:
  - name: "sunset"
    output_dir: "dataset_sunset"
    fraction: 1.0
    seed_base: 400
    prompt: >
      ...
    negative_prompt: >
      ...
```

## Control Behavior

For each control (`seg`, `depth`, `edge`):

- `external`:
  - Requires `<input_root>/<original_dir>/<subdir>`
  - Requires a matching file per image
  - Sends `control_path` + `control_weight` to Cosmos
- `on_the_fly`:
  - Does not require a dataset folder
  - Sends control block with `control_weight` and no `control_path`
- `disabled`:
  - Not used
  - Omitted from Cosmos payload

## Installation

```bash
pip install -e .
```

## Run

Run augmentations and merge:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml run-all
```

Run only augmentation stage:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml run-augmentations
```

Run only merge stage:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml merge
```

Optional logging overrides:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml --log-level INFO --log-file logs/run.log run-all
```

## Output

Per augmentation, generated files are written under:

```text
<output_root>/<augmentation.output_dir>/
  images/
  <external control subdirs that apply>
```

Merged output is written to:

```text
<output_root>/complete_dataset/
  images/
  <external control subdirs that apply>
```

## Error Handling

- Invalid config values raise clear `ConfigError`
- Missing required dataset/control files raise `DatasetScanError`
- Cosmos failures raise `CosmosRunnerError` with subprocess stdout/stderr excerpts
