# Configuration Reference

This repository is configured through a YAML file, typically `config/augmentations.yaml`. The configuration is split into four main sections:

- `cosmos`
- `dataset`
- `augmentations`
- `logging`

## `cosmos`

### `repo_root`

Absolute path to the local `cosmos-transfer2.5` repository.

Example:

```yaml
repo_root: "/path/to/cosmos-transfer2.5"
```

### `model`

Optional model name passed to Cosmos. Leave unset if your local Cosmos setup already resolves the desired model through its defaults.

### `model_variant`

Model variant used by Cosmos.

Typical values:

```yaml
model_variant: "edge"
```

### `model_distilled`

Boolean flag that enables the distilled model variant when available.

Allowed values:

- `true`
- `false`

### `disable_guardrails`

Boolean flag to disable Cosmos guardrails.

Allowed values:

- `true`
- `false`

### `resolution`

Target resolution string passed to Cosmos.

Example:

```yaml
resolution: "720"
```

### `guidance`

Guidance scale passed to Cosmos.

Validation in this repository:

- Numeric value

Practical usage:

- Commonly tuned around `2.0` to `6.0`

### `num_steps`

Number of denoising or inference steps.

Validation in this repository:

- Integer value

Practical usage:

- Often tuned in the `20` to `40` range depending on the dataset and desired fidelity

### `max_frames`

Maximum number of frames used by Cosmos.

Validation in this repository:

- Integer value

Recommended for this project:

- `1`, because this repository is currently focused on image augmentation workflows

### `num_video_frames_per_chunk`

Chunk size passed to Cosmos for generation.

Validation in this repository:

- Integer value

Recommended for this project:

- `1`, for image-only workflows

## `cosmos.controls`

The repository supports three control modalities:

- `seg`
- `depth`
- `edge`

Each control block supports the same structure.

### `mode`

Defines how the control is provided.

Allowed values:

- `disabled`
- `external`
- `on_the_fly`

Behavior:

- `disabled`: the control is not sent to Cosmos
- `external`: the control is read from the dataset folders
- `on_the_fly`: Cosmos generates the control internally

### `weight`

Control strength.

Validated range:

- `0.0` to `1.0`

Important note:

- If multiple control weights sum to more than `1.0`, Cosmos may normalize them internally

### `subdir`

Folder name used when the control mode is `external`.

Typical defaults:

- `seg`: `labels`
- `depth`: `depth`
- `edge`: `edges`

### `encoding`

Currently relevant for `seg`.

Allowed values for `seg`:

- `rgb`
- `id`

Meaning:

- `rgb`: semantic segmentation masks are colorized
- `id`: semantic segmentation masks store class ids directly

For depth and edge controls, the repository internally adapts the input to a Cosmos-compatible RGB representation when needed.

## `dataset`

### `input_root`

Absolute path to the source dataset root.

### `output_root`

Absolute path where augmentation outputs and the merged dataset are written.

This does not need to be the same as `input_root`.

### `original_dir`

Relative subdirectory inside `input_root` containing the original dataset.

Typical value:

```yaml
original_dir: "."
```

### `image_subdir`

Folder containing the original RGB images.

Typical value:

```yaml
image_subdir: "images"
```

### `label_subdir`

Folder containing the ground-truth semantic segmentation labels.

Typical value:

```yaml
label_subdir: "labels"
```

### `image_ext`

Image extension used to match dataset files.

Typical value:

```yaml
image_ext: ".png"
```

### `cache_dir`

Folder inside `output_root` used for internally adapted control inputs, such as depth converted to Cosmos-compatible RGB.

Typical value:

```yaml
cache_dir: ".cosmos_control_cache"
```

## `augmentations`

This section is a list. Each entry defines one augmentation profile.

### `name`

Logical name of the augmentation.

Example:

```yaml
name: "snow"
```

### `output_dir`

Name of the directory created under `output_root` for this augmentation.

### `fraction`

Fraction of the input dataset to sample for this augmentation.

Validated range:

- `0.0` to `1.0`

Examples:

- `1.0`: use the full dataset
- `0.25`: use 25 percent of the dataset

### `seed_base`

Base integer seed used to derive deterministic seeds for each selected image.

Validation in this repository:

- Integer value

### `prompt`

Positive prompt used to drive the augmentation.

Validation in this repository:

- Non-empty string

### `negative_prompt`

Negative prompt used to restrict unwanted styles or artifacts.

Validation in this repository:

- Non-empty string

## `logging`

This section is optional.

### `level`

Logging level for the CLI.

Typical values:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`

### `file`

Optional path to a log file.

Example:

```yaml
logging:
  level: "INFO"
  file: "logs/run.log"
```

## Minimal Example

```yaml
cosmos:
  repo_root: "/path/to/cosmos-transfer2.5"
  model_variant: "edge"
  model_distilled: false
  disable_guardrails: true
  resolution: "720"
  guidance: 2.5
  num_steps: 24
  max_frames: 1
  num_video_frames_per_chunk: 1
  controls:
    seg:
      mode: external
      weight: 0.9
      subdir: "labels"
      encoding: "rgb"
    depth:
      mode: on_the_fly
      weight: 0.9
      subdir: "depth"
    edge:
      mode: disabled
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
  - name: "snow"
    output_dir: "dataset_snow"
    fraction: 1.0
    seed_base: 400
    prompt: >
      Photorealistic top-down view of the same scene during snowfall.
    negative_prompt: >
      cartoon, CGI, game graphics, oversmoothed, unrealistic colors
```
