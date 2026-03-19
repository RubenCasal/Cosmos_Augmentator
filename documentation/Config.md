# Configuration Reference

This repository is configured through a YAML file, typically `config/augmentations.yaml`. The reference below is intentionally compact and focuses on the parameters used by this project.

## Cosmos

- `repo_root`: Absolute path to the local `cosmos-transfer2.5` repository.
- `model`: Optional model name passed to Cosmos. In many setups this can be omitted.
- `model_variant`: Cosmos model variant, commonly `"edge"`.
- `model_distilled`: Enables distilled mode. Allowed values: `true`, `false`.
- `disable_guardrails`: Disables Cosmos guardrails. Allowed values: `true`, `false`.
- `resolution`: Target Cosmos resolution, typically a string such as `"720"`.
- `guidance`: Guidance scale. This repository validates it as numeric.
- `num_steps`: Number of inference steps. This repository validates it as an integer.
- `max_frames`: Maximum number of frames. For this tool it should normally be `1`.
- `num_video_frames_per_chunk`: Cosmos chunk size. For this tool it should normally be `1`.

## Control Parameters

These parameters apply to each control block under `cosmos.controls.seg`, `cosmos.controls.depth`, and `cosmos.controls.edge`.

- `mode`: How the control is used. Allowed values: `disabled`, `external`, `on_the_fly`.
- `weight`: Control strength. Validated range: `0.0` to `1.0`.
- `subdir`: Dataset subdirectory used when the control mode is `external`.
- `encoding`: Only relevant for `seg`. Allowed values: `rgb`, `id`.

Control defaults are usually:

- `seg.subdir`: `labels`
- `depth.subdir`: `depth`
- `edge.subdir`: `edges`

Behavior summary:

- `disabled`: the control is not sent to Cosmos
- `external`: the control is loaded from the dataset folder
- `on_the_fly`: Cosmos generates the control internally

The repository automatically adapts depth and edge inputs to a Cosmos-compatible RGB format when required.

## Dataset

- `input_root`: Absolute path to the source dataset.
- `output_root`: Absolute path where augmentation outputs and the merged dataset are written.
- `original_dir`: Relative path inside `input_root` containing the original dataset. Common value: `"."`.
- `image_subdir`: Folder containing RGB images. Common value: `"images"`.
- `label_subdir`: Folder containing semantic labels. Common value: `"labels"`.
- `image_ext`: File extension used to match dataset files. Common value: `".png"`.
- `cache_dir`: Internal cache folder for adapted controls. Common value: `".cosmos_control_cache"`.
- `augmentations[].name`: Name of the augmentation profile.
- `augmentations[].output_dir`: Output directory created for that augmentation.
- `augmentations[].fraction`: Fraction of the dataset to sample. Validated range: `0.0` to `1.0`.
- `augmentations[].seed_base`: Base integer seed used to generate deterministic per-image seeds.
- `augmentations[].prompt`: Positive prompt. Must be a non-empty string.
- `augmentations[].negative_prompt`: Negative prompt. Must be a non-empty string.

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
