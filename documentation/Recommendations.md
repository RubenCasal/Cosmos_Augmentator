# Practical Recommendations

This section collects practical lessons learned while using the tool on synthetic semantic segmentation datasets. These are not hard rules, but they can save a lot of trial and error when starting out.

## Control Weights

If the configured control weights sum to more than `1.0`, Cosmos may normalize them internally. Because of that, it is important to think about the relative balance between controls, not only their absolute values.

Cosmos documentation often recommends using `0.9` for segmentation and depth while ignoring the other controls. In practice, one configuration that produced strong results in this project was:

- `seg = 0.5`
- `depth = 0.5`
- `edge = 0.5`

In particular, increasing the contribution of `edge` helped reduce hallucinations and improved the preservation of semantic boundaries.

## Start Small

When designing prompts and tuning control weights, it is strongly recommended to start with a small subset of the dataset, for example `10` images. This makes it much faster to see which configuration works well for a specific environment before launching a large run.

It is also better not to mix very different environments during prompt design. In practice, prompt tuning is usually more reliable when each environment has its own dedicated prompt and configuration.

## Number of Steps

`num_steps` usually requires fine tuning for the specific dataset.

- Too few steps often produce blurry outputs with weak detail
- Too many steps can increase hallucinations and unwanted scene changes

In practice, a good working range was often around `24` to `26` steps.

## Guidance

Although Cosmos often recommends guidance values around `4` to `6`, lower values worked better in this workflow.

- Around `1` often produced weak generations
- Higher values often introduced too many new elements
- A practical range around `2` to `3` usually gave better structure preservation

Even when the visual result looked attractive at high guidance values, it often drifted too far from the original labels for semantic segmentation use cases.

## Negative Prompt Design

Do not use the negative prompt as the literal opposite of what you want to generate.

Example:

- If you want a sunny day, avoid writing a negative prompt focused on cloudy weather

Instead, use the negative prompt mainly to restrict rendering style and unwanted visual domains.

Good examples:

- `CGI`
- `simulation`
- `game graphics`
- `cartoon`

Adding too many semantic restrictions in the negative prompt can confuse the model and reduce consistency.

## Prompt Writing Style

The most effective prompts are usually simple, direct, and descriptive.

It works better to describe the final image as if it already existed than to write instructions as if you were chatting with an assistant.

Better style:

> Photorealistic top-down view of an outdoor robotic scene during snowfall, with soft winter light and preserved spatial layout.

Less effective style:

> Create a snowy image and do not add new structures.

## Keep the Prompt Focused

Do not try to force too many changes into a single prompt. This model was trained with a large amount of driving-related imagery, so if the prompt is too open-ended it may introduce unwanted sky or visual elements that do not belong in a ground-facing camera setup.

For top-down or downward-facing Isaac Sim views, it helps to explicitly mention the camera viewpoint and scene type.

Useful wording:

- `top-down view`
- `downward-facing camera`
- `aerial view of the same scene`

## Preserve Scene Interpretability

The original Isaac Sim images should be as readable as possible before augmentation.

Potential issues:

- Excessive shadows can introduce ambiguity
- Unclear object boundaries can confuse the model
- Very flat materials or unrealistic simulator colors can reduce realism and hurt generation quality

In practice, the better the original scene is visually interpretable, the more consistent Cosmos tends to be when generating realistic variations while keeping the semantic structure intact.

## Curate the Dataset Manually

A manual cleanup pass is highly recommended before using the dataset at scale. Even with good controls and prompts, Cosmos can still produce hallucinations or inconsistent outputs on difficult samples.

Removing problematic source images and reviewing generated results manually usually improves the final dataset quality more than trying to solve every issue through prompt engineering alone.
