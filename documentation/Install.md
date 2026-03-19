# Installation

This guide explains how to install NVIDIA Cosmos-Transfer2.5 together with `cosmos_augmentor` on a Linux machine. The steps below are based on the official Cosmos setup flow and adapted to the way this repository is used in practice.

## 1. Install System Dependencies

Install the required system packages:

```bash
sudo apt update
sudo apt install -y curl ffmpeg git git-lfs libx11-dev tree wget
git lfs install
```

## 2. Install `uv`

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Verify the installation:

```bash
uv --version
```

## 3. Clone Cosmos-Transfer2.5

Clone the official Cosmos repository and download LFS assets:

```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5
git lfs pull
```

## 4. Create the Cosmos Environment

Install Python and create the virtual environment with the CUDA 12.8 dependency set:

```bash
uv python install
uv sync --extra=cu128
source .venv/bin/activate
```

If your machine targets a different CUDA setup, check the official Cosmos setup guide before changing the extra.

## 5. Configure Hugging Face Access

Cosmos checkpoints are downloaded automatically during inference, so you need a valid Hugging Face token with read access.

Install the Hugging Face CLI:

```bash
uv tool install -U "huggingface_hub[cli]"
```

Login:

```bash
hf auth login
```

You must also accept the NVIDIA model license for the Cosmos checkpoints you plan to use.

## 6. Clone This Repository

Move to your workspace and clone `cosmos_augmentor`:

```bash
cd ..
git clone <YOUR_COSMOS_AUGMENTOR_REPOSITORY_URL>
cd cosmos_augmentor
```

## 7. Install `cosmos_augmentor`

Install this repository into the active Cosmos environment:

```bash
uv pip install -e .
```

## 8. Run the Pipeline

Launch the full pipeline:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml run-all
```

Run only augmentation:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml run-augmentations
```

Run only merge:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml merge
```

## 9. Optional Logging Overrides

You can override logging from the CLI:

```bash
python -m cosmos_augmentor.cli --config config/augmentations.yaml --log-level INFO --log-file logs/run.log run-all
```

## Official Cosmos References

- Cosmos repository: `https://github.com/nvidia-cosmos/cosmos-transfer2.5`
- Cosmos setup guide: `https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md`
