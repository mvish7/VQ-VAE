# Trajectory VQ-VAE

Contains the PyTorch implementation and artifacts for a Vector Quantized Variational Autoencoder (VQ-VAE) designed to model autonomous vehicle trajectories. It is built upon [NVIDIA's PhysicalAI AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) dataset.

This Trajectory VQ-VAE module acts as a core component for [F2Q-VLA](https://github.com/mvish7/F2Q-VLA) training, enabling implicit reasoning and dynamics modeling for autonomous agents.

## 🌟 Key Features
* **1D ResNet Architecture:** Tailored encoder and decoder blocks specifically designed for sequential trajectory data.
* **Vector Quantization:** Learns a discrete codebook of trajectory primitives utilizing Exponential Moving Average (EMA) updates.
* **Comprehensive Loss Function:** Incorporates Reconstruction, Dynamics, Commitment, Entropy, and Unit Circle losses to ensure stable mappings, accurate forecasting, diverse codebook usage, and valid 2D yaw representations.
* **Dead Code Restart:** Automatically reinitializes unused codebook vectors during training to maximize codebook utilization.
* **On-the-fly Data Augmentation:** Supports random rotation, lateral mirroring, and Gaussian noise injections directly within the DataLoader.

## 📂 Project Structure
```text
VQ-VAE/
├── dataset/
│   ├── dataloader.py      # TrajDataset definition and batch generation
│   ├── augmentation.py    # On-the-fly trajectory augmentations
├── model/
│   ├── blocks.py          # Core 1D ResNet and processing blocks
│   ├── encoder.py         # Trajectory Encoder
│   ├── quantizer.py       # VQ EMA Vector Quantizer with dead code restart
│   ├── decoder.py         # Trajectory Decoder
├── trainer/
│   ├── trainer.py         # Training loop, validation, metric tracking, and checkpointing
├── tests/                 # Unit tests (e.g., augmentations)
├── train.py               # Main entry point for training
├── evaluate.py            # Model evaluation and codebook analysis
├── demo.py                # Inference/Demo script
└── README.md
```

## 🚀 Installation & Requirements
This project uses `uv` for dependency management.

1. Clone the repository:
   ```bash
   git clone https://github.com/mvish7/VQ-VAE.git
   cd VQ-VAE
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```

## 🧠 Model Architecture & Hyperparameters
The default model configuration defined in `train.py`:
* **Input Channels:** 5 (Trajectory features: xyz + 2D yaw)
* **Hidden Dim:** 256
* **Number of Embeddings:** 768 (Codebook size)
* **Embedding Dimension:** 256
* **Commitment Cost:** 0.10
* **Dynamics Weight:** 1.0
* **Unit Circle Weight:** 0.01
* **Entropy Weight:** 0.5

## 🚦 Training
To start training from scratch, run the main training script. Ensure the `dataset_path` in `train.py` points to your downloaded NVIDIA AV dataset.
```bash
python train.py
```

### Resuming Training
To resume training from a specific checkpoint, simply update the `resume_from` key in the `train.py` config block:
```python
    "resume_from": "checkpoints/last.pt",  # Path to your checkpoint
```
The trainer will automatically restore the model weights, optimizer states, learning rate schedules, and the current epoch step.

### Logging
Training logs and scalars (loss, reconstruction, dynamics, commitment, entropy, unit circle, and perplexity) are automatically recorded via Tensorboard. To view the metrics:
```bash
uv run tensorboard --logdir=runs/
```

## 📊 Checkpoints & Validation
During training, checkpoints are saved automatically in the `checkpoints/` directory:
* `last.pt` - The most recent epoch's state.
* `best.pt` - The model state with the lowest validation reconstruction loss.

## 📏 Evaluation
Run the evaluation script against a trained checkpoint to get a full report on reconstruction quality, dynamics preservation, and codebook health:
```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

Optional arguments:
| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to a `.pt` checkpoint |
| `--dataset_path` | `<configured path>` | Path to the test dataset |
| `--batch_size` | `512` | Evaluation batch size |
| `--output_dir` | Checkpoint directory | Where to save `eval_results.json` |

### Reported Metrics
* **Reconstruction MSE** — Overall trajectory reconstruction accuracy.
* **Velocity MSE** — First-order dynamics preservation.
* **Acceleration MSE** — Second-order dynamics preservation.
* **Batch / Global Perplexity** — Codebook usage diversity.
* **Active Codes & Utilization %** — How many codebook entries are actually used.
* **Top / Bottom 10 Code Frequencies** — Distribution of codebook usage.

Results are printed to the console and saved as `eval_results.json` in the output directory.
