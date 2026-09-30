# Dual-Teacher Knowledge Distillation for Deepfake Detection

This repository contains the implementation of a Dual-Teacher Knowledge Distillation framework designed for robust and cross-domain deepfake detection. The project leverages domain-specific teachers to train a highly capable student model on multiple datasets simultaneously.

## Main Script
The core logic for data loading, model architecture, loss computation, and the training/evaluation loop is contained within a single comprehensive script: **`Fusing teacher.py`**. 

## Overview (V15 Architecture)

The system tackles the challenge of deepfake detection generalization by distilling knowledge from two domain-specific teachers into a single student network. 

### Key Features
*   **Dual-Teacher Setup:** 
    *   *Teacher A* is specialized in FaceForensics++ (FF++).
    *   *Teacher B* is specialized in Celeb-DF (CDF).
*   **6-Channel Input Integration:** The student backbone is modified to accept a 6-channel input consisting of standard RGB combined with a High-Frequency (HF) Laplacian filter channel to better capture blending artifacts.
*   **Student Backbone:** `EfficientNet-B2` (via `timm`), featuring expert heads, a soft domain gate, and dual domain-specific classifiers.
*   **Advanced Loss Function (`DualKDLoss`):**
    *   Soft Knowledge Distillation (Soft KD) with Conflict Weighting.
    *   Hard Focal Loss to handle class imbalances.
    *   Feature KD utilizing Cosine Similarity mapping student projections to teacher features.
    *   Bimodal gate sparsity loss to optimize domain routing.
*   **Domain-Aware MixUp:** A critical fix in V15 that applies MixUp exclusively to FF++ samples while protecting Celeb-DF (which suffers from severe performance drops when mixed).
*   **Robust Evaluation:** Built-in Stochastic Weight Averaging (SWA), Test-Time Augmentation (TTA), Temperature Scaling for calibration, and bootstrapping for confidence intervals.

## Datasets

The script is configured to train and evaluate on two standard deepfake datasets:
1.  **FaceForensics++ (FF++)**: Includes Real, Face2Face, Deepfakes, NeuralTextures, FaceShifter, and FaceSwap data.
2.  **Celeb-DF (CDF)**: A challenging cross-domain dataset.

*Note: The script utilizes a multithreaded fast scanner (`parallel_fast_scan`) to bypass I/O bottlenecks when loading frames.*

## Requirements

To run `Fusing teacher.py`, you will need the following primary libraries:
*   Python 3.8+
*   PyTorch & torchvision
*   `timm` (PyTorch Image Models)
*   `opencv-python` (cv2)
*   `scikit-learn`
*   `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`, `Pillow`

## Usage

1. **Configure Paths:** 
   Open `Fusing teacher.py` and update the paths in **Section 1 — Configuration** to point to your local or cloud dataset directories and pre-trained teacher model checkpoints:
   ```python
   FF_FRAME_ROOT = "/path/to/FaceForensics_Frames"
   CDF_FRAME_ROOT = "/path/to/Celeb-DF_Frames"
   TEACHER_A_PATH = "/path/to/teacher_A.pth"
   TEACHER_B_PATH = "/path/to/teacher_B.pth"
   OUT_DIR = Path("/path/to/output_dir")
   ```
2. **Run the Script:**
   Execute the script to start the training and evaluation pipeline:
   ```bash
   python "Fusing teacher.py"
   ```

## Outputs

Upon execution, the script automatically manages checkpoints and evaluation files in the specified `OUT_DIR`:
*   **Checkpoints:** Best models, crash backups, and SWA weights (`.pth`).
*   **Logs:** Training history CSVs.
*   **Metrics:** Detailed predictions, per-manipulation accuracy, ROC-AUC, ECE (Expected Calibration Error), and bootstrap CI files.
*   **Figures:** Generates performance plots (`fig01` to `fig05`) visualizing loss convergence, ROC curves, and score distributions directly in the `/figures/` subdirectory.