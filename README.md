# 3DGazeNet: Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views

3DGazeNet is a general gaze estimation model that can be directly employed in novel environments without fine-tuning or adaptation. This approach leverages 3D eye mesh regression combined with multi-view consistency constraints to achieve robust gaze estimation across diverse scenarios.

## Overview

The core innovation of 3DGazeNet is expressing gaze estimation as dense 3D eye mesh prediction rather than direct angle regression. This approach is inspired by successful methods in pose estimation (head, body, hand) that reformulate the problem as dense 3D coordinate prediction. The method is trained with weak supervision from synthetic views and uses unlabeled in-the-wild face images to enhance generalization in real-world conditions.


### Key Features

- Dense 3D eye mesh prediction for robust gaze estimation
- Multi-view consistency constraints for improved generalization
- Training on diverse public gaze datasets
- State-of-the-art zero-shot generalization
- Real-time inference on videos and images
- Extended features: drowsiness detection, blink detection, yawn detection, and depth estimation

## Supported Datasets

This project supports training and inference on multiple public gaze estimation datasets:

- **ETH-XGaze** (80 identities, ~750K images)
- **GazeCapture** (1450 identities, ~2M images)
- **Gaze360** (238 identities, ~150K images)
- **MPIIFaceGaze** (15 identities, ~45K images)
- **VFHQ** (in-the-wild face dataset for unsupervised training)

## Demo

<p align="center">
  <img src="assets/teaser_1_2.png" height="200" title="teaser1">
  <img src="assets/teaser_2_2.png" height="200" title="teaser2">
</p>

https://github.com/Vagver/dense3Deyes/assets/25174551/4de4fb76-9577-4209-ba07-779356230131

For a demo of 3DGazeNet on videos and single images visit the [demo folder](demo).


## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 12.1 compatible GPU (recommended for training)
- Conda package manager

### Setup

Create and activate a conda environment with dependencies:

```bash
conda env create --file env_requirements.yaml
conda activate 3DGazeNet
```

Alternatively, for a minimal setup:

```bash
conda create -n 3DGazeNet python=3.9
conda activate 3DGazeNet
pip install -r demo/requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Download Pre-trained Models

Download the pre-trained model weights from [here](https://drive.google.com/file/d/1mYvKRJGS8LY5IU3I8Qfvm-xINQyby1z5/view?usp=sharing). Extract and place the folder in the root directory of this repository.

## Quick Start

### Running Inference

#### On Videos

```bash
cd demo
python inference_video.py --cfg configs/infer_res18_x128_all_vert.yaml \
                          --video_path <path_to_video> \
                          --smooth_predictions
```

#### On Images

```bash
cd demo
python inference.py --image_path <path_to_image>
```

#### Integrated Analysis (Group Project Extension)

For the enhanced version with drowsiness, blink, yawn, and depth detection:

```bash
cd demo
python inference_video_integrated.py
```

Note: Update the video path in the script before running.

### Training

To train the model on a specific dataset:

```bash
python train.py --cfg configs/singleview/<dataset>/<dataset>_train.yaml
```

Example configurations:
- Gaze360: `configs/singleview/gaze360/gaze360_train.yaml`
- XGaze: `configs/singleview/xgaze/xgaze_train.yaml`
- MPIIFaceGaze: `configs/singleview/mpiiface/mpiiface_train.yaml`
- GazeCapture: `configs/singleview/gazecapture/gazecapture_train.yaml`

### Testing

To evaluate on test sets:

```bash
python test.py --cfg configs/singleview/<dataset>/<dataset>_train.yaml \
               --checkpoint <path_to_checkpoint>
```

## Dataset Preparation

Before training, datasets must be preprocessed to fit 3D eyes on face images.

### XGaze

1. Download the 448x448 pixel version from https://ait.ethz.ch/xgaze
2. Place in `datasets/xgaze/` folder
3. Run preprocessing:
   ```bash
   cd tools
   python xgaze_preprocess.py
   ```

### Gaze360

1. Download from http://gaze360.csail.mit.edu/
2. Place in `datasets/gaze360/` folder
3. Run preprocessing:
   ```bash
   cd tools
   python gaze360_preprocess.py
   ```

### MPIIFaceGaze

1. Download from https://www.mpi-inf.mpg.de/
2. Place in `datasets/mpiiface/` folder
3. Run preprocessing:
   ```bash
   cd tools
   python mpiiface_preprocess.py
   ```

### Visualization

After preprocessing, visualize 3D eye fittings:

- `notebooks/xgaze_view_dataset.ipynb` - For XGaze dataset
- `notebooks/gaze360_view_dataset.ipynb` - For Gaze360 dataset
- `notebooks/mpiiface_view_dataset.ipynb` - For MPIIFaceGaze dataset

## Project Structure

- `lib/` - Core library modules
  - `core/` - Training, testing, and inference loops
  - `models/` - Model builders and components
  - `dataset/` - Dataset implementations and loaders
  - `utils/` - Utilities for metrics, configuration, logging
- `configs/` - Configuration files for different datasets
- `demo/` - Demo scripts for inference on videos and images
- `tools/` - Dataset preprocessing tools
- `notebooks/` - Jupyter notebooks for visualization and analysis
- `scripts/` - Shell scripts for batch processing

## Inference Options

### Video Inference

Optional arguments:
- `--no_draw` - Skip frame drawing and video export (faster processing)
- `--smooth_predictions` - Enable prediction smoothing across consecutive frames

### Image Inference

Optional arguments:
- `--no_draw` - Skip drawing results on image
- `--draw_detection` - Display face detection bounding boxes

## Group Project Extensions

This implementation extends the base 3DGazeNet with additional features:

- **Drowsiness Detection** - Monitors eye closure ratios to detect fatigue
- **Blink Detection** - Identifies and counts blinks
- **Yawn Detection** - Detects yawning events
- **Depth Estimation** - Estimates face depth in 3D space

These features are integrated into `demo/inference_video_integrated.py` for comprehensive eye-based analysis.


## Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{ververas20253dgazenet,
  title={3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views},
  author={Ververas, Evangelos and Gkagkos, Polydefkis and Deng, Jiankang and Doukas, Michail Christos and Guo, Jia and Zafeiriou, Stefanos},
  booktitle={European Conference on Computer Vision},
  pages={387--404},
  year={2025},
  organization={Springer}
}
```

## License

This project is provided for research and educational purposes.

## Acknowledgments

This work builds upon the original 3DGazeNet framework and extends it with additional functionality for comprehensive eye-based behavior analysis in a group project context.
