# 🚗 3DGazeNet: Driver Monitoring System (DMS) using 3D Gaze Estimation

## 🧠 An AI-Powered Computer Vision System for Autonomous Driving Safety

3DGazeNet is a general gaze estimation model that can be directly employed in novel environments without fine-tuning or adaptation. This approach leverages 3D eye mesh regression combined with multi-view consistency constraints to achieve robust gaze estimation across diverse scenarios.

---

## 📖 Executive Summary

**The Problem:** According to the **World Health Organization (WHO)**, approximately **1.35 million people** die annually from road traffic crashes, with human error (distraction and fatigue) being a leading cause.

**The Solution:** This project is a **non-intrusive Driver Monitoring System (DMS)** developed to assess driver alertness in real-time using a standard monocular camera. Unlike older systems that require expensive hardware or wearables, this system uses **Deep Learning (ResNet-18)** and **Geometric Computer Vision** to detect:
1. **Gaze Direction:** Determining exactly where the driver is looking (e.g., *Side Mirror, Infotainment, Road*).
2. **Drowsiness:** Detecting micro-sleeps and yawning patterns based on **EuroNCAP standards**.
3. **Distraction:** Identifying when the driver's head is averted for unsafe durations.

**Key Result:** The system achieved **90.5% accuracy** in Gaze Classification across 9 distinct Areas of Interest (AOIs) in a realistic vehicle setup.

---

## Overview

The core innovation of 3DGazeNet is expressing gaze estimation as dense 3D eye mesh prediction rather than direct angle regression. This approach is inspired by successful methods in pose estimation (head, body, hand) that reformulate the problem as dense 3D coordinate prediction. The method is trained with weak supervision from synthetic views and uses unlabeled in-the-wild face images to enhance generalization in real-world conditions.

### Key Features

- Dense 3D eye mesh prediction for robust gaze estimation
- Multi-view consistency constraints for improved generalization
- Training on diverse public gaze datasets
- State-of-the-art zero-shot generalization
- Real-time inference on videos and images
- Extended features: drowsiness detection, blink detection, yawn detection, and depth estimation
- **Non-intrusive driver monitoring** for autonomous vehicle safety
- **Robust performance** across diverse driver demographics and vehicle setups

## 🛠️ Tech Stack & Architecture

| Component | Technology / Library | Role in Pipeline |
| :--- | :--- | :--- |
| **Gaze Estimation** | **3D GazeNet** (PyTorch) | A Deep Learning model (ResNet-18) that predicts a 3D gaze vector from eye images. |
| **Face Tracking** | **MediaPipe Face Mesh** | Lightweight CPU-based tracking to find facial landmarks (eyes, lips) and Head Pose. |
| **Logic/Intersection** | **Möller-Trumbore Algorithm** | Calculates where the 3D gaze vector "hits" the 3D model of the car dashboard. |
| **Smoothing** | **Kalman Filter** | Reduces the "jitter" or shaking of the gaze vector to provide a stable reading. |
| **Hardware** | **Logitech C920** + **Laptop GPU** | Camera input and inference engine. (Tested on Jetson Nano, deployed on Laptop). |

---

## Supported Datasets

This project supports training and inference on multiple public gaze estimation datasets:

- **ETH-XGaze** (80 identities, ~750K images)
- **GazeCapture** (1450 identities, ~2M images)
- **Gaze360** (238 identities, ~150K images)
- **MPIIFaceGaze** (15 identities, ~45K images)
- **VFHQ** (in-the-wild face dataset for unsupervised training)

---

## ⚙️ How It Works: The "Under the Hood" Logic

### 1. 3D Gaze Estimation (The Deep Learning Core)
We utilized an appearance-based method adapted from **Park et al. (3D GazeNet)**.
* **Why ResNet-18?** We chose a Residual Network backbone to solve the **Vanishing Gradient Problem** using "skip connections." This allows the model to learn deep features from the eye images without losing signal during backpropagation.
* **Inputs:** Left Eye Image ($224 \times 224$) + Right Eye Image ($224 \times 224$) + Head Pose Angles (Yaw, Pitch).
* **Outputs:** A 3D Unit Vector ($g_x, g_y, g_z$).

**Data Normalization:**
To make the model robust against different driver positions, we "warped" the eye images to a standardized "Virtual Camera" view using this transformation matrix:
$$W = S \cdot R \cdot K^{-1}$$
*(Where $S$ is scaling, $R$ is rotation from head pose, and $K$ is the camera intrinsic matrix.)*

### 2. Gaze Mapping (The "Laser Beam" Logic)
Once the model outputs a vector (think of it as a laser beam shooting from the eye), we need to know **what** it hit.
* We created a 3D virtual map of the car interior with **9 Areas of Interest (AOIs)** (e.g., Left Mirror, Radio, Speedometer).
* We used the **Möller-Trumbore Intersection Algorithm** to mathematically calculate if the gaze vector intersects with any of these AOI planes.

### 3. Drowsiness & Fatigue Detection (The Mathematical Standards)
We adhered to **EuroNCAP** safety protocols to classify driver state. This is calculated using facial landmarks extracted via MediaPipe.

#### **A. Eye Aspect Ratio (EAR)** - *Detecting Sleep*
This formula measures how "open" the eye is.
$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$
* **Microsleep:** If EAR < Threshold for **> 3 seconds**.
* **Unresponsive:** If EAR < Threshold for **> 6 seconds** (Critical Warning).

#### **B. Mouth Aspect Ratio (MAR)** - *Detecting Yawning*
$$MAR = \frac{||p_{51} - p_{59}|| + ||p_{52} - p_{58}|| + ||p_{53} - p_{57}||}{3 ||p_{49} - p_{55}||}$$
* Triggers a warning if the mouth remains wide open (yawn) beyond a specific time threshold.

#### **C. Monocular Depth Estimation**
We estimated the distance ($D$) of the driver from the camera to ensure they are in a safe position:
$$D = \frac{F \times W}{P}$$
* $F$: Focal Length (from camera calibration).
* $W$: Average Inter-pupillary distance (~63mm).
* $P$: Pixel width between eyes on the sensor.

---

## Demo

<p align="center">
  <img src="assets/teaser_1_2.png" height="200" title="teaser1">
  <img src="assets/teaser_2_2.png" height="200" title="teaser2">
</p>

https://github.com/Vagver/dense3Deyes/assets/25174551/4de4fb76-9577-4209-ba07-779356230131

For a demo of 3DGazeNet on videos and single images visit the [demo folder](demo).

---


## 🧪 Experimental Setup (The "BMW Lab Car")

To validate the system without the legal risks of open-road testing, we utilized the **AI Motion Lab BMW Vehicle**.

* **Scenario:** The car was stationary, but participants simulated driving tasks (checking mirrors, turning the steering wheel).
* **Occlusion Testing:** A key part of the experiment was ensuring the camera could still track the face even when the driver's hands crossed the steering wheel (blocking part of the view).
* **Dataset:**
    * **27 Videos** collected (30s each).
    * **5 Diverse Participants** (varying heights, glasses vs. no glasses).
    * **9 Target Zones** (Mirrors, Windshield, Center Stack, Road, Infotainment, etc.).

---

## 📊 Results & Performance Analysis

### Accuracy by Region

| Area of Interest (AOI) | Accuracy | Notes |
| :--- | :--- | :--- |
| **Road / Center** | **High** | Best performance; face is frontal. |
| **Infotainment** | **High** | Strong detection. |
| **Side Mirrors** | **Lower** | Worst performance due to extreme head rotation angles. |
| **Overall** | **90.5%** | Robust for general safety monitoring. |

### The "Real-Time" Engineering Challenge
We initially attempted to run the full pipeline on an **NVIDIA Jetson Nano** (embedded computer).
* **Problem:** The 3D GazeNet model was too computationally heavy for the Nano, resulting in **5-8 FPS** (too slow for safety).
* **Solution:** We offloaded the inference to a **Laptop with a dedicated GPU**, achieving **~30 FPS** (Real-time).
* **Verification:** The code was later verified by supervisors on the car's internal Linux system, proving compatibility despite the hardware bottleneck.

---

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

---

## 🔮 Future Work
1. **Model Quantization:** Pruning the ResNet-18 model to allow it to run on the Jetson Nano at 30 FPS.
2. **Night Mode:** Integrating an **IR (Infrared) Camera** to allow monitoring in total darkness (RGB cameras fail at night).
3. **Personalization:** Adding a 5-second "calibration phase" for new drivers to learn their specific eye shape, improving accuracy.

---

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
