# 🚗 3DGazeNet: Driver Monitoring System (DMS) using 3D Gaze Estimation

## An AI based Computer Vision System for Autonomous Driving Safety

3DGazeNet is a general gaze estimation model that can be directly employed in novel environments without fine-tuning or adaptation. This approach leverages 3D eye mesh regression combined with multi-view consistency constraints to achieve robust gaze estimation across diverse scenarios.

---

## Motivation

According to the **World Health Organization (WHO)**, approximately **1.35 million people** die annually from road traffic crashes, with human error (distraction and fatigue) being a leading cause.

## Solution

This project is a **non-intrusive Driver Monitoring System (DMS)** developed to assess driver alertness in real-time using a standard monocular camera. Unlike older systems that require expensive hardware or wearables, this system uses **Deep Learning (ResNet-18)** and **Geometric Computer Vision** to detect:
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

## ⚙️ How It Works

### 1. 3D Gaze Estimation
This implementation employs an appearance-based method adapted from **Park et al. (3D GazeNet)**.
* **Architecture:** ResNet-18 is used as the backbone to address the vanishing gradient problem through skip connections, enabling effective learning of deep features from eye images.
* **Inputs:** Left Eye Image ($224 \times 224$) + Right Eye Image ($224 \times 224$) + Head Pose Angles (Yaw, Pitch).
* **Outputs:** 3D Unit Vector ($g_x, g_y, g_z$).

**Data Normalization:**
Eye images are warped to a standardized virtual camera view using the following transformation matrix to ensure robustness across different driver positions:
$$W = S \cdot R \cdot K^{-1}$$
*(Where $S$ is scaling, $R$ is rotation from head pose, and $K$ is the camera intrinsic matrix.)*

### 2. Gaze Mapping
The gaze vector is mapped to specific regions of interest within the vehicle interior using geometric algorithms.
* A 3D virtual model of the car interior defines **9 Areas of Interest (AOIs)** (e.g., Left Mirror, Radio, Speedometer).
* The **Möller-Trumbore Intersection Algorithm** calculates whether the gaze vector intersects with any of these predefined AOI planes.

### 3. Drowsiness & Fatigue Detection
Driver state classification follows **EuroNCAP** safety standards using facial landmarks extracted via MediaPipe.

#### **A. Eye Aspect Ratio (EAR)**
Eye Aspect Ratio quantifies the degree of eye opening:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

* **Microsleep Detection:** EAR below threshold for > 3 seconds indicates fatigue.
* **Unresponsive State:** EAR below threshold for > 6 seconds triggers critical alert.

#### **B. Mouth Aspect Ratio (MAR)**

$$MAR = \frac{||p_{51} - p_{59}|| + ||p_{52} - p_{58}|| + ||p_{53} - p_{57}||}{3 ||p_{49} - p_{55}||}$$

* Yawn detection is triggered when MAR exceeds the defined threshold for a sustained duration.

#### **C. Monocular Depth Estimation**
Driver distance from the camera is estimated using monocular depth calculation:

$$D = \frac{F \times W}{P}$$

* $F$: Focal Length (from camera calibration).
* $W$: Average inter-pupillary distance (~63mm).
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


## Experimental Setup

Validation was conducted using a stationary **AI Motion Lab vehicle** to enable controlled testing without safety risks associated with open-road evaluation.

* **Testing Scenario:** Participants performed simulated driving tasks including mirror checking and steering wheel manipulation.
* **Occlusion Robustness:** The system was tested for camera robustness to facial occlusion caused by hands during steering wheel operations.
* **Dataset Characteristics:**
    * **27 Video sequences** collected at 30 seconds duration each.
    * **5 Participants** with diverse characteristics (height, presence/absence of corrective eyewear).
    * **9 Areas of Interest** mapped within the vehicle (mirrors, windshield, center stack, road, infotainment, etc.).

---

## 📊 Results & Performance Analysis

### Accuracy by Region

| Area of Interest (AOI) | Accuracy | Notes |
| :--- | :--- | :--- |
| **Road / Center** | **High** | Best performance; face is frontal. |
| **Infotainment** | **High** | Strong detection. |
| **Side Mirrors** | **Lower** | Worst performance due to extreme head rotation angles. |
| **Overall** | **90.5%** | Robust for general safety monitoring. |

### Real-Time Performance Considerations
Initial implementation on **NVIDIA Jetson Nano** (embedded hardware) demonstrated computational constraints:
* **Performance Limitation:** The 3D GazeNet model achieved only **5-8 FPS** on Jetson Nano, insufficient for real-time safety applications.
* **Implementation Solution:** Inference was relocated to **laptop GPU**, achieving **~30 FPS** for real-time operation.
* **System Compatibility:** Code was verified on the vehicle's internal Linux system, confirming cross-platform compatibility despite hardware optimization requirements.

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

## Future Work
1. **Model Optimization:** Model pruning and quantization to enable efficient execution on embedded devices (Jetson Nano) at 30 FPS.
2. **Infrared Integration:** Incorporation of **IR (Infrared) camera** technology to extend monitoring capabilities to low-light and nighttime conditions.
3. **Personalization Framework:** Implementation of user calibration protocols to adapt gaze estimation to individual driver characteristics.

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

This implementation extends the original 3DGazeNet framework with integrated functionality for comprehensive eye-based behavior analysis and driver monitoring system applications.
