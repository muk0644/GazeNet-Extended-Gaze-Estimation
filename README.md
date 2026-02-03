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

3DGazeNet predicts where someone is looking by analyzing their eye images in 3D space. Instead of just calculating simple angles, it creates a detailed 3D model of the eyes to better understand gaze direction. This makes it work reliably across different camera angles, lighting conditions, and environments.

The system is trained using multiple datasets and can work immediately in new settings without needing additional training or calibration.

### Key Features

- **3D Eye Modeling:** Creates detailed 3D models of eyes for accurate gaze prediction
- **Works in Multiple Environments:** Can be used in different settings without retraining
- **Trained on Large Datasets:** Uses thousands of images from public gaze datasets
- **Real-Time Processing:** Analyzes video frames instantly for immediate feedback
- **Safety Features:** Detects drowsiness, blinks, yawns, and driver distance from camera
- **Non-Intrusive:** Uses only a regular camera, no special glasses or sensors needed
- **Robust:** Works with different people regardless of height, glasses, or face shape

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
The system analyzes eye images to determine where a person is looking in 3D space.

* **Neural Network Architecture:** Uses ResNet-18, a proven deep learning model that excels at image analysis. It has "skip connections" which help it learn better by allowing information to flow more easily through the network.
* **What Goes In:** Two eye images (left and right, each 224×224 pixels) plus the angle of the head (how much it's tilted or turned).
* **What Comes Out:** A 3D direction vector showing where the eyes are pointing (coordinates: x, y, z).

**Image Preparation:**
Before analyzing, the system adjusts eye images to a standard viewpoint. This ensures consistent results regardless of where the driver is sitting or how the camera is positioned.

The transformation uses this formula:
$$W = S \cdot R \cdot K^{-1}$$

Where:
- $S$ = Scaling factor (adjusts image size)
- $R$ = Rotation (corrects for head pose)  
- $K$ = Camera properties (lens characteristics)

### 2. Gaze Mapping (Finding What the Driver is Looking At)
Once the system knows the gaze direction (like an invisible line from the eyes), it needs to figure out what object that line hits.

* **Virtual 3D Car Model:** A digital map of the car interior is created with **9 specific zones** (Side Mirrors, Radio, Speedometer, Road ahead, etc.).
* **Intersection Calculation:** The **Möller-Trumbore Algorithm** (a mathematical method) checks if the gaze line crosses any of these 9 zones.

**Simple Analogy:** Imagine shining a laser pointer from the driver's eyes - this algorithm tells us which car part the laser beam hits.

### 3. Drowsiness & Fatigue Detection
The system monitors the driver's alertness using facial features tracked by MediaPipe (a face tracking library). It follows **EuroNCAP** safety standards - the same standards used by automotive safety organizations in Europe.

#### **A. Eye Aspect Ratio (EAR) - Measuring How Open the Eyes Are**
This formula calculates how wide open the eyes are:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

The formula compares vertical eye opening distances to horizontal eye width. When the value drops below a threshold:
* **Microsleep (Light Drowsiness):** Eyes closed for more than 3 seconds → Driver is getting tired
* **Unresponsive (Critical):** Eyes closed for more than 6 seconds → Emergency alert triggered

#### **B. Mouth Aspect Ratio (MAR) - Detecting Yawning**

$$MAR = \frac{||p_{51} - p_{59}|| + ||p_{52} - p_{58}|| + ||p_{53} - p_{57}||}{3 ||p_{49} - p_{55}||}$$

This measures mouth opening. A sustained high value indicates yawning, which is a sign of fatigue.

#### **C. Distance Estimation - How Far is the Driver?**
The system estimates how far the driver is sitting from the camera using this calculation:

$$D = \frac{F \times W}{P}$$

Where:
- $F$ = Focal Length (a property of the camera from calibration)
- $W$ = Average distance between pupils (~63mm for most adults)
- $P$ = How many pixels wide the space between eyes appears on camera

**Why This Matters:** If the driver is too close or too far, the system accuracy changes. This measurement helps ensure proper positioning.

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

The system was tested in a **parked AI Motion Lab vehicle** to ensure safety during testing (no actual driving on roads).

**Test Details:**
* **What Participants Did:** Sat in driver's seat and performed realistic driving actions like checking side mirrors, adjusting radio, and turning steering wheel.
* **Challenge Tested:** Ensured the camera could still track the face even when hands temporarily block the view (like when turning the steering wheel).
* **Data Collected:**
    * **27 Video clips** (each 30 seconds long)
    * **5 Different People** tested (various heights, some with glasses, some without)
    * **9 Zones Tracked:** Side mirrors, windshield, center console, radio, speedometer, road view, etc.

---

## 📊 Results & Performance Analysis

### Accuracy by Region

| Area of Interest (AOI) | Accuracy | Notes |
| :--- | :--- | :--- |
| **Road / Center** | **High** | Best performance; face is frontal. |
| **Infotainment** | **High** | Strong detection. |
| **Side Mirrors** | **Lower** | Worst performance due to extreme head rotation angles. |
| **Overall** | **90.5%** | Robust for general safety monitoring. |

### Real-Time Performance (Hardware Challenges)
**Initial Challenge:**  
The system was first tested on an **NVIDIA Jetson Nano** (a small, embedded computer designed for AI projects).

* **Problem Found:** The Jetson Nano could only process **5-8 frames per second** (FPS), which is too slow for real-time driver safety monitoring (you need ~30 FPS for smooth, safe operation).
* **Solution:** Moved the processing to a **laptop with a dedicated GPU** (graphics card), which achieved **~30 FPS** - fast enough for real-time use.
* **Compatibility Verified:** The code was tested on the car's Linux computer system and works correctly, proving it can run on different platforms (the Jetson Nano just needs optimization to run faster).

**Takeaway:** The system works in real-time on standard laptop hardware. Future work will optimize it to run efficiently on smaller embedded devices.

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

**Planned Improvements:**

1. **Optimize for Embedded Devices:** 
   - Reduce the model size (called "pruning" and "quantization") so it can run at 30 FPS on small computers like the Jetson Nano
   - This would allow the system to be installed directly in vehicles without needing a laptop

2. **Night Vision Capability:** 
   - Add support for **Infrared (IR) cameras** that can see in complete darkness
   - Current system uses regular RGB cameras which don't work well at night

3. **Personal Calibration:** 
   - Add a quick setup step where the system learns each new driver's unique eye shape and position
   - This would improve accuracy for individual users (similar to Face ID setup on phones)

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
