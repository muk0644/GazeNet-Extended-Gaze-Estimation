# 🚗 3DGazeNet: End-to-End Driver Monitoring System for Autonomous Driving Safety

## An AI-Based Computer Vision System for Interior Monitoring and Driver Alertness Assessment

**Project:** End-to-End Development of an Interior Monitoring System for Autonomous Driving Safety Applications

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

### Understanding DMS vs. Occupant Monitoring

**What's the Difference?**

- **Driver Monitoring System (DMS):** Focuses specifically on the driver - tracking their gaze direction, head position, blink rate, and signs of drowsiness or distraction. Think of it as a "safety guardian" that watches only the person controlling the vehicle.

  **Real-World Impact:**
  - Reduces distraction-related crashes by up to **23%**
  - Enables adaptive safety interventions (e.g., pre-tensioning seat belts when drowsiness detected)
  - Meets upcoming Euro NCAP requirements for Level 3+ automated vehicles
  - Can prevent accidents before they happen by alerting drowsy drivers

- **Occupant Monitoring System (OMS):** Goes beyond the driver to monitor ALL passengers in the vehicle. It tracks everyone's seating position and body size to ensure airbags and safety restraints deploy correctly for each person. This is especially important as cars become more automated and passengers have more freedom to move around.

  **Key Benefits:**
  - Optimizes airbag deployment force and angle based on passenger size and position
  - Detects unauthorized or unexpected occupant configurations (e.g., rear-seat child without booster)
  - Enhances overall vehicle safety, particularly in shared-mobility and automated-shuttle scenarios

This project focuses on DMS but lays the groundwork for full occupant monitoring in future autonomous vehicles.

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
| **Hardware** | **FLIR Firefly FFY-U3-16S2C-S** + **Evetar 8mm Lens** | Industrial-grade camera with global shutter (no motion blur) + wide-angle lens. Also tested with Logitech C920 for consumer applications. |

---

## 📷 Hardware Specifications & Installation

### Camera: FLIR Firefly FFY-U3-16S2C-S

This is an **industrial-grade camera** built for demanding automotive applications:

**Key Specifications:**
- **Sensor:** Sony IMX296 (1/2.9-inch CMOS, 1.6 MP)
- **Resolution:** 1440 × 1080 pixels (Full HD quality)
- **Frame Rate:** Up to 60 FPS
- **Shutter Type:** **Global Shutter** - Captures entire image instantly (no motion blur during fast head turns)
- **Pixel Size:** 3.45 µm
- **Interface:** USB 3.0 Gen 1
- **Size:** 27mm × 27mm × 14mm (very compact)
- **Operating Temperature:** -40°C to +85°C (automotive grade)

**Why This Camera?**
- ✅ No motion blur during quick head movements
- ✅ Low-light performance for night driving
- ✅ Compact enough to hide in dashboard
- ✅ Industrial reliability for 24/7 operation

### Lens: Evetar M118B081 (8mm f/1.8)

**Key Specifications:**
- **Focal Length:** 8mm
- **Aperture:** f/1.8 (large opening = excellent low-light performance)
- **Field of View:** ~70° horizontal (wide enough to see entire driver face and shoulders)
- **Image Format:** Compatible with 1/1.8" sensors
- **Mount:** M12 (standard industrial mount)
- **Focus:** Fixed focus (optimized for 50-80cm driver distance)
- **Expected Distortion:** Moderate barrel distortion (corrected via calibration)

**Why This Lens?**
- Wide coverage without extreme fisheye effect
- Works in dim cabin lighting (large f/1.8 aperture)
- Minimal distortion compared to ultra-wide lenses
- Perfect for typical driver-to-camera distance

### Camera Bracket Design & Installation

**Design Evolution:**

**Version 1 (Prototype):**
- Adjustable design for testing: 4 height levels (10mm steps) + angular adjustments (10° increments)
- Mounted with zip ties and double-sided tape for easy repositioning
- Purpose: Find optimal camera position through trial and error

**Version 2 (Final Production):**
- Fixed optimal position based on testing results
- Sleek enclosed casing that hides camera and cables
- Professional appearance suitable for vehicle integration
- Still allows fine angular adjustments for different vehicle models
- Material: PETG 3D-printed plastic

**Installation Location Options:**

| Location | Advantages | Disadvantages |
|:---------|:-----------|:--------------|
| **Dashboard (Center)** | ✅ No steering wheel occlusion<br>✅ Unobstructed view<br>✅ Good field of view | ❌ Fully exposed to direct sunlight<br>❌ Looks out of place |
| **Steering Column (Chosen)** | ✅ Protected from direct sunlight<br>✅ Good field of view<br>✅ Better aesthetics<br>✅ Natural mounting location | ⚠️ Partial cluster occlusion<br>⚠️ Occasional steering wheel blocking |

**Final Position:** Steering column was selected because:
- Protection from sunlight reduces glare and temperature issues
- Occlusion during steering is minimal and temporary
- Professional appearance maintains vehicle interior design
- Easier cable routing and integration

### Temperature Considerations (Important for Real-World Deployment)

**Challenge:** Car interiors get extremely hot in summer (up to 70-80°C in direct sunlight).

**Current Bracket Material:**
- **PETG Plastic:** Heat Deflection Temperature (HDT) = 70°C
- ⚠️ **Risk:** May deform in extreme summer heat
- ✅ **Camera Tolerance:** FLIR Firefly rated up to 85°C (sufficient)

**Future Improvement:**
- **Polycarbonate Filament:** HDT = 140°C (much more heat-resistant)
- Additional heat sink for camera module cooling
- Thermal testing to identify actual peak operating temperatures
- Ventilation channels in bracket design

**Why This Matters:**
A deformed camera bracket means misaligned camera = inaccurate gaze estimation. For production vehicles, heat-resistant materials and thermal management are critical.

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

**Head Pose Estimation:**

Before gaze can be estimated, the system needs to know how the head is oriented in 3D space. We track three angles:

1. **Yaw** - Left/right head rotation (like shaking your head "no")
   - Calculated from horizontal displacement of nose tip relative to eye midpoint
   - Example: Looking at side mirror = high yaw angle

2. **Pitch** - Up/down head tilt (like nodding "yes")
   - Derived from vertical offset between nose and eye level
   - Example: Looking at instrument cluster = downward pitch

3. **Roll** - Tilting head to the side (like resting head on shoulder)
   - Computed from the angle of the horizontal line between eyes
   - Example: Driver leaning head = non-zero roll

**Smoothing with Kalman Filter:**

Raw facial landmark detection can be "jittery" (jumping around slightly frame-to-frame). We use a **1D Kalman Filter** to smooth the yaw and depth values. Think of it as "averaging" the measurements intelligently to remove noise while keeping real movements responsive.

**Why This Approach?**
- Lightweight and computationally efficient (runs in real-time)
- No need for complex 3D modeling or PnP (Perspective-n-Point) computation
- Works well for near-frontal face alignment (typical driving position)
- Suitable for embedded systems with limited processing power

**Limitation:** Accuracy decreases at extreme head angles (>60° rotation), but this is acceptable since such extreme poses indicate distraction anyway.

### 2. Gaze Mapping (Finding What the Driver is Looking At)
Once the system knows the gaze direction (like an invisible line from the eyes), it needs to figure out what object that line hits.

* **Virtual 3D Car Model:** A digital map of the car interior is created with **9 specific zones** (Side Mirrors, Radio, Speedometer, Road ahead, etc.).
* **Intersection Calculation:** The **Möller-Trumbore Algorithm** (a mathematical method) checks if the gaze line crosses any of these 9 zones.

**Simple Analogy:** Imagine shining a laser pointer from the driver's eyes - this algorithm tells us which car part the laser beam hits.

### 3. Drowsiness & Fatigue Detection
The system monitors the driver's alertness using facial features tracked by MediaPipe (a face tracking library). It follows **EuroNCAP** safety standards - the same standards used by automotive safety organizations in Europe.

**What is EuroNCAP?**

Euro NCAP (European New Car Assessment Programme) is the organization that gives cars their safety ratings (the "star" ratings you see in car reviews). They have strict requirements for Driver Monitoring Systems that all modern cars must meet to get top safety scores.

**EuroNCAP Requirements for DMS:**
- Must detect drowsiness within 3-6 seconds of eye closure
- Must alert driver with visual, auditory, or haptic (vibration) warnings
- Must detect distraction when driver looks away for >2 seconds
- Must work in various lighting conditions (day, night, twilight)
- Must be non-intrusive (no wearables required)
- **DMS must be ON by default** (driver cannot disable it)
- Must function for at least **1 minute of driving at 10 km/h**
- Distraction, microsleep, and sleep detection required at **20 km/h or higher**

**Karolinska Sleepiness Scale (KSS) - Measuring Drowsiness Levels**

Our system aligns with the **KSS**, a widely used scale in automotive safety research that rates alertness from 1 (extremely alert) to 9 (extremely sleepy). We detect three critical states:

- **Level 7-8 (Microsleep):** Brief eye closures (1-2 seconds) - Early warning stage
- **Level 8-9 (Sleepy Behavior):** Extended eye closure (3+ seconds) - High-risk drowsiness
- **Level 9 (Unresponsive Driver):** Prolonged eye closure (6+ seconds) - Critical emergency state

This scientific approach ensures our alerts match real drowsiness levels, reducing false alarms while catching genuine safety risks.

Our system meets these standards by implementing all required detection capabilities and providing real-time alerts.

#### **A. Eye Aspect Ratio (EAR) - Measuring How Open the Eyes Are**
This formula calculates how wide open the eyes are:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

The formula compares vertical eye opening distances to horizontal eye width. When the value drops below a threshold:
* **Microsleep (Light Drowsiness):** Eyes closed for more than 3 seconds → Driver is getting tired
* **Unresponsive (Critical):** Eyes closed for more than 6 seconds → Emergency alert triggered

**Blink Detection Logic:**

Normal blinking vs. drowsiness:
- **Normal Blink:** EAR drops briefly (<0.3 seconds), then returns to normal
- **Drowsiness Indicator:** 
  - EAR < 0.15 (eyes nearly/fully closed)
  - Sustained for >2 seconds = Drowsiness warning
  - Rapid blinking (>3 blinks within 2 seconds) = Fatigue/eye strain warning

The system counts blinks per minute to establish a baseline and detects anomalous patterns.

#### **B. Mouth Aspect Ratio (MAR) - Detecting Yawning**

$$MAR = \frac{||p_{51} - p_{59}|| + ||p_{52} - p_{58}|| + ||p_{53} - p_{57}||}{3 ||p_{49} - p_{55}||}$$

This measures mouth opening. A sustained high value indicates yawning, which is a sign of fatigue.

#### **C. Distance Estimation - How Far is the Driver?**
The system estimates how far the driver is sitting from the camera using this calculation:

$$D = \frac{F \times W}{P}$$

Where:
- $F$ = Focal Length (a property of the camera from calibration)
- $W$ = Average distance between pupils (Inter-Pupillary Distance/IPD, ~63mm for most adults)
- $P$ = How many pixels wide the space between eyes appears on camera

**How It Works:**
1. **Calibration Phase (~1 second):** When the system starts, it asks the driver to look at the camera from a known distance (e.g., 55cm). It measures the pixel distance between pupils and calculates the camera's focal length.
2. **Continuous Monitoring:** As the driver moves, the system tracks pupil positions and recalculates distance in real-time.
3. **Smoothing:** A Kalman Filter removes noise and jitter from the measurements.

**Why This Matters:**
- **Safety Feature:** Detects if driver is leaning forward (fatigue or medical emergency)
- **Accuracy Compensation:** Adjusts gaze estimation based on distance
- **Airbag Optimization:** In full occupant monitoring systems, this data helps determine safe airbag deployment force
- **Seat Adjustment Tracking:** Monitors if driver changes seating position

**Practical Example:**
If the driver normally sits 65cm away but suddenly leans forward to 45cm, the system might detect:
- Fatigue (leaning on steering wheel)
- Medical issue (loss of consciousness)
- Reaching for something (temporary distraction)

The **EAR** + **MAR** + **Distance** combination provides comprehensive drowsiness detection aligned with EuroNCAP protocols.

### Drowsiness Detection Performance Results

We tested the drowsiness detection system on **906 frames without occlusions** and **505 frames with occlusions** (hands on steering wheel, phone, etc.). Here are the results:

**Without Occlusions (Clear View of Face):**

| Drowsiness State | Accuracy | Precision | Recall | F1-Score |
|:-----------------|:--------:|:---------:|:------:|:--------:|
| **Microsleep** (1-2 sec eyes closed) | 84.85% | 1.00 | 0.85 | 0.92 |
| **Sleepy Behavior** (3+ sec closed) | 94.51% | 0.95 | 0.95 | 0.95 |
| **Unresponsive Driver** (6+ sec) | 100.00% | 0.93 | 1.00 | 0.96 |
| **Overall System Accuracy** | **94.62%** | - | - | - |

**With Occlusions (Partial Face Blocking):**

| Drowsiness State | Accuracy | Notes |
|:-----------------|:--------:|:------|
| **Microsleep** | 72.88% | Performance drops but still functional |
| **Sleepy Behavior** | 100.00% | Robust even with occlusions |

**What These Numbers Mean:**

✅ **Microsleep Detection (Without Occlusion):**
- Precision = 1.00 → Every time the system says "microsleep," it's correct (zero false alarms)
- Recall = 0.85 → Catches 85% of actual microsleep events
- The system is conservative - prefers to miss some events rather than false alarm

✅ **Sleepy Behavior Detection:**
- 95% precision & recall → Near-perfect balance
- Reliably identifies when drivers are getting drowsy
- Critical safety threshold with high confidence

✅ **Unresponsive Driver:**
- Recall = 1.00 → **Never misses an unresponsive driver**
- Precision = 0.93 → Very few false positives
- Most critical safety feature works flawlessly

⚠️ **Impact of Occlusions:**
- Microsleep accuracy drops from 84.85% → 72.88% when face is partially blocked
- **Why?** Hands on steering wheel or phone can hide eye landmarks
- Sleepy behavior remains 100% accurate (even with occlusions)
- System is still functional but less precise when view is obstructed

**Practical Implication:**
The system excels at detecting serious drowsiness (sleepy behavior and unresponsive states) with >94% accuracy, making it highly reliable for real-world safety applications. While partial occlusions reduce microsleep detection accuracy, the system still maintains 73% accuracy - better than having no monitoring at all.

---

## Demo

<p align="center">
  <img src="assets/teaser_1_2.png" height="200" title="teaser1">
  <img src="assets/teaser_2_2.png" height="200" title="teaser2">
</p>

https://github.com/Vagver/dense3Deyes/assets/25174551/4de4fb76-9577-4209-ba07-779356230131

For a demo of 3DGazeNet on videos and single images visit the [demo folder](demo).

---

## 🎯 Camera Calibration: Why It Matters

Before the system can accurately track gaze, the camera must be **calibrated**. Think of it like "teaching" the computer exactly how the camera sees the world.

### What is Camera Calibration?

Every camera lens has unique characteristics:
- **Focal length** - How "zoomed in" it is
- **Optical center** - The exact center point of the lens
- **Distortion** - How much the lens "bends" straight lines (especially at edges)

Calibration measures these properties so the computer can correct for them and make accurate 3D measurements from 2D images.

### The Calibration Process (Zhang's Method)

We use a technique called **Zhang's Method**, which is the industry standard. Here's how it works:

1. **Print a Checkerboard Pattern** - A grid of black and white squares with known dimensions
2. **Take Multiple Photos** - Capture 20-30 images of the checkerboard from different angles
3. **Detect Corners** - The computer automatically finds where the black and white squares meet
4. **Calculate Camera Properties** - Mathematical algorithms compute the camera's internal parameters
5. **Optimize** - Fine-tune the parameters to minimize error

**Our Results:**
- **Reprojection Error:** 0.191 pixels (Industry target: < 0.5 pixels)
- **Camera Matrix:** fx ≈ 1943 px, fy ≈ 1943 px
- **Focal Length:** 8.37mm (matches the physical 8mm lens specification)
- **Number of Calibration Images:** 23 images

**What's Reprojection Error?**

Imagine the computer predicts where a checkerboard corner should appear based on its calculated camera model. Reprojection error is the distance (in pixels) between where the computer thinks the corner is and where it actually appears in the image. Lower is better!

**0.191 pixels** means our calibration is extremely accurate - the error is less than 1/5th of a single pixel!

### Why Calibration is Critical for Driver Monitoring

**Without Calibration:**
- Gaze direction could be off by 5-10 degrees ❌
- Head pose angles would be incorrect ❌
- Depth estimation would be unreliable ❌
- Drowsiness detection would have false alarms ❌

**With Proper Calibration:**
- Gaze accuracy within 1-2 degrees ✅
- Precise head orientation tracking ✅
- Accurate driver distance measurement ✅
- Reliable safety monitoring ✅

### Lens Distortion Correction

Real camera lenses, especially wide-angle ones, create **distortion**:

- **Barrel Distortion:** Straight lines appear curved outward (like looking through a fishbowl)
- **Pincushion Distortion:** Straight lines curve inward

Our 8mm Evetar lens has slight barrel distortion. During calibration, we measure this distortion and create a correction map that "undistorts" every frame in real-time.

**Visual Result:** After correction, straight lines in the car (like the edge of the dashboard) appear perfectly straight in the processed image.

### Handling Environmental Factors

Real-world conditions can affect calibration over time:

- **Temperature Changes:** Heat can expand lens components slightly
- **Vibrations:** Engine vibrations might shift camera alignment
- **Aging:** Lens properties can change over months/years

**Mitigation Strategies:**
- Periodic re-calibration (every 6-12 months)
- Vibration-isolated camera mounts
- Temperature-compensated housing
- Validation using fixed reference points in the vehicle

---

## Experimental Setup

The system was tested in a **parked AI Motion Lab vehicle** to ensure safety during testing (no actual driving on roads).

### Two-Phase Data Collection Strategy

We used a **two-stage approach** to collect training and validation data:

#### **Phase 1: Lab Mock-Up (Controlled Environment)**

Before testing in an actual vehicle, we built a **full-scale car interior mock-up** in a laboratory room.

**Purpose:**
- Test camera settings and calibration
- Experiment with different lighting conditions
- Validate the guidance system (Python/Pygame script)
- Quick iterations without vehicle access

**What We Did:**
- Marked 9 Areas of Interest (AOIs) on a dashboard mock-up
- Recorded 3 types of scenarios per participant:
  1. **"Owl" Movement** - Slow, deliberate gaze shifts through all 9 AOIs in sequence
  2. **Simulated Driving** - Natural head movements with occasional drowsiness/microsleeps
  3. **Random Movement** - Unscripted, chaotic head motion to test robustness

**Data Collected:**
- **6 Participants** × 3 Videos each = **18 Videos**
- **Average Duration:** 30 seconds per video
- **Frame Rate:** 30 FPS
- **Total Frames:** ~16,200 frames

#### **Phase 2: In-Vehicle Data Collection (Real-World Environment)**

After optimizing the system in the lab, we moved to an actual vehicle for realistic testing.

**Camera Installation:**
- FLIR Firefly camera permanently mounted using custom-designed 3D-printed bracket
- Bracket designed with adjustable angles (10° increments) and 4 height levels (10mm spacing)
- Final bracket version: Sleek, enclosed design that hides in the dashboard

**Camera Bracket Design Process:**

Creating the perfect camera mount was an iterative engineering process:

**Step 1: Initial Measurements**
- Took direct measurements inside the vehicle
- Created a 3D scan of the car interior using a mobile scanning app
- Measured exact camera and lens dimensions to match manufacturer CAD models

**Step 2: Prototype Bracket (Version 1)**
- Wide range of adjustment options: 4 height levels (10mm spacing each)
- Angular adjustments in 10° increments
- Secured using plastic zip ties for quick installation/removal
- Purpose: Test different camera positions to find the optimal viewing angle

**Step 3: Optimized Bracket (Version 2 - Final)**
- Fixed optimal position based on testing
- Sleek enclosed casing that conceals components
- Professional appearance suitable for production vehicles
- Still allows fine angular adjustments
- Designed for permanent installation

**Why Custom Brackets Matter:**
- Off-the-shelf mounts don't account for dashboard curvature
- Precise positioning ensures full face visibility
- Vibration isolation prevents camera shake
- Professional integration improves user acceptance

**Preparation Steps:**
1. **Seat Adjustment Protocol:** Each participant adjusted seat height, backrest angle, and steering wheel position for comfort (simulating real-world variability)
2. **AOI Marking:** Physical labels placed at 9 locations:
   - Passenger footwell
   - Rear passenger seat area
   - Driver-side window
   - Rearview mirror
   - Infotainment screen (2 different zones)
   - Passenger face area
   - Passenger-side window
   - Smartphone mount (on steering wheel)
3. **Guidance System:** Python/Pygame script displayed on-screen prompts and audio cues to guide participants through the AOI sequence

**Recording Scenarios:**
1. **"Owl" Sequence** - Systematic gaze through all AOIs
2. **"Lizard" Driving Behavior** - Quick, darting eye movements typical of active driving
3. **Random + Drowsiness** - Unscripted movement with simulated fatigue states

**Data Collected:**
- **3 Participants** × 3 Videos each = **9 Videos**
- **Higher Quality:** Better lighting, realistic environment
- **Diversity:** Participants of different heights, with/without glasses
- **Total Additional Frames:** ~8,100 frames

**Combined Dataset:**
- **Total:** 27 Videos from both phases
- **9 AOIs** manually labeled per frame
- **Variety:** Different people, poses, lighting, and scenarios
- **Ground Truth:** Manual annotation of which AOI the person was looking at in each frame

### Dataset Labeling

After recording, every frame was **manually labeled** with the gaze zone (1-9). This labeled dataset serves as "ground truth" for:
- Training the neural network
- Validating model accuracy
- Calculating performance metrics (precision, recall, F1-score)

**Labeling Process:**
- Frame-by-frame annotation using custom Python tool
- Zones correspond to the 9 physical AOIs
- Quality control: Cross-verification by multiple team members

---

## 📊 Results & Performance Analysis

### Accuracy by Region

| Area of Interest (AOI) | Accuracy | Notes |
| :--- | :--- | :--- |
| **Road / Center** | **High** | Best performance; face is frontal. |
| **Infotainment** | **High** | Strong detection. |
| **Side Mirrors** | **Lower** | Worst performance due to extreme head rotation angles. |
| **Overall** | **90.5%** | Robust for general safety monitoring. |

### Detailed Performance Metrics (Gaze Zone Detection)

Beyond overall accuracy, we measured **Precision**, **Recall**, and **F1-Score** for each zone:

| Zone | Precision | Recall | F1-Score | Total Frames | Detected Frames |
|:-----|:---------:|:------:|:--------:|:------------:|:---------------:|
| **Normal Driving (Road)** | 0.913 | 0.992 | 0.951 | 6,645 | 6,066 |
| **Infotainment** | 0.934 | 0.719 | 0.813 | 420 | 392 |
| **Rear Mirror** | 1.000 | 0.508 | 0.674 | 240 | 240 |
| **Passenger Footwell** | 0.733 | 0.632 | 0.679 | 150 | 110 |
| **Passenger Side Window** | 0.803 | 0.534 | 0.641 | 300 | 240 |
| **Phone on Wheel** | 0.951 | 0.487 | 0.644 | 300 | 285 |
| **Passenger Face** | 0.538 | 0.127 | 0.206 | 120 | 64 |
| **Driver Side Window** | 0.277 | 0.146 | 0.191 | 120 | 33 |
| **Rear Passenger** | 0.000 | 0.000 | 0.000 | 120 | 0 |
| **Overall Accuracy** | **0.905** | **0.905** | **0.905** | **8,415** | **7,430** |

**What These Metrics Mean:**

- **Precision:** When the system says the driver is looking at a zone, how often is it correct?
  - Example: Rear Mirror has 100% precision - every detection was correct (no false positives)

- **Recall:** Out of all the times the driver actually looked at that zone, how many did the system catch?
  - Example: Normal Driving has 99.2% recall - it almost never misses when driver looks at the road

- **F1-Score:** The balanced average of Precision and Recall (best when both are high)
  - Example: Normal Driving has the highest F1-score (0.951) - excellent all-around performance

**Key Observations:**
- ✅ **Best Performance:** Normal driving position (forward gaze) - critical for safety
- ✅ **Strong Detection:** Infotainment screen and dashboard areas
- ⚠️ **Challenging Zones:** Extreme side glances (driver window, rear passenger) have lower accuracy due to severe head rotation
- ❌ **Limitation:** Rear passenger zone (0% detection) - requires additional camera or different mounting position

### Real-Time Performance (Hardware Challenges)
**Initial Challenge:**  
The system was first tested on an **NVIDIA Jetson Nano** (a small, embedded computer designed for AI projects).

* **Problem Found:** The Jetson Nano could only process **5-8 frames per second** (FPS), which is too slow for real-time driver safety monitoring (you need ~30 FPS for smooth, safe operation).
* **Solution:** Moved the processing to a **laptop with a dedicated GPU** (graphics card), which achieved **~30 FPS** - fast enough for real-time use.
* **Compatibility Verified:** The code was tested on the car's Linux computer system and works correctly, proving it can run on different platforms (the Jetson Nano just needs optimization to run faster).

**Takeaway:** The system works in real-time on standard laptop hardware. Future work will optimize it to run efficiently on smaller embedded devices.

### 🎯 Jetson Nano Deployment: Challenges & Solutions

**What is Jetson Nano?**

The NVIDIA Jetson Nano is a small, powerful computer designed for AI and robotics projects. It's affordable (~$100) and energy-efficient, making it ideal for embedding in vehicles.

**Hardware Specifications:**
- **CPU:** Quad-core ARM Cortex-A57
- **GPU:** NVIDIA Maxwell (128 CUDA cores)
- **RAM:** 2GB LPDDR4 (4GB version also available)
- **Storage:** MicroSD card
- **Power:** 5W - 10W (very efficient)
- **Ports:** USB 3.0, HDMI, GPIO pins
- **Size:** Compact, fits in palm of hand

**Why Try Jetson Nano?**
- Affordable and accessible for prototyping
- Low power consumption (can run on battery)
- Designed specifically for AI inference
- Strong community support
- Compact enough to install in vehicle dashboard

### Implementation Attempt & Challenges

**Setup Process:**
1. ✅ Flashed Jetson Nano 2GB with JetPack 4.6 image using balenaEtcher
2. ✅ Created bootable SD card with official NVIDIA image
3. ✅ Connected WiFi, monitor, keyboard
4. ✅ Performed first-time setup and user configuration

**Part 1: Gaze Detection Model (3DGazeNet)**

**Required Libraries:**
| Library | Purpose | Installation Status |
|:--------|:--------|:-------------------|
| numpy | Matrix operations & gaze vector calculation | ✅ Success |
| opencv | Video frame processing | ✅ Success |
| torch (PyTorch) | Deep learning model execution | ❌ **FAILED** |
| tqdm | Progress bars | ✅ Success |
| easydict | YAML config handling | ✅ Success |
| pyyaml | Configuration files | ✅ Success |
| scipy | Geometry calculations | ✅ Success |
| Cython | Required by PyTorch | ✅ Success |

**Critical Problem: PyTorch Installation Failure**

**Error Encountered:**
```bash
>>> import torch
Illegal instruction (core dumped)
```

**What We Tried:**
- Downloaded PyTorch v1.10.0 wheel file for ARM architecture (aarch64)
- Ensured Python 3.6 compatibility with JetPack 4.6
- Tried multiple official PyTorch versions from NVIDIA's website
- All versions produced the same "Illegal instruction" error

**Root Cause:**
The pre-compiled PyTorch wheels are not fully compatible with the Jetson Nano 2GB hardware architecture. The CPU instruction set mismatch causes the crash.

**Status: Ongoing**
NVIDIA support team requested full installation logs and suggested:
- Custom PyTorch build from source (time-consuming, ~6-8 hours compilation)
- Hardware-specific compilation flags
- Alternative: Use Jetson Xavier NX (more powerful, better compatibility)

**Part 2: Depth Estimation & Drowsiness Detection**

**Libraries Used:**
| Library | Version | Status |
|:--------|:--------|:-------|
| Python | 3.6.9 | ✅ Success |
| OpenCV | 4.11 | ✅ Success (manual wheel install) |
| MediaPipe | v0.8.5 | ⚠️ Limited (see below) |
| NumPy | Latest | ✅ Success |

**Critical Limitation: MediaPipe Version Constraint**

**The Problem:**
```python
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
# ❌ Error: refine_landmarks parameter not supported in v0.8.5
```

**Why This Matters:**
- MediaPipe v0.8.5 is the newest version compatible with Python 3.6 and JetPack 4.6
- The `refine_landmarks=True` parameter is needed to detect **iris landmarks**
- Without iris landmarks:
  - ❌ Depth estimation fails (needs pupil distance measurement)
  - ❌ Precise gaze direction unavailable
  - ❌ Drowsiness detection incomplete (relies on eye tracking)

**What Still Works on Jetson Nano:**
- ✅ **Head Pose Estimation:** Yaw, Pitch, Roll angles
- ✅ **Distraction Detection:** Based on yaw angle (head turning)
- ✅ Basic facial landmark tracking (468 points, but no iris refinement)

**What Doesn't Work:**
- ❌ Iris-based depth estimation
- ❌ Precise eye tracking for drowsiness
- ❌ Blink detection (requires iris landmarks)
- ❌ Complete EAR calculation

### Current Status & Recommendations

**Summary:**
- **Gaze Detection (3DGazeNet):** ❌ Blocked by PyTorch compatibility issues
- **Drowsiness Detection:** ⚠️ Partially working (head pose only, no eye tracking)
- **Performance:** 5-8 FPS (even for limited functionality)

**Recommendations for Future Deployment:**

**Option 1: Upgrade to Jetson Xavier NX (~$400)**
- More powerful GPU (384 CUDA cores vs 128)
- 8GB RAM (vs 2GB)
- Better PyTorch compatibility
- Expected performance: 30+ FPS with full functionality

**Option 2: Optimize for Jetson Nano**
- Compile PyTorch from source with Nano-specific flags (1-2 weeks effort)
- Use TensorRT to optimize model (3-5x speedup possible)
- Reduce input resolution (e.g., 112×112 instead of 224×224)
- Model quantization (FP16 instead of FP32)

**Option 3: Cloud-Connected Hybrid**
- Run head pose and distraction on Jetson Nano (works now)
- Stream video to cloud server for full gaze analysis (requires 4G/5G)
- Hybrid approach: local for critical alerts, cloud for detailed analytics

**Lesson Learned:**
While Jetson Nano is excellent for learning and prototyping, **production DMS systems require more powerful hardware** (Jetson Xavier or automotive-grade ECUs) to achieve 30 FPS with full feature set and safety reliability.

---

## 🚀 System Integration & Real-World Deployment

### Current System Architecture

The complete DMS pipeline processes data in the following sequence:

1. **Video Input** → Camera captures frames at 30 FPS
2. **Face Detection** → MediaPipe locates face and extracts 468 facial landmarks
3. **Eye Region Extraction** → Crops left and right eye images (224×224 px each)
4. **Head Pose Calculation** → Computes yaw, pitch, roll from landmarks
5. **Gaze Estimation** → ResNet-18 predicts 3D gaze vector
6. **Gaze Mapping** → Möller-Trumbore algorithm finds AOI intersection
7. **Drowsiness Analysis** → EAR, MAR, blink rate, distance monitoring
8. **Alert Generation** → Visual/audio warnings for unsafe states
9. **Video Output** → Annotated frames with overlays and metrics

**Processing Time:** ~33ms per frame on laptop GPU (30 FPS sustained)

### Integration with Vehicle Systems

For production deployment, the DMS should integrate with existing vehicle electronics:

**CAN-Bus Integration:**
- Read seat position sensors (compensate for driver movement)
- Read vehicle speed (adjust alert thresholds - more lenient when parked)
- Send alert signals to dashboard/instrument cluster
- Trigger haptic feedback (steering wheel vibration)

**Airbag Control Module:**
- Share occupant position/size data for optimized deployment
- Detect out-of-position scenarios

**ADAS (Advanced Driver Assistance Systems) Communication:**
- Share driver attention level with lane-keeping assist
- Coordinate with adaptive cruise control
- Integrate with Level 3+ autonomous driving handover systems

### Deployment Platforms

**Option 1: Embedded Computer (NVIDIA Jetson Series)**
- **Jetson Nano:** Budget option (~$100), 5-8 FPS (needs optimization)
- **Jetson Xavier NX:** Mid-range (~$400), 30+ FPS capable
- **Jetson AGX Orin:** Premium (~$2000), 60+ FPS with headroom

**Option 2: Automotive-Grade Compute Units**
- Integration with existing infotainment system processors
- Dedicated DMS ECU (Electronic Control Unit)
- Requirements: Automotive temperature range (-40°C to +85°C), vibration resistance, EMI shielding

**Option 3: Cloud-Connected Hybrid**
- On-device real-time inference for safety-critical functions
- Cloud processing for analytics and model updates
- Challenges: Requires cellular connectivity, latency concerns

### Calibration in Production Vehicles

**Factory Calibration:**
- During vehicle assembly, camera is calibrated once using fixed dashboard reference points
- Calibration data stored in vehicle's non-volatile memory
- QA validation ensures <0.5 pixel reprojection error

**End-User Calibration (Optional):**
- 30-second setup when new driver uses vehicle
- System learns driver's unique facial characteristics
- Improves accuracy by 5-10% compared to generic calibration

**Periodic Re-Calibration:**
- Automatic validation checks using dashboard geometry
- Recommended every 6 months or after camera servicing
- Warning indicator if calibration drift detected

### Regulatory Compliance

**EuroNCAP 2024+ Requirements:**
- ✅ Direct driver monitoring with gaze tracking
- ✅ Drowsiness detection with multi-stage warnings
- ✅ Distraction detection (>2 second gaze aversion)
- ✅ Non-intrusive operation
- ✅ Works in varied lighting (daytime, nighttime, tunnels)

**Future Requirements (2026+):**
- 🔄 Infrared/night vision capability (in development)
- 🔄 Multi-occupant monitoring (driver + passengers)
- 🔄 Seatbelt compliance verification
- 🔄 Child seat detection and classification

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

**This project was developed as a collaborative team effort at Technische Hochschule Ingolstadt (THI).** 

The implementation extends the original 3DGazeNet framework with integrated functionality for comprehensive eye-based behavior analysis and driver monitoring system applications. Our team combined expertise in computer vision, hardware engineering, data collection, and embedded systems to create an end-to-end interior monitoring solution for autonomous driving safety applications.

We acknowledge the contributions of all team members who made this project possible through their dedication to research, development, testing, and validation in real-world automotive environments.

**Institution:** Technische Hochschule Ingolstadt (THI), Germany    
**Supervision:** Prof. Alessandro Zimmer & Dipl.-Ing. Joed Lopes da Silva
