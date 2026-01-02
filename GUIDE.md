# DeepSteer-Vis Developer Guide

## SYSTEM ARCHITECTURE

The system consists of three main components:
1.  **Data Processing Pipeline** (`utils.py`)
2.  **Behavioral Cloning Model** (`model.py`)
3.  **Real-time Drive Server with Explainability** (`drive.py`)

### 1. Data Processing Pipeline
-   **Input**: Images from Udacity Simulator (Center, Left, Right cameras).
-   **Preprocessing**:
    -   **Cropping**: Removes top 70px (sky) and bottom 25px (hood) to reduce noise.
    -   **Resizing**: Scales to 200x66 (NVIDIA PilotNet standard).
    -   **YUV Conversion**: Matches the color space used in the original NVIDIA research paper.
-   **Augmentation**:
    -   **Steering Correction**: `left_angle = center + 0.2`, `right_angle = center - 0.2`.
    -   **Random Flip**: Horizontally flips images and negates steering angle.

### 2. Model Architecture (Modified NVIDIA PilotNet)
We introduced **Hyper-Regularization** to prevent overfitting on the simple simulator tracks.

| Layer | Type | Specifications | Activation | Regularization |
| :--- | :--- | :--- | :--- | :--- |
| Input | Lambda | Norm: `x/127.5 - 1.0` | - | - |
| 1 | Conv2D | 24 filters, 5x5, stride 2 | ELU | Dropout (0.5) |
| 2 | Conv2D | 36 filters, 5x5, stride 2 | ELU | Dropout (0.5) |
| 3 | Conv2D | 48 filters, 5x5, stride 2 | ELU | Dropout (0.5) |
| 4 | Conv2D | 64 filters, 3x3 | ELU | Dropout (0.5) |
| 5 | Conv2D | 64 filters, 3x3 | ELU | Dropout (0.5) |
| 6 | Flatten | - | - | - |
| 7 | Dense | 100 units | ELU | Dropout (0.5) |
| 8 | Dense | 50 units | ELU | Dropout (0.5) |
| 9 | Dense | 10 units | ELU | Dropout (0.5) |
| Output| Dense | 1 unit (Steering) | Linear | - |

### 3. Saliency Map Visualization (Grad-CAM)
The "Competitive Edge" of this project is the real-time explainability layer.
-   **Method**: Gradient-weighted Class Activation Mapping (Grad-CAM).
-   **Logic**:
    1.  Compute the gradient of the steering output with respect to the last convolutional layer.
    2.  Pool gradients to get channel importance weights.
    3.  Compute weighted sum of feature maps to generate a generated heatmap.
    4.  Overlay this heatmap on the camera feed in real-time.

---

## TRAINING GUIDE

1.  **Collect Data**: Drive the car in Udacity Simulator (Training Mode) for 2-3 laps.
    -   Drive smoothly in the center of the lane.
    -   Record recovery laps (drive from side to center).
2.  **Prepare Data**: Ensure `driving_log.csv` and `IMG/` folder are in `data/`.
3.  **Train**:
    ```bash
    python model.py
    ```
    -   Checkpoints are saved as `model.h5` if validation loss improves.

---

## DEPLOYMENT GUIDE

1.  **Start Server**:
    ```bash
    python drive.py model.h5
    ```
2.  **Launch Simulator**: Open Udacity Simulator and select **Autonomous Mode**.
3.  **Monitor**:
    -   Terminal: Shows Steering Angle, Throttle, Speed.
    -   Window: "DeepSteer-Vis" window shows the car's camera view with Saliency Map overlay.

## TROUBLESHOOTING

-   **Car drives in circles?** Check if you trained on Track 2 but are testing on Track 1, or flip augmentation is broken (checked: it's implemented).
-   **OOM Errors?** Reduce `batch_size` in `model.py` (default: 32).
-   **Latency?** Real-time Grad-CAM is compute-intensive. If laggy, disable the visualization in `drive.py`.
