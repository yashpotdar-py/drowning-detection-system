# Drowning Detection System

This project aims to create an AI-based drowning detection system using YOLOv8, capable of detecting and classifying different stages of drowning or swimming activity from drone footage. The system is trained on a custom dataset and leverages object detection techniques to identify potential drowning victims in real-time.

## Table of Contents

- [Drowning Detection System](#drowning-detection-system)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Directory Structure](#directory-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup Environment](#setup-environment)
  - [Usage](#usage)
    - [Data Collection](#data-collection)
    - [Data Exploration](#data-exploration)
    - [Model Training](#model-training)
    - [Model Evaluation](#model-evaluation)
    - [Model Export](#model-export)
  - [Dataset](#dataset)
  - [TODO](#todo)
    - [1. Webcam Interface Integration](#1-webcam-interface-integration)
    - [2. Raspberry Pi Optimization](#2-raspberry-pi-optimization)
    - [3. Frontend Development](#3-frontend-development)
    - [4. Model Performance Optimization](#4-model-performance-optimization)
    - [5. Documentation \& Testing](#5-documentation--testing)
  - [License](#license)

---

## Project Overview

This system uses the YOLOv8 model to detect and classify different states of a person in water:

1. **Active Drowning**
2. **Possible Passive Drowning**
3. **Swimming**

The project involves:

- Collecting and labeling drone footage.
- Training the model using YOLOv8.
- Evaluating and exporting the trained model for deployment.

---

## Directory Structure

```bash
📦drowning-detection-system
 ┣ 📂.git                  # Git version control
 ┣ 📂.venv                 # Python virtual environment
 ┣ 📂data                  # Contains dataset for training, validation, and testing
 ┃ ┣ 📂train               # Training data
 ┃ ┣ 📂test                # Testing data
 ┃ ┣ 📂valid               # Validation data
 ┃ ┗ 📜data.yaml           # YOLOv8 data config file
 ┣ 📂models                # Contains model checkpoints and predictions
 ┣ 📂notebooks             # Jupyter Notebooks for different project stages
 ┣ 📜best.onnx             # Best model exported to ONNX format
 ┣ 📜.env                  # Environment variables
 ┣ 📜.gitignore            # Ignored files by Git
 ┣ 📜LICENSE               # Project License
 ┗ 📜README.md             # Project documentation
```

---

## Installation

### Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow API Key](https://roboflow.com/)

### Setup Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/yashpotdar-py/drowning-detection-system.git
   cd drowning-detection-system
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # If you have a CUDA enabled GPU (CUDA 12.4 or above)
   pip3 install torch torchvision torchaudio # If you don't have a CUDA enabled GPU or CUDA version below 12.2
   pip3 install roboflow pandas numpy matplotlib
   ```

3. Set up environment variables by creating a `.env` file:
   ```bash
   ROBOFLOW_API_KEY=your_roboflow_api_key
   ```

---

## Usage

### Data Collection

To collect data, run the `01_Collecting_Data.ipynb` notebook. This notebook uses the Roboflow API to download the dataset:

```bash
cd notebooks
jupyter notebook 01_Collecting_Data.ipynb
```

### Data Exploration

To explore the dataset, including class distribution, image resolutions, and bounding box statistics, run `02_Exploring_Data.ipynb`:

```bash
jupyter notebook 02_Exploring_Data.ipynb
```

### Model Training

To train the YOLOv8 model on the dataset, run `03_Model_Training.ipynb`:

```bash
jupyter notebook 03_Model_Training.ipynb
```

### Model Evaluation

To evaluate the trained model and generate predictions, use `04_Model_Evaluation_and_Exporting.ipynb`:

```bash
jupyter notebook 04_Model_Evaluation_and_Exporting.ipynb
```

### Model Export

Once the model is trained, you can export it to different formats such as ONNX for deployment:

```bash
jupyter notebook 04_Model_Evaluation_and_Exporting.ipynb
```

This will generate the `best.onnx` model file for use in production systems.

---

## Dataset

The dataset used in this project is split into training, validation, and test sets:

- **Training Set**: Located in `data/train/`
- **Validation Set**: Located in `data/valid/`
- **Test Set**: Located in `data/test/`

The dataset includes labels for three classes:

1. **Active Drowning**
2. **Possible Passive Drowning**
3. **Swimming**

The dataset's details and source can be found in `data/README.dataset.txt` and `data/README.roboflow.txt`.

---

## TODO

### 1. Webcam Interface Integration

- [ ] Implement a webcam interface to stream real-time video for detection.
- [ ] Integrate the **best.onnx** model to process the live webcam feed and perform detection.
- [ ] Add an option to switch between video file input and webcam feed for real-time detection.
- [ ] Ensure low-latency, real-time detection on the webcam feed with smooth bounding box visualization.

### 2. Raspberry Pi Optimization

- [ ] Optimize the code and model to run efficiently on a Raspberry Pi:
  - [ ] Convert the **best.onnx** model to a more lightweight format (e.g., TensorFlow Lite, PyTorch Mobile).
  - [ ] Implement hardware acceleration using the Raspberry Pi GPU (e.g., OpenCV with GPU support).
  - [ ] Optimize resource usage (memory, CPU, etc.) for smoother processing.
- [ ] Reduce the input image size and adjust detection confidence thresholds to balance accuracy and performance.
- [ ] Test the system thoroughly on a Raspberry Pi to minimize lag and eliminate hiccups.

### 3. Frontend Development

- [ ] Design and develop a user-friendly web interface for the drowning detection system.
- [ ] Integrate live monitoring dashboard for real-time detection and alerts.
- [ ] Display detection results (bounding boxes, confidence scores) on the frontend.
- [ ] Include functionality to view recent detection history and performance analytics (e.g., graphs, confusion matrix).

### 4. Model Performance Optimization

- [ ] Fine-tune hyperparameters (batch size, frame rate, confidence threshold) to optimize detection speed without sacrificing accuracy.
- [ ] Implement post-processing techniques to reduce false positives and enhance detection reliability.
- [ ] Test with various webcam resolutions and optimize the detection pipeline for different lighting conditions.

### 5. Documentation & Testing

- [ ] Create a clear step-by-step setup guide for deploying the system on a Raspberry Pi.
- [ ] Provide testing instructions to validate the webcam interface and ensure smooth performance on low-power devices.
- [ ] Add troubleshooting steps for common issues like lag, detection errors, or webcam feed problems.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
