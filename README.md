# Drowning Detection Project

This project implements a drowning detection system using YOLOv8, PyTorch with CUDA 12.5, Ultralytics, Roboflow, and OpenCV.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Training the Model](#training-the-model)
5. [Running Inference](#running-inference)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA 12.5
- Python 3.8+

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yashpotdar-py/drowning-detection-system.git
   ```

   ```bash
   cd drowning-detection-system
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On Linux or macOS use:

   ```bash
   source .venv/bin/activate
   ```

   On Windows use:

   ```shell
   .venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

   Verify PyTorch installation:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   ```bash
   pip3 install roboflow python-dotenv ipykernel
   ```

## Dataset Preparation

1. Create a Roboflow account and upload your drowning detection dataset.
2. Export the dataset in YOLOv8 format.
3. Download the dataset to your project directory.

## Training the Model

1. Prepare your dataset configuration file `data.yaml`:

   path: path/to/your/dataset
   train: train/images
   val: valid/images
   test: test/images

   nc: 3 # number of classes
   names: [- Active drowning, Possible Passive Drowner, Swimming] # class names

2. Train the model:

   ```bash
   yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
   ```

## Running Inference

1. For image inference:

   ```bash
   yolo task=detect mode=predict model=path/to/best.pt source=path/to/image.jpg
   ```

2. For video inference:

   ```bash
   yolo task=detect mode=predict model=path/to/best.pt source=path/to/video.mp4
   ```

3. For webcam inference:

   ```bash
   yolo task=detect mode=predict model=path/to/best.pt source=0 # 0 for default webcam
   ```

## Troubleshooting

- If you encounter CUDA-related errors, ensure that your NVIDIA drivers and CUDA toolkit are properly installed and compatible with PyTorch.
- For any other issues, please check the official documentation of YOLOv8, Ultralytics, and Roboflow.

For more information and advanced usage, refer to the [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/).
