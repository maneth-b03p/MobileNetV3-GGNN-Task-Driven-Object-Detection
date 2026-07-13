# Task-Driven Object Detection with MobileNetV3 Backbone

## Overview
This repository presents a modified implementation of a task-driven object detection framework, where the original ResNet101 backbone is replaced with MobileNetV3. The model has to figure out the best object in the scene for a given task.

The goal of this modification is to reduce computational complexity and parameter count, making the model more suitable for resource-constrained and edge computing environments.

## Original Work
This project is based on the following repository:

https://github.com/yassersouri/task-driven-object-detection

Most of the core implementation and structure are derived from the original work. Proper credit goes to the original authors for their contribution.

## Pipeline
The general architecture follows a task-driven object detection framework. Different configurations explored:

- **Original pipeline:** RCNN/YOLO → ResNet101 → GGNN  
- **This repository (main):** YOLOv8n → MobileNetV3 → GGNN  
- **Refinement01_2 branch:** PicoDet-L → MobileNetV3 → GGNN  
  - PicoDet-L lightweight object detector (optimized for edge devices)
  - MobileNetV3 replaces ResNet101 for efficiency  
  - Maintains the graph reasoning step (GGNN)  
  - Reduces parameter count and computation for edge deployment
  - ONNX Runtime support for cross-platform inference

## My Contributions
- Replaced ResNet101 backbone with MobileNetV3
- **Refinement01_2 branch:** Integrated PicoDet-L as object detector (replacing YOLOv8n)
  - PicoDet-L ONNX model for improved efficiency
  - CUDA/CPU support via ONNX Runtime
- Implemented modified graph network components:
  - `mobilenet_graphnetwork.py`
  - `ggnn_mobilenetv3.py`
- Adapted the pipeline to support lightweight feature extraction
- Created web app interface for task-driven inference on user-uploaded images
- Conducted initial experiments (to be expanded)

## Motivation
Deep object detection models often rely on heavy backbones such as ResNet101, which are computationally expensive. This work explores whether a lightweight alternative like MobileNetV3 can maintain or improve accuracy while significantly reducing computational requirements. The **Refinement01_2 branch** further optimizes this by replacing YOLOv8n with PicoDet-L, a model specifically designed for edge device deployment.

### Hardware Acceleration Opportunity
Both PicoDet-L and MobileNetV3-Large utilize **depthwise separable convolutions** as a core architectural component. This structural similarity enables the design and implementation of a **unified hardware accelerator** that can efficiently execute both models on edge devices. A single accelerator optimized for depthwise convolution operations can significantly improve inference throughput and reduce power consumption across the entire detection pipeline.

## Project Structure
```
src/
├── coco_tasks/ 
│   ├── graph_datasets.py 
│   ├── graph_experiments.py
│   ├── mobilenet_graphnetwork.py (modified)
│   └── ...
├── task_detection_full.py (training and validation with PicoDet-L)
├── app.py (web app interface with YOLOv8n)
└── ggnn_mobilenetv3.py (modified)
```

## Key Dependencies

### Refinement01_2 Branch (PicoDet-L)
```bash
pip install onnxruntime torch torchvision
pip install opencv-python tqdm pycocotools
pip install sentence-transformers
```

The PicoDet-L ONNX model (~14MB) is automatically downloaded on first run.

### Main Branch (YOLOv8n)
```bash
pip install ultralytics torch torchvision
pip install opencv-python tqdm pycocotools
pip install sentence-transformers
```

## Notes on Compatibility

- The original implementation uses older PyTorch versions.
- Some minor modifications were required for compatibility with newer versions:
  - Updated tensor operations and model loading
  - Refinement01_2: ONNX Runtime integration for PicoDet-L inference

## Usage

### Training and Validation (Refinement01_2 - PicoDet-L)

Run the main script for training and validation with PicoDet-L:

```bash
python3 src/task_detection_full.py \
    --test-on-gt False \
    --detector yolo \
    --only-test False
```

**Available Options:**

- `--test-on-gt`: 
  - `False`: Use object detection model (PicoDet-L in Refinement01_2)
  - `True`: Use ground truth bounding boxes

- `--detector`: Choose object detection backend
  - `yolo` (default): PicoDet-L ONNX for lightweight, edge-optimized detection
  - `faster_rcnn`: Faster R-CNN (original implementation)

- `--only-test`:
  - `True`: Run inference only (no training)
  - `False`: Run full training pipeline

- `--random-seed`: Random seed for reproducibility (default: 0)
- `--fusion`: Graph fusion strategy - `none` or `avg` (default: none)
- `--weighted-aggregation`: Use weighted aggregation (default: True)

### Web App Interface

Run the interactive web application to upload an image and get task-driven object detection results:

```bash
python3 src/app.py
```

**Note:** The web app (`app.py`) currently uses YOLOv8n. To use PicoDet-L in the app, integration updates are required.

The app provides:
- Image upload interface
- Real-time object detection with MobileNetV3 backbone
- Task-specific object highlighting
- Visualization of detected objects and their task relevance scores

## Future Work
- Detailed performance comparison with ResNet101
- Inference time benchmarking on edge devices
- PicoDet-L integration in web app (`app.py`)
- Unified hardware accelerator design for depthwise convolution operations across PicoDet-L and MobileNetV3-Large
- Deployment on embedded platforms (Raspberry Pi, Jetson Nano, etc.)
- Mathematical formulation of the implemented model

## Acknowledgment
This work builds upon the original implementation by the authors of the task-driven object detection framework. Their contribution is gratefully acknowledged.

## License
This project follows the MIT License of the original repository.
