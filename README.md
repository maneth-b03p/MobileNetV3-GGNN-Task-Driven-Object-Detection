# Task-Driven Object Detection with MobileNetV3 Backbone

## Overview
This repository presents a modified implementation of a task-driven object detection framework, where the original ResNet101 backbone is replaced with MobileNetV3. The model has to figure out the best object to focus on for a specific task using graph neural networks.

The goal of this modification is to reduce computational complexity and parameter count, making the model more suitable for resource-constrained and edge computing environments.

## Original Work
This project is based on the following repository:

https://github.com/yassersouri/task-driven-object-detection

Most of the core implementation and structure are derived from the original work. Proper credit goes to the original authors for their contribution.

## Pipeline
The general architecture follows a task-driven object detection framework. Different configurations explored:

- **Original pipeline:** RCNN/YOLO → ResNet101 → GGNN  
- **This repository:** RCNN/YOLO → MobileNetV3 → GGNN  
  - YOLOv8 (default) or Faster R-CNN for object detection
  - MobileNetV3 replaces ResNet101 for efficiency  
  - Maintains the graph reasoning step (GGNN)  
  - Reduces parameter count and computation for edge deployment
 
## My Contributions
- Replaced ResNet101 backbone with MobileNetV3
- Integrated YOLOv8 as default object detector (Faster R-CNN support maintained)
- Implemented modified graph network components:
  - `mobilenet_graphnetwork.py`
  - `ggnn_mobilenetv3.py`
- Adapted the pipeline to support lightweight feature extraction
- Created web app interface for task-driven inference on user-uploaded images
- Conducted initial experiments (to be expanded)

## Motivation
Deep object detection models often rely on heavy backbones such as ResNet101, which are computationally expensive. This work explores whether a lightweight alternative like MobileNetV3 can maintain performance while significantly reducing computational cost. The integration of YOLOv8 provides a modern, efficient object detection pipeline suitable for real-time applications.

## Project Structure
```
src/
├── coco_tasks/ 
│   ├── graph_datasets.py 
│   ├── graph_experiments.py
│   ├── mobilenet_graphnetwork.py (modified)
│   └── ...
├── task_detection_full.py (training and validation)
├── app.py (web app interface)
└── ggnn_mobilenetv3.py (modified)
```

## Notes on Compatibility

- The original implementation uses older PyTorch versions.
- Some minor modifications were required for compatibility with newer versions:
  - Updated tensor operations and model loading
  - YOLOv8 integration for modern object detection

## Usage

### Training and Validation

Run the main script for training and validation:

```bash
python3 src/task_detection_full.py \
    --test-on-gt False \
    --detector yolo \
    --only-test False
```

**Available Options:**

- `--test-on-gt`: 
  - `False`: Use object detection model (YOLOv8 or Faster R-CNN)
  - `True`: Use ground truth bounding boxes

- `--detector`: Choose object detection backend
  - `yolo` (default): YOLOv8 for real-time detection
  - `faster_rcnn`: Faster R-CNN (original implementation)

- `--only-test`:
  - `True`: Run inference only (no training)
  - `False`: Run full training pipeline

### Web App Interface

Run the interactive web application to upload an image and get task-driven object detection results:

```bash
python3 src/app.py
```

The app provides:
- Image upload interface
- Real-time object detection with MobileNetV3 backbone
- Task-specific object highlighting
- Visualization of detected objects and their task relevance scores

## Future Work
- Detailed performance comparison with ResNet101
- Inference time benchmarking
- Deployment on edge devices
- Mathematical formulation of the implemented model

## Acknowledgment
This work builds upon the original implementation by the authors of the task-driven object detection framework. Their contribution is gratefully acknowledged.

## License
This project follows the MIT License of the original repository.
