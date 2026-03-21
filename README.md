# Task-Driven Object Detection with MobileNetV3 Backbone

## Overview
This repository presents a modified implementation of a task-driven object detection framework, where the original ResNet101 backbone is replaced with MobileNetV3.

The goal of this modification is to reduce computational complexity and parameter count, making the model more suitable for resource-constrained and edge computing environments.

## Original Work
This project is based on the following repository:

https://github.com/yassersouri/task-driven-object-detection

Most of the core implementation and structure are derived from the original work. Proper credit goes to the original authors for their contribution.

## Pipeline
The general architecture follows a task-driven object detection framework. Different configurations explored:

- **Original pipeline:** RCNN/YOLO → ResNet101 → GGNN  
- **This repository:** RCNN/YOLO → MobileNetV3 → GGNN  
  - MobileNetV3 replaces ResNet101 for efficiency  
  - Maintains the graph reasoning step (GGNN)  
  - Reduces parameter count and computation for edge deployment
 
## My Contributions
- Replaced ResNet101 backbone with MobileNetV3
- Implemented modified graph network components:
  - `mobilenet_graphnetwork.py`
  - `ggnn_mobilenetv3.py`
- Adapted the pipeline to support lightweight feature extraction
- Conducted initial experiments (to be expanded)

## Motivation
Deep object detection models often rely on heavy backbones such as ResNet101, which are computationally expensive. This work explores whether a lightweight alternative like MobileNetV3 can maintain performance while significantly improving efficiency.

## Project Structure
```
src/
├── coco_tasks/ 
│ ├── graph_datasets.py 
│ ├── graph_experiments.py
│ ├── mobilenet_graphnetwork.py (modified)
│ └── ...
├── ggnn_mobilenetv3.py (modified)
```

## Notes on Compatibility

- The original implementation uses older PyTorch versions.
- Some minor modifications were required for compatibility with newer versions:

## Usage
(To be updated)

## Future Work
- Detailed performance comparison with ResNet101
- Inference time benchmarking
- Deployment on edge devices
- Mathematical formulation of the implemented model

## Acknowledgment
This work builds upon the original implementation by the authors of the task-driven object detection framework. Their contribution is gratefully acknowledged.

## License
This project follows the MIT License of the original repository.
