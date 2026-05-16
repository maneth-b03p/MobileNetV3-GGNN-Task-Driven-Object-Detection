"""
# Modified by Maneth Banula Perera (2026)
# Description: MobileNetV3-based graph network execution runner with real-time YOLOv8 evaluation
"""

import json
import os
import random
from contextlib import redirect_stdout

import click
import numpy as np
import torch
from boltons.fileutils import mkdir_p
from torch.optim import SGD
from tqdm import tqdm

from coco_tasks.graph_datasets import JointCocoTasks, CocoTasksTest, CocoTasksTestGT, task_number_to_task_id
from coco_tasks.graph_experiments import JointGraphExperiment
from coco_tasks.mobilenet_graph_networks import (
    GGNNDiscLoss,
    InitializerMul,
    AllLinearAggregator,
    OutputModelFirstLast,
    AllLinearAggregatorWeightedWithDetScore,
)
from coco_tasks.settings import SAVING_DIRECTORY, TASK_NUMBERS
from coco_tasks.single_task_experiments import get_device
from coco_tasks.single_task_datasets import get_image_file_name

from pycocotools.cocoeval import COCOeval


# Try importing Ultralytics for real-time YOLOv8 object detection
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics to use the YOLOv8 detector stage: pip install ultralytics")


# ── COCO category mapping lookup (Maps YOLOv8 0-79 output indices to official 1-91 COCO IDs) ──
YOLO_TO_COCO_MAPPING = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


def run_yolo_inference_on_db(test_db, yolo_model):
    """
    Directly loops over valid test image paths,
    executes real-time YOLOv8 inference on CPU, transforms bounding box coordinates, 
    and handles category ID remapping back to COCO 1-indexed format.
    """
    print("Executing real-time object detection via YOLOv8 backbone model...")
    per_image_detections = {}
    
    # Iterate over valid images registered in the test database structure
    for img_id in tqdm(test_db.list_of_valid_images, desc="YOLOv8 Detection"):
        img_dict = test_db.task_coco.loadImgs(img_id)[0]
        
        # Resolve the image path using folder registry criteria
        # Matches logic found inside coco_tasks.single_task_datasets.get_image_file_name
        img_path = get_image_file_name(img_dict)
        
        # Execute real-time prediction using underlying ultralytics context
        results = yolo_model.predict(img_path, verbose=False, device='cpu')[0]
        
        img_detections = []
        boxes = results.boxes
        
        for box in boxes:
            # YOLOv8 returns coordinates as xyxy absolute format
            xyxy = box.xyxy[0].tolist()
            xmin, ymin, xmax, ymax = xyxy
            
            # Transformation matrix: absolute [xmin, ymin, xmax, ymax] -> [xmin, ymin, width, height]
            width = xmax - xmin
            height = ymax - ymin
            coco_bbox = [xmin, ymin, width, height]
            
            score = float(box.conf[0].item())
            yolo_cls = int(box.cls[0].item())
            
            # Resolve alignment mismatch by checking mapping array boundaries
            if yolo_cls < len(YOLO_TO_COCO_MAPPING):
                coco_category_id = YOLO_TO_COCO_MAPPING[yolo_cls]
            else:
                coco_category_id = yolo_cls + 1  # Fallback safety margin
                
            img_detections.append({
                "image_id": int(img_id),
                "category_id": int(coco_category_id),
                "bbox": coco_bbox,
                "score": score
            })
            
        per_image_detections[img_id] = img_detections
        
    return per_image_detections


@click.command()
@click.option("--random-seed", envvar="SEED", default=0)
@click.option("--test-on-gt", type=bool, default=False)  # Defaulting to False to test YOLOv8 pipeline directly
@click.option("--only-test", type=bool, default=False)
@click.option("--overfit", type=bool, default=False)
@click.option("--fusion", type=click.Choice(choices=["none", "avg"]), default="none")
@click.option("--weighted-aggregation", type=bool, default=True)
@click.option("--detector", type=click.Choice(["faster_rcnn", "yolo"]), default="yolo")

def main(random_seed, test_on_gt, only_test, overfit, fusion, weighted_aggregation, detector):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    n_epochs = 3
    lr = 1e-2
    wd = 0
    lr_scheduler = False

    # graph settings
    h_dim = 128
    x_dim = 128
    c_dim = 90
    max_steps = 3

    phi_dim = 128

    # Using task 8 oversampling parameter optimization to fix sparse data bias
    train_db = JointCocoTasks(oversample_tasks={8: 4})
    initializer = InitializerMul(h_dim=h_dim, phi_dim=phi_dim, c_dim=c_dim)

    if weighted_aggregation:
        aggregator = AllLinearAggregatorWeightedWithDetScore(
            in_features=h_dim, out_features=x_dim
        )
        print("Weighted Aggregation is used for GGNN")
    else:
        aggregator = AllLinearAggregator(in_features=h_dim, out_features=x_dim)
        print("Linear Aggregation is used for GGNN")

    output_model = OutputModelFirstLast(h_dim=h_dim, num_tasks=len(TASK_NUMBERS))

    network = GGNNDiscLoss(
        initializer=initializer,
        aggregator=aggregator,
        output_model=output_model,
        max_steps=max_steps,
        h_dim=h_dim,
        x_dim=x_dim,
        class_dim=c_dim,
        fusion=fusion,
    )

    optimizer = SGD(network.parameters(), lr=lr, weight_decay=wd)
    experiment = JointGraphExperiment(
        network=network,
        optimizer=optimizer,
        dataset=train_db,
        tensorboard=True,
        seed=random_seed,
    )

    train_folder = "ggnn-mobilenet-seed:{s}".format(s=random_seed)
    folder = os.path.join(SAVING_DIRECTORY, train_folder)
    mkdir_p(folder)

    if not only_test:
        checkpoint_path = os.path.join(folder, "model.mdl")
        if os.path.exists(checkpoint_path):
            print("Resuming from checkpoint - loading saved weights...")
            network.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-3
            lr_scheduler = False
            print("Resuming at lr=1e-3")
        else:
            print("No checkpoint found - training from scratch...")

        experiment.train_n_epochs(n_epochs, overfit=overfit, lr_scheduler=lr_scheduler)
        torch.save(network.state_dict(), os.path.join(folder, "model.mdl"))
    else:
        network.load_state_dict(
            torch.load(os.path.join(folder, "model.mdl"), map_location="cpu")
        )

    # Load YOLOv8 detection model explicitly for inference evaluations
    if not test_on_gt and detector == "yolo":
        print("Initializing nano-scale YOLOv8 network parameters...")
        yolo_model = YOLO("yolov8n.pt")

    for task_number in TASK_NUMBERS:
        if test_on_gt:
            test_db = CocoTasksTestGT(task_number)
        else:
            test_db = CocoTasksTest(task_number, detector_type="yolo")
            
            # Option A integration check: Override the default file-loaded detections dictionary
            if detector == "yolo":
                live_yolo_detections = run_yolo_inference_on_db(test_db, yolo_model)
                test_db.per_image_detections = live_yolo_detections

        print("testing task {}".format(task_number), "---------------------")

        # Process updated bounding boxes through graph experiment model weights
        detections = experiment.do_test(test_db, task_number=task_number)

        detection_file_name = "detections_wa:{}_tn:{}_tgt:{}_f:{}.json".format(
            weighted_aggregation, task_number, test_on_gt, fusion
        )

        with open(os.path.join(folder, detection_file_name), "w") as f:
            json.dump(detections, f)

        with redirect_stdout(open(os.devnull, "w")):
            gtCOCO = test_db.task_coco
            dtCOCO = gtCOCO.loadRes(os.path.join(folder, detection_file_name))
            cocoEval = COCOeval(gtCOCO, dtCOCO, "bbox")
            cocoEval.params.catIds = 1
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

        print("mAP:\t\t %1.6f" % cocoEval.stats[0])
        print("ap@.5:\t\t %1.6f" % cocoEval.stats[1])

        result_file_name = "result_wa:{}_tn:{}_tgt:{}_f:{}.txt".format(
            weighted_aggregation, task_number, test_on_gt, fusion
        )

        with open(os.path.join(folder, result_file_name), "w") as f:
            f.write("%1.6f, %1.6f" % (cocoEval.stats[0], cocoEval.stats[1]))


if __name__ == "__main__":
    main()