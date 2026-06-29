"""
# Modified by Maneth Banula Perera (2026)
# Description: MobileNetV3-based graph network replacing ResNet101 backbone
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

from coco_tasks.graph_datasets import JointCocoTasks, CocoTasksTest, CocoTasksTestGT
from coco_tasks.graph_experiments import JointGraphExperiment
from coco_tasks.mobilenet_graph_networks import (
    GGNNDiscLoss,
    InitializerMul,
    AllLinearAggregator,
    OutputModelFirstLast,
    AllLinearAggregatorWeightedWithDetScore,
)
from coco_tasks.settings import SAVING_DIRECTORY, TASK_NUMBERS
from pycocotools.cocoeval import COCOeval
from coco_tasks.single_task_datasets import get_image_file_name

try:
    import onnxruntime as ort
    import urllib.request, os as _os
    _PICODET_ONNX_URL = "https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_640_lcnet_postprocessed.onnx"
    _PICODET_ONNX_PATH = os.path.join(os.path.expanduser("~"), ".picodet", "picodet_l_640_coco.onnx")
except ImportError:
    raise ImportError("Please install onnxruntime: pip install onnxruntime")


# PicoDet trains on COCO 80 classes in the same order as YOLO.
# 0-indexed class id -> official COCO 1-indexed category_id
DETECTOR_TO_COCO_MAPPING = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

DETECTION_THRESH = 0.02
MAX_DETECTIONS   = 128


def _load_picodet_session():
    """Download PicoDet-L ONNX once and create ONNXRuntime session."""
    _os.makedirs(os.path.dirname(_PICODET_ONNX_PATH), exist_ok=True)
    if not os.path.exists(_PICODET_ONNX_PATH):
        print(f"Downloading PicoDet-L ONNX (~14MB)...")
        urllib.request.urlretrieve(_PICODET_ONNX_URL, _PICODET_ONNX_PATH)
        print(f"Saved to {_PICODET_ONNX_PATH}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # suppress ONNX Runtime warnings (0=verbose, 1=info, 2=warning, 3=error)
    sess = ort.InferenceSession(_PICODET_ONNX_PATH, providers=providers, sess_options=opts)
    return sess


def _picodet_infer_one(sess, img_path, img_id, conf_thresh=DETECTION_THRESH):
    """
    Run PicoDet-L on one image.
    Returns list of dicts: {image_id, category_id, bbox:[x,y,w,h], score}
    """
    import cv2
    import numpy as np

    img = cv2.imread(img_path)
    if img is None:
        return []

    img_h, img_w = img.shape[:2]
    target = 640

    # letterbox resize (same as PaddleDetection LetterBoxResize)
    scale = target / max(img_h, img_w)
    nw, nh = int(img_w * scale), int(img_h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    padded = np.full((target, target, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)

    # ImageNet normalisation (PicoDet expects mean/std after /255)
    padded /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    padded = (padded - mean) / std                         # [H,W,C]
    inp = np.transpose(padded, (2, 0, 1))[None]            # [1,3,640,640]

    # Build input dict dynamically from what the model expects
    input_names = [x.name for x in sess.get_inputs()]
    feed = {}
    for name in input_names:
        name_lower = name.lower()
        if "image" in name_lower:
            feed[name] = inp
        elif "scale" in name_lower:
            feed[name] = np.array([[scale, scale]], dtype=np.float32)
        elif "shape" in name_lower or "im_shape" in name_lower:
            feed[name] = np.array([[img_h, img_w]], dtype=np.float32)

    outputs = sess.run(None, feed)

    # Postprocessed model returns [N,6] or [N,7]; first row: [class_id, score, x1, y1, x2, y2, ...]
    raw = outputs[0]
    if raw is None or len(raw) == 0:
        return []

    detections = []
    for det in raw:
        cls_id, score, x1, y1, x2, y2 = det[:6]
        if float(score) < conf_thresh:
            continue
        cls_id = int(cls_id)
        x1 = max(0.0, float(x1))
        y1 = max(0.0, float(y1))
        x2 = min(float(img_w), float(x2))
        y2 = min(float(img_h), float(y2))
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        coco_cat = DETECTOR_TO_COCO_MAPPING[cls_id] \
            if cls_id < len(DETECTOR_TO_COCO_MAPPING) else cls_id + 1
        detections.append({
            "image_id":    int(img_id),
            "category_id": int(coco_cat),
            "bbox":        [x1, y1, w, h],
            "score":       float(score),
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections[:MAX_DETECTIONS]


def run_detector_inference_on_db(test_db, picodet_sess):
    """Replaces run_yolo_inference_on_db. Same output format."""
    print("Running PicoDet-L detection (CPU)...")
    per_image_detections = {}
    for img_id in tqdm(test_db.task_coco.getImgIds(), desc="PicoDet Detection"):
        img_dict  = test_db.task_coco.loadImgs(img_id)[0]
        img_path  = get_image_file_name(img_dict)
        per_image_detections[img_id] = _picodet_infer_one(
            picodet_sess, img_path, img_id
        )
    return per_image_detections


@click.command()
@click.option("--random-seed", envvar="SEED", default=0)
@click.option("--test-on-gt", type=bool, default=False)
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

    if not test_on_gt and detector == "yolo":
        print(f"Initializing PicoDet-L ONNX session for CPU inference...")
        picodet_sess = _load_picodet_session()

    for task_number in TASK_NUMBERS:
        if test_on_gt:
            test_db = CocoTasksTestGT(task_number)
        else:
            test_db = CocoTasksTest(task_number, detector_type="yolo")

            if detector == "yolo":
                live_detections = run_detector_inference_on_db(
                    test_db, picodet_sess
                )
                test_db.per_image_detections = live_detections

                # Use the method already implemented in the original script to filter valid images
                test_db.list_of_valid_images = []
                for image_id in test_db.task_coco.getImgIds():
                    if len(test_db.per_image_detections[image_id]) > 0:
                        test_db.list_of_valid_images.append(image_id)

        print("testing task {}".format(task_number), "---------------------")

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
