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
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics to use the YOLOv8 detector stage: pip install ultralytics")


# --- YOLO confidence threshold (matches app.py defaults) ---
DETECTION_THRESH = 0.02
MAX_YOLO_DETECTIONS = 128
PRE_NMS_TOPK = 1000

# CRITICAL FIX: ASCII COMMENT REPLACES PREVIOUS UTF-8 BOX-DRAWING CHARACTERS
# THAT WERE CORRUPTED TO "ââ" BY ENCODING/MERGE ARTIFACTS.
# Maps YOLOv8 0-79 output indices to official 1-91 COCO category ids.
YOLO_TO_COCO_MAPPING = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


def run_yolo_inference_on_db(test_db, yolo_model, detection_cache=None):
    # CODEX_QUANTIZED_YOLO: raw YOLOv8 ONNX + dynamic INT8 weights + Python NMS.
    print("Executing real-time object detection via quantized YOLOv8 (ONNXRuntime)...")

    _debug_one_image_done = False
    import numpy as _np

    try:
        import onnxruntime as ort  # type: ignore
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
        _HAS_ONNXRUNTIME = True
    except Exception as e:
        _HAS_ONNXRUNTIME = False
        print(f"[DEBUG] ONNXRUNTIME_IMPORT_ERROR={type(e).__name__}: {e}")

    print(f"[DEBUG] ONNXRUNTIME_AVAILABLE={_HAS_ONNXRUNTIME}")

    if not _HAS_ONNXRUNTIME:
        # FALLBACK: original FP32 ultralytics inference
        print("[DEBUG] ONNXRuntime path disabled -> using FP32 YOLOv8.predict()")

        per_image_detections = {}
        all_task_image_ids = test_db.task_coco.getImgIds()
        for img_id in tqdm(all_task_image_ids, desc="YOLOv8 Detection"):
            img_dict = test_db.task_coco.loadImgs(img_id)[0]
            img_path = get_image_file_name(img_dict)
            results = yolo_model.predict(img_path, verbose=False, device='cpu')[0]
            img_detections = []
            boxes = results.boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                xmin, ymin, xmax, ymax = xyxy
                width = xmax - xmin
                height = ymax - ymin
                coco_bbox = [xmin, ymin, width, height]
                score = float(box.conf[0].item())
                yolo_cls = int(box.cls[0].item())
                if yolo_cls < len(YOLO_TO_COCO_MAPPING):
                    coco_category_id = YOLO_TO_COCO_MAPPING[yolo_cls]
                else:
                    coco_category_id = yolo_cls + 1
                img_detections.append({
                    "image_id": int(img_id),
                    "category_id": int(coco_category_id),
                    "bbox": coco_bbox,
                    "score": score
                })
            per_image_detections[img_id] = img_detections
        return per_image_detections

    # Export YOLOv8 without embedded NMS. Quantizing a model that contains
    # NonMaxSuppression can break ONNXRuntime execution, so NMS is done below.
    onnx_dir = SAVING_DIRECTORY if isinstance(SAVING_DIRECTORY, str) and len(SAVING_DIRECTORY) else os.getcwd()
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "yolov8n_raw.onnx")
    quantized_onnx_path = os.path.join(onnx_dir, "yolov8n_raw_int8.onnx")

    if not (os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0):
        _exported = yolo_model.export(
            format="onnx", dynamic=False, imgsz=640, opset=12,
            device="cpu", half=False, nms=False,
        )
        if isinstance(_exported, str) and os.path.exists(_exported) and os.path.getsize(_exported) > 0:
            if os.path.abspath(_exported) != os.path.abspath(onnx_path):
                import shutil
                shutil.copyfile(_exported, onnx_path)

    if not (os.path.exists(quantized_onnx_path) and os.path.getsize(quantized_onnx_path) > 0):
        print(f"[DEBUG] Quantizing YOLOv8 ONNX -> {quantized_onnx_path}")
        quantize_dynamic(
            onnx_path,
            quantized_onnx_path,
            weight_type=QuantType.QInt8,
        )

    runtime_model_path = quantized_onnx_path

    # -------- ONNXRuntime session --------
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(runtime_model_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    # -------- Pre/post-processing helpers --------
    import cv2

    def _letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def _nms_xyxy(boxes, scores, iou_thresh=0.7):
        if len(boxes) == 0:
            return []
        boxes = boxes.astype(_np.float32)
        scores = scores.astype(_np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = _np.maximum(0.0, x2 - x1) * _np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = _np.maximum(x1[i], x1[order[1:]])
            yy1 = _np.maximum(y1[i], y1[order[1:]])
            xx2 = _np.minimum(x2[i], x2[order[1:]])
            yy2 = _np.minimum(y2[i], y2[order[1:]])
            inter = _np.maximum(0.0, xx2 - xx1) * _np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter + 1e-7
            inds = _np.where((inter / union) <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    def _infer_one(img_path: str, img_id: int):
        img = cv2.imread(img_path)
        if img is None:
            return []
        img0 = img
        img, r, (dw, dh) = _letterbox(img, new_shape=(640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(_np.float32) / 255.0
        img = _np.transpose(img, (2, 0, 1))
        img = _np.expand_dims(img, 0)

        out = sess.run(None, {input_name: img})
        pred = _np.asarray(out[0])
        if pred is None or pred.size == 0:
            return []

        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1] and pred.shape[0] in (84, 85):
            pred = pred.T
        if pred.ndim != 2 or pred.shape[1] < 84:
            return []

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:84]
        yolo_classes = class_scores.argmax(axis=1)
        scores = class_scores[_np.arange(class_scores.shape[0]), yolo_classes]
        keep = scores >= DETECTION_THRESH
        if not keep.any():
            return []

        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        yolo_classes = yolo_classes[keep]

        if len(scores) > PRE_NMS_TOPK:
            topk = scores.argsort()[::-1][:PRE_NMS_TOPK]
            boxes_xywh = boxes_xywh[topk]
            scores = scores[topk]
            yolo_classes = yolo_classes[topk]

        x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        boxes_xyxy = _np.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), axis=1)

        detections = []
        for cls in _np.unique(yolo_classes):
            cls_mask = yolo_classes == cls
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = scores[cls_mask]
            for local_i in _nms_xyxy(cls_boxes, cls_scores):
                x1, y1, x2, y2 = cls_boxes[local_i]
                score = float(cls_scores[local_i])

                x1 = (x1 - dw) / r
                x2 = (x2 - dw) / r
                y1 = (y1 - dh) / r
                y2 = (y2 - dh) / r

                xmin = float(max(0.0, min(x1, img0.shape[1] - 1)))
                ymin = float(max(0.0, min(y1, img0.shape[0] - 1)))
                xmax = float(max(0.0, min(x2, img0.shape[1] - 1)))
                ymax = float(max(0.0, min(y2, img0.shape[0] - 1)))

                width = xmax - xmin
                height = ymax - ymin
                if width <= 0 or height <= 0:
                    continue

                yolo_cls = int(cls)
                if yolo_cls < len(YOLO_TO_COCO_MAPPING):
                    coco_category_id = YOLO_TO_COCO_MAPPING[yolo_cls]
                else:
                    coco_category_id = yolo_cls + 1

                detections.append({
                    "bbox": [xmin, ymin, float(width), float(height)],
                    "score": score,
                    "category_id": int(coco_category_id),
                    "image_id": int(img_id),
                })
        detections = sorted(detections, key=lambda d: d["score"], reverse=True)[:MAX_YOLO_DETECTIONS]
        return detections

    per_image_detections = {}
    all_task_image_ids = test_db.task_coco.getImgIds()
    if detection_cache is None:
        detection_cache = {}
    for img_id in tqdm(all_task_image_ids, desc="YOLOv8 Detection"):
        cache_key = int(img_id)
        if cache_key in detection_cache:
            img_detections = detection_cache[cache_key]
        else:
            img_dict = test_db.task_coco.loadImgs(img_id)[0]
            img_path = get_image_file_name(img_dict)
            img_detections = _infer_one(img_path, cache_key)
            detection_cache[cache_key] = img_detections
        per_image_detections[img_id] = img_detections

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
        print("Initializing nano-scale YOLOv8 network parameters...")
        yolo_model = YOLO("yolov8n.pt")
        yolo_detection_cache = {}

    for task_number in TASK_NUMBERS:
        if test_on_gt:
            test_db = CocoTasksTestGT(task_number)
        else:
            test_db = CocoTasksTest(task_number, detector_type="yolo")

            if detector == "yolo":
                live_yolo_detections = run_yolo_inference_on_db(
                    test_db, yolo_model, detection_cache=yolo_detection_cache
                )
                test_db.per_image_detections = live_yolo_detections

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
