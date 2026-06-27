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

# CRITICAL FIX: ASCII COMMENT REPLACES PREVIOUS UTF-8 BOX-DRAWING CHARACTERS
# THAT WERE CORRUPTED TO "ââ" BY ENCODING/MERGE ARTIFACTS.
# Maps YOLOv8 0-79 output indices to official 1-91 COCO category ids.
YOLO_TO_COCO_MAPPING = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


def run_yolo_inference_on_db(test_db, yolo_model):
    # CRITICAL FIX: HEADER MESSAGE NO LONGER CLAIMS QUANTIZATION.
    # WE USE A NMS-EMBEDDED FP32 ONNX MODEL VIA ONNXRUNTIME.
    # (Dynamic INT8 quantization is intentionally skipped: it corrupts the
    # post-NMS graph in modern onnxruntime and is not the cause of mAP=0.)
    print("Executing real-time object detection via YOLOv8 (ONNXRuntime)...")

    _debug_one_image_done = False
    import tempfile
    import numpy as _np

    try:
        import onnxruntime as ort  # type: ignore
        # CRITICAL FIX: QuantType and quantize_dynamic are no longer used.
        # Removed their import to avoid an unused-import drift and confusion
        # with the (now-skipped) quantization step.
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

    # CRITICAL FIX: EXPORT YOLOV8N TO ONNX ONCE PER PROCESS (NMS-EMBEDDED).
    # We re-use the file if it already exists to avoid re-exporting per image.
    onnx_dir = SAVING_DIRECTORY if isinstance(SAVING_DIRECTORY, str) and len(SAVING_DIRECTORY) else os.getcwd()
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "yolov8n.onnx")

    # CRITICAL FIX: EXPORT ONLY WHEN THE FILE IS MISSING OR EMPTY.
    # We do NOT re-export on every call to run_yolo_inference_on_db.
    if not (os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0):
        # CRITICAL FIX: nms=True so the ONNX output is post-NMS, shape
        # (1, max_dets, 6) = [x1, y1, x2, y2, score, class_id] in letterbox
        # space. This is the format _infer_one expects.
        _exported = yolo_model.export(
            format="onnx", dynamic=False, imgsz=640, opset=12,
            device="cpu", half=False, nms=True,
        )
        # CRITICAL FIX: USE THE PATH RETURNED BY ultralytics (it may differ
        # from onnx_path if CWD != onnx_dir).
        if isinstance(_exported, str) and os.path.exists(_exported) and os.path.getsize(_exported) > 0:
            onnx_path = _exported

    # CRITICAL FIX: FALLBACK TO A yolov8n.onnx IN CWD IF onnx_path IS WRONG.
    if (not os.path.exists(onnx_path)) or os.path.getsize(onnx_path) == 0:
        _fallback = "yolov8n.onnx"
        if os.path.exists(_fallback) and os.path.getsize(_fallback) > 0:
            onnx_path = _fallback

    # CRITICAL FIX: USE THE EXPORTED ONNX MODEL DIRECTLY. NO quantize_dynamic.
    # Dynamic INT8 quantization corrupts the post-NMS graph in modern
    # onnxruntime, so we intentionally do not call it. The model is FP32
    # ONNX with embedded NMS, which is what the rest of this file expects.
    runtime_model_path = onnx_path

    # -------- ONNXRuntime session --------
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(runtime_model_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    # -------- Pre/post-processing helpers --------
    # CRITICAL FIX: STRAIGHT-FORWARD LETTERBOX + 0..1 FLOAT32.
    # ultralytics' nms=True ONNX export embeds BGR->RGB and 0-1 normalization
    # in the graph, so we feed it a (1, 3, 640, 640) BGR float32 tensor
    # that is already in [0, 1]. We do NOT do a manual BGR->RGB swap here.
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

    def _infer_one(img_path: str, img_id: int):
        # CRITICAL FIX: INPUT PIPELINE FOR THE NMS-EMBEDDED ONNX MODEL.
        # (1) Read the image as BGR uint8 (OpenCV default).
        # (2) Letterbox to 640x640 (BGR uint8).
        # (3) Cast to float32 and scale to [0, 1].
        # (4) HWC -> CHW -> NCHW.
        # The exported ONNX graph does the BGR->RGB swap and final normalize,
        # so we DO NOT call cv2.cvtColor(..., BGR2RGB) here.
        img = cv2.imread(img_path)
        if img is None:
            return []
        img0 = img
        img, r, (dw, dh) = _letterbox(img, new_shape=(640, 640))
        img = img.astype(_np.float32) / 255.0
        img = _np.transpose(img, (2, 0, 1))
        img = _np.expand_dims(img, 0)

        out = sess.run(None, {input_name: img})
        # CRITICAL FIX: HANDLE THE 3D (1, max_dets, 6) POST-NMS OUTPUT.
        # ultralytics pads unused detection rows with all zeros; we drop them
        # by filtering on the score column (index 4) before any further
        # processing. This avoids feeding `(0,0,0,0,0,0)` rows into the loop.
        det = _np.asarray(out[0])
        if det is None or det.size == 0:
            return []
        if det.ndim == 3:
            det = det[0]
        if det.ndim == 1:
            det = _np.expand_dims(det, 0)
        if det.shape[1] >= 6:
            non_pad = det[:, 4] > 0
            if non_pad.any():
                det = det[non_pad]
            else:
                return []

        detections = []
        # CRITICAL FIX: ROW LAYOUT IS [x1, y1, x2, y2, score, class_id] IN
        # LETTERBOX SPACE. class_id IS THE COCO CATEGORY ID (1..90), NOT A
        # YOLO INDEX, SO WE DO NOT LOOK IT UP IN YOLO_TO_COCO_MAPPING.
        for row in det:
            if row.shape[0] < 6:
                continue
            x1, y1, x2, y2, score, cls = row[:6]
            score = float(score)
            if score < DETECTION_THRESH:
                continue
            cls = int(cls)

            # CRITICAL FIX: MAP BOXES FROM LETTERBOX SPACE BACK TO ORIGINAL
            # IMAGE SPACE. dw, dh are HALF padding on each side; r is the
            # resize ratio used by _letterbox.
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

            # CRITICAL FIX: DROP ZERO-SIZE BOXES (CAN OCCUR AT IMAGE EDGES).
            if width <= 0 or height <= 0:
                continue

            # CRITICAL FIX: cls IS THE COCO CATEGORY ID (1..90). KEEP IT
            # DIRECTLY; DO NOT REMAP VIA YOLO_TO_COCO_MAPPING. graph_datasets
            # assumes category_id-1 is in [0..89], so we filter to [1..90].
            if cls < 1 or cls > 90:
                continue
            coco_category_id = cls

            detections.append({
                "bbox": [xmin, ymin, float(width), float(height)],
                "score": score,
                "category_id": int(coco_category_id),
                "image_id": int(img_id),
            })
        return detections

    per_image_detections = {}
    all_task_image_ids = test_db.task_coco.getImgIds()
    for img_id in tqdm(all_task_image_ids, desc="YOLOv8 Detection"):
        img_dict = test_db.task_coco.loadImgs(img_id)[0]
        img_path = get_image_file_name(img_dict)
        img_detections = _infer_one(img_path, int(img_id))
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

    for task_number in TASK_NUMBERS:
        if test_on_gt:
            test_db = CocoTasksTestGT(task_number)
        else:
            test_db = CocoTasksTest(task_number, detector_type="yolo")

            if detector == "yolo":
                live_yolo_detections = run_yolo_inference_on_db(test_db, yolo_model)
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
