"""
Task-Driven Object Detection App
─────────────────────────────────
Pipeline:  Text prompt → TextToTaskEncoder (MiniLM)
                       → YOLOv8n  (object detection)
                       → MobileNetV3 + GGNN  (task-preferred re-ranking)
                       → Draw bounding box of most-preferred object

Place this file alongside ggnn_mobilenet.py / task_detection_full.py,
i.e. one level above the `coco_tasks/` package directory.

Usage:
    python app.py --model /path/to/ggnn-mobilenet-seed:0/model.mdl
"""

import argparse
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageTk
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO




# ── make sure the coco_tasks package on the path ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── COCO helpers ──────────────────────────────────────────────────────────────
COCO_NAMES = {
    1:"person", 2:"bicycle", 3:"car", 4:"motorcycle", 5:"airplane",
    6:"bus", 7:"train", 8:"truck", 9:"boat", 10:"traffic light",
    11:"fire hydrant", 13:"stop sign", 14:"parking meter", 15:"bench",
    16:"bird", 17:"cat", 18:"dog", 19:"horse", 20:"sheep", 21:"cow",
    22:"elephant", 23:"bear", 24:"zebra", 25:"giraffe", 27:"backpack",
    28:"umbrella", 31:"handbag", 32:"tie", 33:"suitcase", 34:"frisbee",
    35:"skis", 36:"snowboard", 37:"sports ball", 38:"kite",
    39:"baseball bat", 40:"baseball glove", 41:"skateboard",
    42:"surfboard", 43:"tennis racket", 44:"bottle", 46:"wine glass",
    47:"cup", 48:"fork", 49:"knife", 50:"spoon", 51:"bowl",
    52:"banana", 53:"apple", 54:"sandwich", 55:"orange", 56:"broccoli",
    57:"carrot", 58:"hot dog", 59:"pizza", 60:"donut", 61:"cake",
    62:"chair", 63:"couch", 64:"potted plant", 65:"bed",
    67:"dining table", 70:"toilet", 72:"tv", 73:"laptop", 74:"mouse",
    75:"remote", 76:"keyboard", 77:"cell phone", 78:"microwave",
    79:"oven", 80:"toaster", 81:"sink", 82:"refrigerator", 84:"book",
    85:"clock", 86:"vase", 87:"scissors", 88:"teddy bear",
    89:"hair drier", 90:"toothbrush",
}

TASK_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

TASK_DESCRIPTIONS = {
    1:  "What object in the scene would a human use to step on?",
    2:  "What object in the scene would a human use to sit comfortably?",
    3:  "What object in the scene would a human use to pour a drink into?",
    4:  "What object in the scene would a human use to carry things in?",
    5:  "What object in the scene would a human use to eat food from?",
    6:  "What object in the scene would a human use to eat food with?",
    7:  "What object in the scene would a human use to ride?",
    8:  "What object in the scene would a human use to hit something?",
    9:  "What object in the scene would a human use to cut something?",
    10: "What object in the scene would a human use to serve a drink?",
    11: "What object in the scene would a human use to wash dishes?",
    12: "What object in the scene would a human use to clean something?",
    13: "What object in the scene would a human use to hold liquid?",
    14: "What object in the scene would a human use to store or hold objects?",
}

# YOLOv8 class-index → official COCO category-id
YOLO_TO_COCO = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

MAX_GPU_SIZE = 64
DETECTION_THRESH = 0.02

# ── image pre-processing (same as training) ───────────────────────────────────
image_transforms = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def crop_img_to_bbox(img: Image.Image, bbox) -> Image.Image:
    x, y, w, h = bbox
    return img.crop((x, y, x + w, y + h))


def make_square_pad(bbox, padding=0.1):
    """MakeBBoxSquare + ScaleAwarePadding — mirrors coco_tasks/transforms.py."""
    x, y, w, h = bbox
    s = max(w, h)
    x -= (s - w) / 2
    y -= (s - h) / 2
    w = h = s
    pw, ph = padding * w, padding * h
    return [x - pw, y - ph, w + 2 * pw, h + 2 * ph]


def get_one_hot(labels, num_classes=90):
    oh = np.zeros((len(labels), num_classes), dtype=np.float32)
    oh[np.arange(len(labels)), labels] = 1
    return oh


# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#1a0a2e"   # deep purple background
PANEL     = "#2d1b4e"   # slightly lighter panel
ACCENT    = "#9b59b6"   # purple accent
ACCENT2   = "#7d3c98"   # darker accent for buttons
TEXT      = "#e8d5f5"   # near-white lavender text
SUBTEXT   = "#b39ddb"   # muted purple text
BTN_TEXT  = "#ffffff"
BORDER    = "#5b2c8d"


# ═════════════════════════════════════════════════════════════════════════════
class TextToTaskEncoder:
    """Lightweight sentence-transformer task matcher (all-MiniLM-L6-v2)."""

    def __init__(self):
        #self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
        #self.model = SentenceTransformer("C:/Users/Lenovo/Downloads/local_minilm", local_files_only=True, device="cpu")
        descriptions = [TASK_DESCRIPTIONS[t] for t in TASK_NUMBERS]
        self.task_embeddings = self.model.encode(
            descriptions, convert_to_tensor=True, device="cpu"
        )

    def encode(self, query: str) -> int:
        q_emb = self.model.encode(query, convert_to_tensor=True, device="cpu")
        sims  = F.cosine_similarity(q_emb.unsqueeze(0), self.task_embeddings)
        best  = int(sims.argmax().item())
        return TASK_NUMBERS[best], TASK_DESCRIPTIONS[TASK_NUMBERS[best]], float(sims[best])


# ═════════════════════════════════════════════════════════════════════════════
class DetectionPipeline:
    """Loads models once; run() accepts a PIL image + task_number."""

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_ggnn(model_path)
        self._load_yolo()

    def _load_ggnn(self, model_path: str):
        from coco_tasks.mobilenet_graph_networks import (
            GGNNDiscLoss, InitializerMul,
            AllLinearAggregatorWeightedWithDetScore, OutputModelFirstLast,
        )
        h_dim, x_dim, c_dim, phi_dim, max_steps = 128, 128, 90, 128, 3

        initializer  = InitializerMul(h_dim=h_dim, phi_dim=phi_dim, c_dim=c_dim)
        aggregator   = AllLinearAggregatorWeightedWithDetScore(in_features=h_dim, out_features=x_dim)
        output_model = OutputModelFirstLast(h_dim=h_dim, num_tasks=len(TASK_NUMBERS))

        self.network = GGNNDiscLoss(
            initializer=initializer,
            aggregator=aggregator,
            output_model=output_model,
            max_steps=max_steps,
            h_dim=h_dim, x_dim=x_dim, class_dim=c_dim,
            fusion="none",
        )

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            self.network.load_state_dict(state)
            print(f"[GGNN] Loaded weights from {model_path}")
        else:
            print("[GGNN] WARNING — no model weights loaded (running with random weights).")

        self.network.eval()
        self.network.to(self.device)

    def _load_yolo(self):
        self.yolo = YOLO("yolov8n.pt")
        print("[YOLO] YOLOv8n ready.")


    # ── main inference ────────────────────────────────────────────────────────
    def run(self, pil_image: Image.Image, task_number: int):
        """
        Returns list of dicts sorted by final task score (highest first).
        Each dict: bbox [x,y,w,h], score, coco_category_id, label, task_score.

        'None' result is impossible — sigmoid always yields a value per detection,
        and argmax always returns the highest-scoring one.  If YOLO finds zero
        objects, we return an empty list (edge case only for blank/featureless images).
        """
        task_id = TASK_NUMBERS.index(task_number)

        # ── Stage 1: YOLOv8 detection ─────────────────────────────────────────
        results = self.yolo.predict(pil_image, verbose=False, device="cpu")[0]
        boxes   = results.boxes

        raw_detections = []
        for box in boxes:
            xyxy  = box.xyxy[0].tolist()
            xmin, ymin, xmax, ymax = xyxy
            w, h  = xmax - xmin, ymax - ymin
            score = float(box.conf[0].item())
            if score < DETECTION_THRESH:
                continue
            yolo_cls = int(box.cls[0].item())
            coco_id  = YOLO_TO_COCO[yolo_cls] if yolo_cls < len(YOLO_TO_COCO) else yolo_cls + 1
            raw_detections.append({
                "bbox": [xmin, ymin, w, h],
                "score": score,
                "category_id": coco_id,
            })

        if not raw_detections:
            return []

        # sort by detection score, cap at MAX_GPU_SIZE
        raw_detections = sorted(raw_detections, key=lambda d: d["score"], reverse=True)
        raw_detections = raw_detections[:MAX_GPU_SIZE]

        # ── Stage 2: crop images for each detection ───────────────────────────
        img_rgb = pil_image.convert("RGB")
        crops = []
        for det in raw_detections:
            bbox_sq = make_square_pad(det["bbox"])
            crops.append(image_transforms(crop_img_to_bbox(img_rgb, bbox_sq)))

        x = torch.stack(crops).to(self.device)                          # [B x 3 x 224 x 224]
        c = torch.tensor(
            get_one_hot([(d["category_id"] - 1) for d in raw_detections], num_classes=90)
        ).to(self.device)                                                # [B x 90]
        d_scores = torch.tensor(
            [[det["score"]] for det in raw_detections], dtype=torch.float32
        ).to(self.device)                                                # [B x 1]

        # ── Stage 3: GGNN re-ranking ──────────────────────────────────────────
        with torch.no_grad():
            probs = self.network.estimate_probability(x, c, d_scores)   # [B x 14]
            task_probs = probs[:, task_id].cpu().numpy()                 # [B]

        # combine YOLO score × GGNN task probability (mirrors do_test in graph_experiments.py)
        for i, det in enumerate(raw_detections):
            det["task_score"] = float(task_probs[i] * det["score"])
            det["label"] = COCO_NAMES.get(det["category_id"], f"cls{det['category_id']}")

        return sorted(raw_detections, key=lambda d: d["task_score"], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):

    CANVAS_W = 560
    CANVAS_H = 420

    def __init__(self, model_path: str):
        super().__init__()
        self.title("Task-Driven Object Detection")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.model_path   = model_path
        self.pipeline     = None          # lazy-loaded on first run
        self.text_encoder = None
        self.pil_image    = None
        self._tk_image    = None          # keep reference

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=16, pady=8)

        # ── title ─────────────────────────────────────────────────────────────
        tk.Label(self, text="Task-Driven Object Detection",
                 font=("Helvetica", 16, "bold"),
                 bg=BG, fg=ACCENT).pack(pady=(18, 4))
        tk.Label(self, text="YOLOv8  ▸  MobileNetV3  ▸  GGNN",
                 font=("Helvetica", 9), bg=BG, fg=SUBTEXT).pack(pady=(0, 10))

        # ── prompt ────────────────────────────────────────────────────────────
        frm_prompt = tk.Frame(self, bg=PANEL, bd=0, highlightthickness=1,
                              highlightbackground=BORDER)
        frm_prompt.pack(fill="x", **pad)

        tk.Label(frm_prompt, text="Task prompt", font=("Helvetica", 9, "bold"),
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=10, pady=(8, 2))

        self.prompt_var = tk.StringVar(value="What would I sit on?")
        tk.Entry(frm_prompt, textvariable=self.prompt_var,
                 font=("Helvetica", 12),
                 bg="#3d2460", fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=6).pack(fill="x", padx=10, pady=(0, 10))

        # ── image upload ──────────────────────────────────────────────────────
        frm_img = tk.Frame(self, bg=PANEL, bd=0, highlightthickness=1,
                           highlightbackground=BORDER)
        frm_img.pack(fill="x", **pad)

        tk.Label(frm_img, text="Input image", font=("Helvetica", 9, "bold"),
                 bg=PANEL, fg=SUBTEXT).pack(anchor="w", padx=10, pady=(8, 4))

        self.img_label_var = tk.StringVar(value="No image selected")
        row = tk.Frame(frm_img, bg=PANEL)
        row.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(row, text="Browse…", command=self._browse_image,
                  bg=ACCENT2, fg=BTN_TEXT, activebackground=ACCENT,
                  relief="flat", padx=10, pady=4,
                  font=("Helvetica", 10, "bold")).pack(side="left")
        tk.Label(row, textvariable=self.img_label_var,
                 bg=PANEL, fg=SUBTEXT, font=("Helvetica", 9),
                 wraplength=380, anchor="w").pack(side="left", padx=10)

        # ── run button ────────────────────────────────────────────────────────
        self.run_btn = tk.Button(self, text="▶  Detect preferred object",
                                 command=self._on_run,
                                 bg=ACCENT, fg=BTN_TEXT, activebackground=ACCENT2,
                                 relief="flat", padx=18, pady=8,
                                 font=("Helvetica", 12, "bold"))
        self.run_btn.pack(pady=6)

        # ── status ────────────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self.status_var,
                 bg=BG, fg=SUBTEXT, font=("Helvetica", 9),
                 wraplength=560).pack()

        # ── result canvas ─────────────────────────────────────────────────────
        self.canvas = tk.Canvas(self, width=self.CANVAS_W, height=self.CANVAS_H,
                                bg="#0d0520", bd=0, highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.pack(padx=16, pady=(8, 4))
        self.canvas.create_text(self.CANVAS_W // 2, self.CANVAS_H // 2,
                                text="Result will appear here",
                                fill=SUBTEXT, font=("Helvetica", 11))

        # ── result label ──────────────────────────────────────────────────────
        self.result_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.result_var,
                 bg=BG, fg=TEXT, font=("Helvetica", 11, "bold"),
                 wraplength=560).pack(pady=(2, 14))

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if path:
            self.pil_image = Image.open(path).convert("RGB")
            name = os.path.basename(path)
            self.img_label_var.set(name)
            # show thumbnail on canvas
            thumb = self._fit_to_canvas(self.pil_image)
            self._tk_image = ImageTk.PhotoImage(thumb)
            self.canvas.delete("all")
            self.canvas.create_image(self.CANVAS_W // 2, self.CANVAS_H // 2,
                                     anchor="center", image=self._tk_image)
            self.result_var.set("")

    def _on_run(self):
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Missing prompt", "Please enter a task prompt.")
            return
        if self.pil_image is None:
            messagebox.showwarning("No image", "Please select an image first.")
            return
        self.run_btn.config(state="disabled")
        self.status_var.set("Loading models (first run may take a moment)…")
        threading.Thread(target=self._run_pipeline, args=(prompt,), daemon=True).start()

    def _run_pipeline(self, prompt: str):
        try:
            # ── lazy-load models ──────────────────────────────────────────────
            if self.text_encoder is None:
                self._set_status("Loading text encoder (MiniLM)…")
                self.text_encoder = TextToTaskEncoder()

            if self.pipeline is None:
                self._set_status("Loading GGNN + MobileNetV3 + YOLOv8…")
                self.pipeline = DetectionPipeline(self.model_path)

            # ── Stage 1: text → task ──────────────────────────────────────────
            self._set_status("Matching prompt to task…")
            task_number, task_desc, sim = self.text_encoder.encode(prompt)
            self._set_status(
                f"Task {task_number}: \"{task_desc}\"  (sim={sim:.2f})  —  running detection…"
            )

            # ── Stages 2-3: detect + re-rank ──────────────────────────────────
            ranked = self.pipeline.run(self.pil_image, task_number)

            # ── draw result ───────────────────────────────────────────────────
            self.after(0, self._show_result, ranked, task_number, task_desc)

        except Exception as e:
            self.after(0, self._set_status, f"Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.after(0, lambda: self.run_btn.config(state="normal"))

    def _show_result(self, ranked, task_number, task_desc):
        if not ranked:
            self._set_status("No objects detected in the image.")
            self.result_var.set("No detectable objects found.")
            return

        best = ranked[0]
        label   = best["label"]
        bbox    = best["bbox"]          # [x, y, w, h] in original image pixels
        t_score = best["task_score"]

        # draw on a copy of the original image
        vis = self.pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(vis)

        x, y, w, h = bbox
        box_coords = [x, y, x + w, y + h]
        box_color  = "#b044ff"

        # thick bounding box (draw 3 concentric rectangles)
        for offset in range(3):
            draw.rectangle(
                [box_coords[0] - offset, box_coords[1] - offset,
                 box_coords[2] + offset, box_coords[3] + offset],
                outline=box_color,
            )

        # label background + text
        tag = f"{label}  ({t_score:.3f})"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        bbox_text = draw.textbbox((x, y - 22), tag, font=font)
        draw.rectangle(bbox_text, fill=box_color)
        draw.text((x, y - 22), tag, fill="white", font=font)

        # show on canvas
        thumb = self._fit_to_canvas(vis)
        self._tk_image = ImageTk.PhotoImage(thumb)
        self.canvas.delete("all")
        self.canvas.create_image(self.CANVAS_W // 2, self.CANVAS_H // 2,
                                 anchor="center", image=self._tk_image)

        self.result_var.set(
            f"Preferred object for Task {task_number}: '{label}'  "
            f"(score {t_score:.4f})"
        )
        self._set_status(
            f"Task {task_number} — \"{task_desc}\""
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    def _fit_to_canvas(self, img: Image.Image) -> Image.Image:
        img.thumbnail((self.CANVAS_W, self.CANVAS_H), Image.LANCZOS)
        return img

    def _set_status(self, msg: str):
        self.after(0, lambda: self.status_var.set(msg))


# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Task-driven object detection GUI")
    p.add_argument(
        "--model",
        default="",
        help=(
            "Path to the trained GGNN model weights (.mdl file). "
            "E.g. --model /path/to/ggnn-mobilenet-seed:0/model.mdl"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    #args = parse_args()
    app = App(model_path="C:/Users/Lenovo/Downloads/MobileNetV3-GGNN-Task-Driven-Object-Detection-Refinement01_1/ggnn-mobilenet-seed0/model.mdl")
    app.mainloop()
