import cv2
import sys
import time
import csv
import os
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from queue import Queue, Empty
from threading import Thread, Lock, Event

import util
from util import (
    CHARS,
    safe_crop,
    match_plate_to_vehicle,
    batch_read_license_plates,
    draw_box,
    draw_quit_hint,
)

# ================= CONFIG =================
MODEL_PATH         = "weights/best_of_both.engine"



# MODEL_PATH_LICENSE = "weigths/license_plate_regonition.onnx"
# MODEL_PATH_LICENSE = "weights/license_plate_regonition.engine"
MODEL_PATH_LICENSE = "weights/license_plate_regonition.pth"

DET_CONF              = 0.30
OCR_EVERY_N_FRAMES    = 1
OCR_CONF_THRESHOLD    = 0.30
SAVE_PATH             = "processed_output.mp4"

DETECT_EVERY_N_FRAMES = 2
OCR_BATCH_SIZE        = 4     
OCR_BATCH_WAIT_MS     = 50

PLATE_CLASS_NAME  = "license_plate"
VEHICLE_CLASSES   = {"car", "motorcycle", "bus", "truck"}

IMG_H, IMG_W = 24, 94          # GeneralizedLPRNet input resolution

os.environ["YOLO_VERBOSE"] = "False"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ================= OCR BACKENDS =================
#
# All three backends expose the same interface:
#   __call__(x: torch.Tensor  (B, 3, 24, 94)  float32  CUDA/CPU)
#         -> torch.Tensor  (B, C, T)  float32   ← format util expects
#
# GeneralizedLPRNet.forward() returns (T, B, C).
# Every backend permutes that to (B, C, T) internally so util is untouched.
# ──────────────────────────────────────────────────────────────────────────────

class _PTHBackend:
    """
    Pure-PyTorch backend. Loads a GeneralizedLPRNet .pth checkpoint.
    Works on CPU too — useful for debugging without a GPU.
    """
    def __init__(self, pth_path: str):
        from model.GeneralizedLPRNet import GeneralizedLPRNet

        self._device = torch.device(DEVICE)
        self._model  = GeneralizedLPRNet(num_classes=len(CHARS) + 1)
        state = torch.load(pth_path, map_location=self._device)
        # tolerate both raw state_dict and {"model": state_dict} checkpoints
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self._model.load_state_dict(state)
        self._model.to(self._device).eval()
        print(f"[OCR] PyTorch backend loaded: {pth_path}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._device).float()
        with torch.no_grad():
            out = self._model(x)          # (T, B, C)
        return out.permute(1, 2, 0)       # → (B, C, T)

    def eval(self):          return self
    def to(self, *a, **kw):  return self
    def parameters(self):    return iter([])


class _ONNXBackend:
    """
    ONNX Runtime backend. Prefers CUDA execution provider, falls back to CPU.
    Input tensor is moved to CPU + numpy before inference (ORT requirement).
    Output is returned as a CUDA torch tensor so the rest of the pipeline is
    unaffected.
    """
    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.log_severity_level = 3          # suppress INFO spam
        self._sess      = ort.InferenceSession(onnx_path, sess_options=opts,
                                               providers=providers)
        self._in_name   = self._sess.get_inputs()[0].name
        self._out_name  = self._sess.get_outputs()[0].name
        self._use_cuda  = torch.cuda.is_available()
        print(f"[OCR] ONNX Runtime backend loaded: {onnx_path}  "
              f"(provider: {self._sess.get_providers()[0]})")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        np_input = x.cpu().numpy().astype(np.float32)
        out_np   = self._sess.run([self._out_name], {self._in_name: np_input})[0]
        # GeneralizedLPRNet ONNX export preserves (T, B, C) axis order
        out = torch.from_numpy(out_np)       # (T, B, C)
        if self._use_cuda:
            out = out.cuda()
        return out.permute(1, 2, 0)          # → (B, C, T)

    def eval(self):          return self
    def to(self, *a, **kw):  return self
    def parameters(self):    return iter([])


# Module-level TRT logger — MUST be a global.
# If it's a local variable, pybind11's factory returns nullptr mid-construction.
_TRT_LOGGER = None

class _TRTBackend:
    """
    TensorRT backend (TRT 10 / 12). Uses execute_async_v3 and torch CUDA
    tensors directly — no pycuda dependency.

    Supports dynamic-batch engines. Input and output buffers are allocated
    fresh per __call__ so any batch size the engine was built with is fine.
    """
    def __init__(self, engine_path: str):
        global _TRT_LOGGER
        import tensorrt as trt

        if _TRT_LOGGER is None:
            _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        runtime       = trt.Runtime(_TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self._engine  = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._in_name  = self._engine.get_tensor_name(0)
        self._out_name = self._engine.get_tensor_name(1)
        print(f"[OCR] TensorRT backend loaded: {engine_path}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda().float()

        # update dynamic batch dimension
        self._context.set_input_shape(self._in_name, tuple(x.shape))

        out_shape = tuple(self._context.get_tensor_shape(self._out_name))
        output    = torch.empty(out_shape, dtype=torch.float32, device="cuda")

        self._context.set_tensor_address(self._in_name,  x.data_ptr())
        self._context.set_tensor_address(self._out_name, output.data_ptr())
        self._context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream
        )
        torch.cuda.synchronize()
        # GeneralizedLPRNet TRT output: (T, B, C)
        return output.permute(1, 2, 0)      # → (B, C, T)

    def eval(self):          return self
    def to(self, *a, **kw):  return self
    def parameters(self):    return iter([])


def load_ocr_model(path: str):
    """
    Auto-selects the backend from the file extension.
      .pth    → _PTHBackend   (GeneralizedLPRNet PyTorch)
      .onnx   → _ONNXBackend  (ONNX Runtime)
      .engine → _TRTBackend   (TensorRT)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pth":
        return _PTHBackend(path)
    elif ext == ".onnx":
        return _ONNXBackend(path)
    elif ext == ".engine":
        return _TRTBackend(path)
    else:
        raise ValueError(
            f"Unsupported OCR model extension '{ext}'. "
            "Expected one of: .pth  .onnx  .engine"
        )

# ================= LOAD MODELS =================
try:
    det_model = YOLO(MODEL_PATH, task="detect")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    sys.exit(1)

try:
    lprnet = load_ocr_model(MODEL_PATH_LICENSE)
except Exception as e:
    print(f"Failed to load OCR model: {e}")
    sys.exit(1)

# ================= HELPERS =================
def detect_all(frame, model, conf_threshold=0.3):
    results = model.predict(
        frame,
        conf=conf_threshold,
        verbose=False,
        device=0 if torch.cuda.is_available() else "cpu",
        imgsz=640
    )

    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id     = int(box.cls[0])
            class_name = str(model.names.get(cls_id, cls_id)).lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append({
                "class_name": class_name,
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
            })
    return detections


def split_detections(detections):
    vehicles, plates = [], []
    for det in detections:
        name = det["class_name"]
        if name in VEHICLE_CLASSES:
            vehicles.append(det)
        elif name == PLATE_CLASS_NAME:
            plates.append(det)
    return vehicles, plates


def assign_simple_ids(detections):
    for i, det in enumerate(detections):
        det["track_id"] = i
    return detections


# ================= TRACKER (unchanged from main_6) =================
class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_lost=10):
        self.next_id       = 0
        self.tracks        = {}
        self.iou_threshold = iou_threshold
        self.max_lost      = max_lost

    def _box_area(self, box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _box_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
        iw  = max(0, ix2 - ix1)
        ih  = max(0, iy2 - iy1)
        inter = iw * ih
        union = self._box_area(a) + self._box_area(b) - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        updated_tracks = {}
        assigned       = set()

        for tid, track in self.tracks.items():
            best_iou, best_idx = 0.0, -1
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                iou = self._box_iou(track["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou > self.iou_threshold:
                updated_tracks[tid] = {"bbox": detections[best_idx]["bbox"], "lost": 0}
                detections[best_idx]["track_id"] = tid
                assigned.add(best_idx)
            else:
                track["lost"] += 1
                if track["lost"] < self.max_lost:
                    updated_tracks[tid] = track

        for i, det in enumerate(detections):
            if i not in assigned:
                tid = self.next_id
                self.next_id += 1
                updated_tracks[tid] = {"bbox": det["bbox"], "lost": 0}
                det["track_id"] = tid

        self.tracks = updated_tracks
        return detections


vehicle_tracker = SimpleTracker()


# ================= OCR BATCH WORKER (unchanged from main_6) =================
class OCRBatchWorker:
    def __init__(
        self,
        lprnet,
        csv_path,
        vehicle_plate_info,
        logged_vehicle_ids,
        pending_vehicle_ids,
        state_lock,
        batch_size=4,
        wait_ms=20,
        conf_threshold=0.5,
    ):
        self.lprnet              = lprnet
        self.csv_path            = csv_path
        self.vehicle_plate_info  = vehicle_plate_info
        self.logged_vehicle_ids  = logged_vehicle_ids
        self.pending_vehicle_ids = pending_vehicle_ids
        self.state_lock          = state_lock
        self.batch_size          = batch_size
        self.wait_ms             = wait_ms
        self.conf_threshold      = conf_threshold
        self.queue               = Queue(maxsize=4000)
        self.stop_event          = Event()
        self.thread              = Thread(target=self._run, daemon=True)
        self.csv_file            = None
        self.writer              = None

    def start(self):
        self.csv_file = open(self.csv_path, mode="w", newline="")
        self.writer   = csv.writer(self.csv_file)
        self.writer.writerow(["Frame", "Vehicle ID", "Class", "Plate", "Confidence"])
        self.thread.start()

    def submit(self, job):
        try:
            self.queue.put_nowait(job)
            return True
        except Exception:
            return False

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()

    def _collect_batch(self):
        batch   = []
        timeout = self.wait_ms / 1000.0
        try:
            batch.append(self.queue.get(timeout=timeout))
        except Empty:
            return batch

        deadline = time.monotonic() + timeout
        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self.queue.get(timeout=remaining))
            except Empty:
                break
        return batch

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            batch = self._collect_batch()
            if not batch:
                continue

            crops = [item["plate_crop"] for item in batch]
            try:
                results = batch_read_license_plates(crops, self.lprnet)
            except Exception:
                results = [(None, 0.0) for _ in batch]

            for job, (plate_text, ocr_conf) in zip(batch, results):
                frame_count = job["frame_count"]
                vid         = job["vehicle_id"]
                class_name  = job["class_name"]

                try:
                    with self.state_lock:
                        self.pending_vehicle_ids.discard(vid)
                        if not plate_text or ocr_conf < self.conf_threshold:
                            continue
                        prev = self.vehicle_plate_info.get(vid)
                        if prev is None or ocr_conf > prev["ocr_conf"]:
                            self.vehicle_plate_info[vid] = {
                                "text":       plate_text,
                                "ocr_conf":   ocr_conf,
                                "class_name": class_name,
                            }
                            if vid not in self.logged_vehicle_ids:
                                self.writer.writerow([
                                    frame_count, vid, class_name,
                                    plate_text, f"{ocr_conf:.2f}"
                                ])
                                self.logged_vehicle_ids.add(vid)
                finally:
                    self.queue.task_done()


# ================= FRAME PROCESSOR (unchanged from main_6) =================
def process_frame(
    frame,
    frame_count,
    vehicle_plate_info,
    state_lock,
    ocr_worker=None,
    last_detections=None,
    track=True,
    detect_every_n_frames=2,
    ocr_every_n_frames=2,
    det_conf_threshold=0.3,
):
    global vehicle_tracker

    if (frame_count - 1) % detect_every_n_frames == 0 or last_detections is None:
        detections = detect_all(frame, det_model, conf_threshold=det_conf_threshold)
    else:
        detections = last_detections

    vehicles, plates = split_detections(detections)

    if track:
        vehicles = vehicle_tracker.update(vehicles)
    else:
        vehicles = assign_simple_ids(vehicles)

    do_ocr = ((frame_count - 1) % ocr_every_n_frames == 0)

    if do_ocr and vehicles and plates and ocr_worker is not None:
        for plate in plates:
            plate_box  = plate["bbox"]
            plate_crop = safe_crop(frame, plate_box)
            if plate_crop is None or plate_crop.size == 0:
                continue

            matched_vehicle = match_plate_to_vehicle(plate_box, vehicles)
            if matched_vehicle is None:
                continue

            vid = matched_vehicle["track_id"]
            with state_lock:
                if vid in vehicle_plate_info:
                    continue
                if vid in ocr_worker.pending_vehicle_ids:
                    continue
                ocr_worker.pending_vehicle_ids.add(vid)

            ocr_worker.submit({
                "frame_count": frame_count,
                "vehicle_id":  vid,
                "class_name":  matched_vehicle["class_name"],
                "plate_crop":  plate_crop.copy(),
            })

    # ── Draw plates ──────────────────────────────────────────────────────────
    for plate in plates:
        x1, y1, x2, y2  = plate["bbox"]
        matched_vehicle  = match_plate_to_vehicle(plate["bbox"], vehicles)
        display_text     = "PLATE"
        if matched_vehicle:
            vid = matched_vehicle["track_id"]
            with state_lock:
                if vid in vehicle_plate_info:
                    display_text = vehicle_plate_info[vid]["text"]
        draw_box(frame, (x1, y1, x2, y2), plate["conf"], display_text, color='b')

    # ── Draw vehicles ─────────────────────────────────────────────────────────
    for v in vehicles:
        vid             = v["track_id"]
        x1, y1, x2, y2 = v["bbox"]
        label           = v["class_name"].upper()
        with state_lock:
            if vid in vehicle_plate_info:
                label += f" | {vehicle_plate_info[vid]['text']}"
        draw_box(frame, (x1, y1, x2, y2), v["conf"], label, color='r')

    return frame, vehicles, plates, detections


# ================= IMAGE MODE =================
def run_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}")
        return

    img        = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    detections = detect_all(img, det_model, conf_threshold=DET_CONF)
    vehicles, plates = split_detections(detections)

    crops, plate_boxes = [], []
    for plate in plates:
        crop = safe_crop(img, plate["bbox"])
        if crop is None or crop.size == 0:
            continue
        crops.append(crop.copy())
        plate_boxes.append(plate["bbox"])

    plate_texts = {}
    if crops:
        results = batch_read_license_plates(crops, lprnet)
        for box, (text, conf) in zip(plate_boxes, results):
            if text and conf >= OCR_CONF_THRESHOLD:
                plate_texts[tuple(box)] = text

    for plate in plates:
        box  = plate["bbox"]
        text = plate_texts.get(tuple(box), "PLATE")
        draw_box(img, box, plate["conf"], text, color='b')

    for v in vehicles:
        draw_box(img, v["bbox"], v["conf"], v["class_name"].upper(), color='r')

    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ================= VIDEO — REAL-TIME =================
def run_video_realtime(path):
    global vehicle_tracker
    vehicle_tracker = SimpleTracker()

    cap              = cv2.VideoCapture(path)
    frame_count      = 0
    vehicle_plate_info  = {}
    logged_vehicle_ids  = set()
    pending_vehicle_ids = set()
    state_lock          = Lock()

    ocr_worker = OCRBatchWorker(
        lprnet=lprnet,
        csv_path="detected_vehicles_and_plates.csv",
        vehicle_plate_info=vehicle_plate_info,
        logged_vehicle_ids=logged_vehicle_ids,
        pending_vehicle_ids=pending_vehicle_ids,
        state_lock=state_lock,
        batch_size=OCR_BATCH_SIZE,
        wait_ms=OCR_BATCH_WAIT_MS,
        conf_threshold=OCR_CONF_THRESHOLD,
    )
    ocr_worker.start()

    try:
        last_detections = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, (1000, 800))

            frame, vehicles, plates, detections = process_frame(
                frame, frame_count, vehicle_plate_info, state_lock,
                ocr_worker=ocr_worker,
                last_detections=last_detections,
                track=True,
                detect_every_n_frames=DETECT_EVERY_N_FRAMES,
                ocr_every_n_frames=OCR_EVERY_N_FRAMES,
                det_conf_threshold=DET_CONF,
            )

            last_detections = detections
            draw_quit_hint(frame, "Quit (Q)")
            cv2.imshow("Video Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ocr_worker.stop()


# ================= VIDEO — OFFLINE =================
def run_video_offline(path):
    global vehicle_tracker
    vehicle_tracker = SimpleTracker()

    cap          = cv2.VideoCapture(path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width, height = 800, 600

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (width, height))

    total_frames        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count         = 0
    vehicle_plate_info  = {}
    logged_vehicle_ids  = set()
    pending_vehicle_ids = set()
    state_lock          = Lock()

    ocr_worker = OCRBatchWorker(
        lprnet=lprnet,
        csv_path="detected_vehicles_and_plates.csv",
        vehicle_plate_info=vehicle_plate_info,
        logged_vehicle_ids=logged_vehicle_ids,
        pending_vehicle_ids=pending_vehicle_ids,
        state_lock=state_lock,
        batch_size=OCR_BATCH_SIZE,
        wait_ms=OCR_BATCH_WAIT_MS,
        conf_threshold=OCR_CONF_THRESHOLD,
    )
    ocr_worker.start()

    try:
        with tqdm(total=total_frames, desc="Processing") as pbar:
            last_detections = None
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frame = cv2.resize(frame, (width, height))

                frame, vehicles, plates, detections = process_frame(
                    frame, frame_count, vehicle_plate_info, state_lock,
                    ocr_worker=ocr_worker,
                    last_detections=last_detections,
                    track=True,
                    detect_every_n_frames=DETECT_EVERY_N_FRAMES,
                    ocr_every_n_frames=OCR_EVERY_N_FRAMES,
                    det_conf_threshold=DET_CONF,
                )

                last_detections = detections
                out.write(frame)
                pbar.update(1)
    finally:
        cap.release()
        out.release()
        ocr_worker.stop()
        print(f"Saved to {SAVE_PATH}")


# ================= UI & ENTRY =================
def pick_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        filetypes=[("Media", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")]
    )
    root.destroy()
    return path


if __name__ == "__main__":
    file_path = pick_file()
    if not file_path:
        print("No file selected.")
        sys.exit(0)

    if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        run_image(file_path)
    else:
        print("\n1. Real-time\n2. Offline (saves to file)")
        mode = input("Choice: ").strip()
        if mode == "1":
            run_video_realtime(file_path)
        elif mode == "2":
            run_video_offline(file_path)
        else:
            print("Invalid choice — exiting.")