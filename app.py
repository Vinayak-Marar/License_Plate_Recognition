import os
import sys
import csv
import time
import traceback
from queue import Queue, Empty
from threading import Thread, Event, Lock

import cv2
import numpy as np
import torch

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QProgressBar, QListWidget,
    QListWidgetItem, QComboBox, QMessageBox
)

os.environ["YOLO_VERBOSE"] = "False"

# ================= CONFIG =================
DET_MODEL_PATH = "weights/best_of_both.engine"
OCR_MODEL_PATH = "weights/license_plate_recognition.pth"

DET_CONF_DEFAULT = 0.30
OCR_CONF_DEFAULT = 0.99

DETECT_EVERY_N_FRAMES = 2
OCR_EVERY_N_FRAMES = 1
OCR_BATCH_SIZE = 4
OCR_BATCH_WAIT_MS = 50

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
PLATE_CLASS_NAME = "license_plate"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TRT_LOGGER = None

# ================= UTIL =================
from util import (
    CHARS,
    safe_crop,
    match_plate_to_vehicle,
    batch_read_license_plates,
    draw_box,
)

def cv2_to_qpixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def split_detections(detections):
    vehicles, plates = [], []
    for det in detections:
        if det["class_name"] in VEHICLE_CLASSES:
            vehicles.append(det)
        elif det["class_name"] == PLATE_CLASS_NAME:
            plates.append(det)
    return vehicles, plates

def _plate_label(track_id, entry, fallback_class=None):
    plate_class = fallback_class or (entry.get("plate_class") if entry else None) or "UNKNOWN"
    plate_class = str(plate_class).upper()
    if entry and entry.get("text"):
        return f"{entry['text']}"
    return f""

def _plate_class_from_vehicle(plate_bbox, vehicles):
    matched_vehicle = match_plate_to_vehicle(plate_bbox, vehicles)
    if matched_vehicle is None:
        return "UNKNOWN"
    return str(matched_vehicle.get("class_name", "UNKNOWN")).upper()

class SimpleTracker:
    def __init__(self, iou_thr=0.3, max_lost=15):
        self.next_id = 0
        self.tracks = {}
        self.iou_thr = iou_thr
        self.max_lost = max_lost

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def update(self, dets):
        updated, assigned = {}, set()

        for tid, tr in self.tracks.items():
            best, bi = 0.0, -1
            for i, d in enumerate(dets):
                if i in assigned:
                    continue
                v = self._iou(tr["bbox"], d["bbox"])
                if v > best:
                    best, bi = v, i
            if best > self.iou_thr:
                updated[tid] = {"bbox": dets[bi]["bbox"], "lost": 0}
                dets[bi]["track_id"] = tid
                assigned.add(bi)
            else:
                tr["lost"] += 1
                if tr["lost"] < self.max_lost:
                    updated[tid] = tr

        for i, d in enumerate(dets):
            if i not in assigned:
                updated[self.next_id] = {"bbox": d["bbox"], "lost": 0}
                d["track_id"] = self.next_id
                self.next_id += 1

        self.tracks = updated
        return dets

# ================= OCR BACKENDS =================
class _PTHBackend:
    def __init__(self, path):
        from model.GeneralizedLPRNet import GeneralizedLPRNet
        self._dev = torch.device(DEVICE)
        m = GeneralizedLPRNet(num_classes=len(CHARS) + 1)
        state = torch.load(path, map_location=self._dev)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        m.load_state_dict(state)
        self._m = m.to(self._dev).eval()

    def __call__(self, x):
        with torch.no_grad():
            return self._m(x.to(self._dev).float()).permute(1, 2, 0)

    def eval(self): return self
    def to(self, *a, **k): return self
    def parameters(self): return iter([])

class _ONNXBackend:
    def __init__(self, path):
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if torch.cuda.is_available() else ["CPUExecutionProvider"])
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._s = ort.InferenceSession(path, sess_options=opts, providers=providers)
        self._in = self._s.get_inputs()[0].name
        self._out = self._s.get_outputs()[0].name
        self._gpu = torch.cuda.is_available()

    def __call__(self, x):
        out = torch.from_numpy(
            self._s.run([self._out], {self._in: x.cpu().numpy().astype(np.float32)})[0]
        )
        if self._gpu:
            out = out.cuda()
        return out.permute(1, 2, 0)

    def eval(self): return self
    def to(self, *a, **k): return self
    def parameters(self): return iter([])

class _TRTBackend:
    def __init__(self, path):
        global _TRT_LOGGER
        import tensorrt as trt
        if _TRT_LOGGER is None:
            _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        rt = trt.Runtime(_TRT_LOGGER)
        with open(path, "rb") as f:
            self._eng = rt.deserialize_cuda_engine(f.read())
        self._ctx = self._eng.create_execution_context()
        self._in_n = self._eng.get_tensor_name(0)
        self._out_n = self._eng.get_tensor_name(1)

    def __call__(self, x):
        x = x.contiguous().cuda().float()
        self._ctx.set_input_shape(self._in_n, tuple(x.shape))
        out = torch.empty(tuple(self._ctx.get_tensor_shape(self._out_n)),
                          dtype=torch.float32, device="cuda")
        self._ctx.set_tensor_address(self._in_n, x.data_ptr())
        self._ctx.set_tensor_address(self._out_n, out.data_ptr())
        self._ctx.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.permute(1, 2, 0)

    def eval(self): return self
    def to(self, *a, **k): return self
    def parameters(self): return iter([])

def load_ocr_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pth":
        return _PTHBackend(path)
    if ext == ".onnx":
        return _ONNXBackend(path)
    if ext == ".engine":
        return _TRTBackend(path)
    raise ValueError(f"Unsupported OCR model extension '{ext}'")

from ultralytics import YOLO as _YOLO

_det_model = None
_ocr_model = None

def init_models(det_path: str, ocr_path: str):
    global _det_model, _ocr_model
    _det_model = _YOLO(det_path, task="detect")
    _ocr_model = load_ocr_model(ocr_path)

def detect_all(frame, model, conf_threshold=0.30):
    results = model.predict(
        frame,
        conf=conf_threshold,
        verbose=False,
        device=0 if torch.cuda.is_available() else "cpu",
        imgsz=640,
    )
    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = str(model.names.get(cls_id, cls_id)).lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_name": class_name,
                "bbox": (x1, y1, x2, y2),
                "conf": float(box.conf[0]),
            })
    return detections

# ================= PLATE OCR WORKER =================
class OCRBatchLogger:
    """
    Batch OCR worker that stores the best OCR text per plate track ID.
    Also carries the inferred vehicle class for the plate.
    """
    def __init__(self, lpr_model, csv_path, plate_memory, pending_ids, state_lock,
                 conf_threshold=0.30, batch_size=4, wait_ms=50, on_plate=None):
        self.lpr_model = lpr_model
        self.csv_path = csv_path
        self.plate_memory = plate_memory
        self.pending_ids = pending_ids
        self.state_lock = state_lock
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.wait_ms = wait_ms
        self.on_plate = on_plate

        self.queue = Queue(maxsize=4000)
        self.stop_event = Event()
        self.thread = Thread(target=self._run, daemon=True)
        self.file = None
        self.writer = None

    def start(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self.file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Frame", "Plate ID", "Plate", "Confidence", "Class", "Status"])
        self.thread.start()

    def submit(self, frame_count, plate_id, plate_class, plate_crop):
        try:
            self.queue.put_nowait((frame_count, plate_id, plate_class, plate_crop))
            return True
        except Exception:
            return False

    def _collect_batch(self):
        batch = []
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

            frame_counts = [x[0] for x in batch]
            plate_ids = [x[1] for x in batch]
            plate_classes = [x[2] for x in batch]
            crops = [x[3] for x in batch]

            try:
                results = batch_read_license_plates(crops, self.lpr_model)
            except Exception:
                results = [(None, 0.0) for _ in crops]

            for fc, pid, pclass, (text, conf) in zip(frame_counts, plate_ids, plate_classes, results):
                pclass = (pclass or "UNKNOWN").upper()
                with self.state_lock:
                    self.pending_ids.discard(pid)
                    if not text or conf < self.conf_threshold:
                        continue

                    prev = self.plate_memory.get(pid)
                    is_better = prev is None or conf > prev.get("conf", 0.0)

                try:
                    if is_better:
                        status = "new" if prev is None else "update"
                        self.plate_memory[pid] = {
                            "text": text,
                            "conf": float(conf),
                            "frame": fc,
                            "plate_class": pclass,
                        }
                        self.writer.writerow([fc, pid, text, f"{conf:.3f}", pclass, status])
                        self.file.flush()

                        if self.on_plate is not None:
                            try:
                                self.on_plate(pid, text, float(conf), pclass)
                            except Exception:
                                pass
                finally:
                    self.queue.task_done()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3.0)
        if self.file:
            self.file.flush()
            self.file.close()

# ================= FRAME PROCESSING =================
def process_frame(frame, plate_tracker, vehicle_tracker, plate_memory, pending_ids, state_lock,
                  frame_count, conf_thr, ocr_thr, ocr_logger=None,
                  detect_every_n_frames=2, ocr_every_n_frames=1):
    """
    Plate-centric processing:
    - vehicle tracking remains only for optional vehicle boxes
    - each plate gets its own track ID
    - OCR best result is stored by plate track ID
    - inferred vehicle class is attached to each plate
    """
    if (frame_count - 1) % detect_every_n_frames == 0 or getattr(process_frame, "_last_dets", None) is None:
        dets = detect_all(frame, _det_model, conf_threshold=conf_thr)
        process_frame._last_dets = dets
    else:
        dets = process_frame._last_dets

    vehicles, plates = split_detections(dets)
    vehicles = vehicle_tracker.update(vehicles)
    plates = plate_tracker.update(plates)

    plate_classes = {}
    for pl in plates:
        pid = pl.get("track_id")
        if pid is not None:
            plate_classes[pid] = _plate_class_from_vehicle(pl["bbox"], vehicles)

    # queue OCR for plates
    if (frame_count - 1) % ocr_every_n_frames == 0 and ocr_logger is not None:
        for pl in plates:
            pid = pl.get("track_id")
            if pid is None:
                continue
            crop = safe_crop(frame, pl["bbox"])
            if crop is None or crop.size == 0:
                continue
            with state_lock:
                if pid in pending_ids:
                    continue
                pending_ids.add(pid)
            ocr_logger.submit(frame_count, pid, plate_classes.get(pid, "UNKNOWN"), crop.copy())

    # draw plate boxes with plate IDs and best known text
    for pl in plates:
        x1, y1, x2, y2 = pl["bbox"]
        pid = pl.get("track_id")
        with state_lock:
            entry = plate_memory.get(pid)

        display_text = _plate_label(pid if pid is not None else -1, entry, plate_classes.get(pid, "UNKNOWN"))
  
        draw_box(frame, (x1, y1, x2, y2), pl["conf"], display_text, color='b')

    # draw vehicle boxes 
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        label = v["class_name"].upper()
        draw_box(frame, (x1, y1, x2, y2), v["conf"], label, color='r')

    return frame, vehicles, plates, dets

# ================= UI HELPERS =================
class VideoLabel(QLabel):
    def __init__(self, placeholder="AWAITING INPUT...", parent=None):
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(700, 500)
        self.setStyleSheet("background:#0b0f14; color:#5a7a96; border:1px solid #1c2a3a;")
        self._pixmap = None

    def set_frame(self, frame: np.ndarray):
        self._pixmap = cv2_to_qpixmap(frame)
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap:
            self._update_display()

    def _update_display(self):
        scaled = self._pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def clear_display(self):
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText("AWAITING INPUT...")

class ControlSlider(QWidget):
    valueChanged = pyqtSignal(float)
    def __init__(self, label, min_v, max_v, default, scale=1.0, fmt="{:.0f}", parent=None):
        super().__init__(parent)
        self.scale = scale
        self.fmt = fmt
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setMinimumWidth(150)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_v / scale), int(max_v / scale))
        self.slider.setValue(int(default / scale))
        self.val_lbl = QLabel(self.fmt.format(default))
        self.val_lbl.setMinimumWidth(55)
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row.addWidget(lbl)
        row.addWidget(self.slider, 1)
        row.addWidget(self.val_lbl)

        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, v):
        real = v * self.scale
        self.val_lbl.setText(self.fmt.format(real))
        self.valueChanged.emit(real)

    def value(self):
        return self.slider.value() * self.scale

# ================= IMAGE TAB =================
class ImageTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig = None
        self._ann = None
        self._plate_items = {}
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.btn_open = QPushButton("Open Image")
        self.btn_run = QPushButton("Run Detection")
        self.btn_save = QPushButton("Save Result")
        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.sl_conf = ControlSlider("Detection Confidence", 0.10, 1.00, DET_CONF_DEFAULT, scale=0.01, fmt="{:.2f}")
        self.sl_ocr = ControlSlider("OCR Threshold", 0.10, 1.00, OCR_CONF_DEFAULT, scale=0.01, fmt="{:.2f}")
        self.sl_scale = ControlSlider("Display Scale", 10, 200, 100, fmt="{:.0f}%")

        self.status = QLabel("OFFLINE")
        self.status.setStyleSheet("color:#5a7a96;")

        self.plates = QListWidget()

        left.addWidget(self.btn_open)
        left.addWidget(self.sl_conf)
        left.addWidget(self.sl_ocr)
        left.addWidget(self.sl_scale)
        left.addWidget(self.btn_run)
        left.addWidget(self.btn_save)
        left.addWidget(self.status)
        left.addWidget(QLabel("Detected Plates"))
        left.addWidget(self.plates, 1)

        self.preview = VideoLabel("Drop or open an image")
        root.addLayout(left, 0)
        root.addWidget(self.preview, 1)

        self.btn_open.clicked.connect(self.open_image)
        self.btn_run.clicked.connect(self.run_detection)
        self.btn_save.clicked.connect(self.save_result)
        self.sl_scale.valueChanged.connect(self.apply_scale)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.jpeg *.png)")
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            return
        self._orig = frame
        self._ann = None
        self._src_name = os.path.splitext(os.path.basename(path))[0]
        self.preview.set_frame(frame)
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.plates.clear()
        self._plate_items.clear()
        self.status.setText(f"Loaded: {os.path.basename(path)}")

    def _upsert_plate_item(self, pid, text, conf, plate_class):
        plate_class = (plate_class or "UNKNOWN").upper()
        label = f"ID {pid} | {plate_class} | {text} ({conf:.3f})"
        item = self._plate_items.get(pid)
        if item is None:
            item = QListWidgetItem(label)
            self._plate_items[pid] = item
            self.plates.addItem(item)
        else:
            item.setText(label)

    def run_detection(self):
        if self._orig is None:
            return

        vehicle_tracker = SimpleTracker()
        process_frame._last_dets = None
        plate_tracker = SimpleTracker()
        plate_memory = {}
        pending_ids = set()
        state_lock = Lock()

        ann, vehicles, plates, dets = process_frame(
            self._orig.copy(),
            plate_tracker,
            vehicle_tracker,
            plate_memory,
            pending_ids,
            state_lock,
            1,
            self.sl_conf.value(),
            self.sl_ocr.value(),
            ocr_logger=None,
            detect_every_n_frames=1,
            ocr_every_n_frames=1
        )

        # OCR directly for image mode
        plate_boxes = []
        plate_ids = []
        crops = []
        for pl in plates:
            crop = safe_crop(self._orig, pl["bbox"])
            if crop is None or crop.size == 0:
                continue
            pid = pl.get("track_id")
            if pid is None:
                continue
            plate_boxes.append(pl["bbox"])
            plate_ids.append(pid)
            crops.append(crop)

        self.plates.clear()
        self._plate_items.clear()

        if crops:
            results = batch_read_license_plates(crops, _ocr_model)
            for pid, box, (text, conf) in zip(plate_ids, plate_boxes, results):
                if text and conf >= self.sl_ocr.value():

                    plate_class = _plate_class_from_vehicle(box, vehicles)

                    with state_lock:
                        prev = plate_memory.get(pid)
                        if prev is None or conf > prev["conf"]:
                            plate_memory[pid] = {
                                "text": text,
                                "conf": float(conf),
                                "frame": 1,
                                "plate_class": plate_class
                            }

                    draw_box(ann, box, conf, _plate_label(pid, plate_memory.get(pid)), color='b')

                    self._upsert_plate_item(
                        pid,
                        text,
                        conf,
                        plate_class
                    )

        self._ann = ann
        self.apply_scale()
        self.btn_save.setEnabled(True)
        self.status.setText("DONE")

    def apply_scale(self, *_):
        frame = self._ann if self._ann is not None else self._orig
        if frame is None:
            return
        pct = self.sl_scale.value() / 100.0
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (max(1, int(w * pct)), max(1, int(h * pct))))
        self.preview.set_frame(resized)

    def save_result(self):
        if self._ann is None:
            return
        # ── CHANGED: default save location is the save/ folder ──
        project_root = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(project_root, "save")
        os.makedirs(save_dir, exist_ok=True)
        src_name = getattr(self, "_src_name", "result")
        default_path = os.path.join(save_dir, src_name + "_result.jpg")
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", default_path, "Images (*.jpg *.png)")
        if path:
            cv2.imwrite(path, self._ann)

# ================= OFFLINE VIDEO TAB =================
class OfflineTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.src = None
        self.out_csv = None
        self.worker = None
        self._plate_items = {}
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.btn_open = QPushButton("Open Video")
        self.btn_start = QPushButton("Start Processing")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.src_lbl = QLabel("No file selected")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        self.status = QLabel("OFFLINE")
        self.plates = QListWidget()

        self.sl_conf = ControlSlider("Detection Confidence", 0.10, 1.00, DET_CONF_DEFAULT, scale=0.01, fmt="{:.2f}")
        self.sl_ocr = ControlSlider("OCR Threshold", 0.10, 1.00, OCR_CONF_DEFAULT, scale=0.01, fmt="{:.2f}")

        left.addWidget(self.btn_open)
        left.addWidget(self.src_lbl)
        left.addWidget(self.sl_conf)
        left.addWidget(self.sl_ocr)
        left.addWidget(self.btn_start)
        left.addWidget(self.btn_stop)
        left.addWidget(self.progress)
        left.addWidget(self.status)
        left.addWidget(QLabel("Logged Plates"))
        left.addWidget(self.plates, 1)

        self.preview = VideoLabel("Offline mode does not render video")
        root.addLayout(left, 0)
        root.addWidget(self.preview, 1)

        self.btn_open.clicked.connect(self.open_video)
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_stop.clicked.connect(self.stop_processing)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.src = path
        self.src_lbl.setText(os.path.basename(path))
        self.status.setText("Ready")

    def _upsert_plate_item(self, pid, text, conf, plate_class):
        plate_class = (plate_class or "UNKNOWN").upper()
        label = f"ID {pid} | {plate_class} | {text} ({conf:.3f})"
        item = self._plate_items.get(pid)
        if item is None:
            item = QListWidgetItem(label)
            self._plate_items[pid] = item
            self.plates.addItem(item)
        else:
            item.setText(label)

    def start_processing(self):
        if not self.src:
            return

        project_root = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(project_root, "save")
        os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.src))[0]
        self.out_csv = os.path.join(save_dir, base_name + "_plates.csv")
        self.out_video = os.path.join(save_dir, base_name + "_processed.mp4")

        self.plates.clear()
        self._plate_items.clear()
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("Processing...")

        self.worker = OfflineVideoWorker(
            self.src, self.out_csv, self.out_video,
            self.sl_conf.value(), self.sl_ocr.value(),
            on_plate=self._on_plate
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def stop_processing(self):
        if self.worker:
            self.worker.stop()

    def _on_plate(self, pid, text, conf, plate_class):
        self._upsert_plate_item(pid, text, conf, plate_class)

    def _on_done(self, message):
        self.status.setText(message)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_error(self, msg):
        self.status.setText("Error")
        QMessageBox.critical(self, "Offline Processing Error", msg)

class OfflineVideoWorker(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(str)
    error = pyqtSignal(str)
    plate_found = pyqtSignal(int, str, float, str)

    def __init__(self, src, csv_path, video_path, conf_thr, ocr_thr, on_plate=None):
        super().__init__()
        self.src = src
        self.csv_path = csv_path
        self.video_path = video_path
        self.conf_thr = conf_thr
        self.ocr_thr = ocr_thr
        self.on_plate = on_plate
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.src)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0

            vehicle_tracker = SimpleTracker()
            process_frame._last_dets = None
            plate_tracker = SimpleTracker()
            plate_memory = {}
            pending_ids = set()
            state_lock = Lock()
            frame_count = 0

            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(self.video_path) or ".", exist_ok=True)

            writer = cv2.VideoWriter(
                self.video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (1280, 960),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {self.video_path}")

            logger = OCRBatchLogger(
                _ocr_model,
                self.csv_path,
                plate_memory=plate_memory,
                pending_ids=pending_ids,
                state_lock=state_lock,
                conf_threshold=self.ocr_thr,
                batch_size=OCR_BATCH_SIZE,
                wait_ms=OCR_BATCH_WAIT_MS,
                on_plate=lambda pid, text, conf, plate_class: self.plate_found.emit(pid, text, conf, plate_class)
            )
            if self.on_plate is not None:
                self.plate_found.connect(self.on_plate)
            logger.start()

            try:
                while cap.isOpened() and not self._stop:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    frame = cv2.resize(frame, (1280, 960))

                    ann, vehicles, plates, dets = process_frame(
                        frame, plate_tracker, vehicle_tracker, plate_memory, pending_ids, state_lock,
                        frame_count, self.conf_thr, self.ocr_thr, ocr_logger=logger,
                        detect_every_n_frames=DETECT_EVERY_N_FRAMES,
                        ocr_every_n_frames=OCR_EVERY_N_FRAMES,
                    )
                    writer.write(ann)
                    self.progress.emit(min(99, int(frame_count / total * 100)))
                self.progress.emit(100)
            finally:
                cap.release()
                writer.release()
                logger.stop()

            self.done.emit(f"Saved video: {self.video_path} | CSV: {self.csv_path}")
        except Exception:
            self.error.emit(traceback.format_exc())

# ================= REALTIME TAB =================
class RealtimeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.src = None
        self.worker = None
        self._plate_items = {}
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["Webcam 0", "Webcam 1", "Video File"])
        self.btn_open = QPushButton("Open Source")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.src_lbl = QLabel("No source")
        self.fps_lbl = QLabel("0 FPS")
        self.plates = QListWidget()
        self.status = QLabel("OFFLINE")

        self.sl_conf = ControlSlider("Detection Confidence", 0.10, 1.00, DET_CONF_DEFAULT, scale=0.01, fmt="{:.2f}")
        self.sl_ocr = ControlSlider("OCR Threshold", 0.10, 1.00, OCR_CONF_DEFAULT, scale=0.01, fmt="{:.2f}")

        left.addWidget(self.mode)
        left.addWidget(self.btn_open)
        left.addWidget(self.src_lbl)
        left.addWidget(self.sl_conf)
        left.addWidget(self.sl_ocr)
        left.addWidget(self.btn_start)
        left.addWidget(self.btn_stop)
        left.addWidget(self.fps_lbl)
        left.addWidget(self.status)
        left.addWidget(QLabel("Logged Plates"))
        left.addWidget(self.plates, 1)

        self.preview = VideoLabel("Live feed")
        root.addLayout(left, 0)
        root.addWidget(self.preview, 1)

        self.btn_open.clicked.connect(self.open_source)
        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)

    def open_source(self):
        if self.mode.currentText() == "Video File":
            path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)")
            if not path:
                return
            self.src = path
            self.src_lbl.setText(os.path.basename(path))
        else:
            self.src = 0 if self.mode.currentText() == "Webcam 0" else 1
            self.src_lbl.setText(self.mode.currentText())

    def _upsert_plate_item(self, pid, text, conf, plate_class):
        plate_class = (plate_class or "UNKNOWN").upper()
        label = f"ID {pid} | {plate_class} | {text} ({conf:.3f})"
        item = self._plate_items.get(pid)
        if item is None:
            item = QListWidgetItem(label)
            self._plate_items[pid] = item
            self.plates.addItem(item)
        else:
            item.setText(label)

    def start_stream(self):
        if self.src is None:
            self.open_source()
        if self.src is None:
            return

        # ── CHANGED: save CSV to the save/ folder ──
        project_root = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(project_root, "save")
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(self.src, int):
            base = f"webcam{self.src}"
        else:
            base = os.path.splitext(os.path.basename(self.src))[0]
        csv_path = os.path.join(save_dir, base + "_realtime_plates.csv")

        self.plates.clear()
        self._plate_items.clear()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("Streaming...")

        self.worker = RealtimeVideoWorker(
            self.src,
            csv_path,
            self.sl_conf.value(),
            self.sl_ocr.value(),
        )
        self.worker.frame_ready.connect(self.preview.set_frame)
        self.worker.plate_found.connect(self._on_plate)
        self.worker.fps_update.connect(self._on_fps)
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def stop_stream(self):
        if self.worker:
            self.worker.stop()

    def _on_plate(self, pid, text, conf, plate_class):
        self._upsert_plate_item(pid, text, conf, plate_class)

    def _on_fps(self, fps):
        self.fps_lbl.setText(f"{fps:.1f} FPS")

    def _on_done(self):
        self.status.setText("Stopped")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_error(self, msg):
        self.status.setText("Error")
        QMessageBox.critical(self, "Realtime Error", msg)

class RealtimeVideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    plate_found = pyqtSignal(int, str, float, str)
    fps_update = pyqtSignal(float)
    done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, src, csv_path, conf_thr, ocr_thr):
        super().__init__()
        self.src = src
        self.csv_path = csv_path
        self.conf_thr = conf_thr
        self.ocr_thr = ocr_thr
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.src)
            vehicle_tracker = SimpleTracker()
            process_frame._last_dets = None
            plate_tracker = SimpleTracker()
            plate_memory = {}
            pending_ids = set()
            state_lock = Lock()
            frame_count = 0

            logger = OCRBatchLogger(
                _ocr_model,
                self.csv_path,
                plate_memory=plate_memory,
                pending_ids=pending_ids,
                state_lock=state_lock,
                conf_threshold=self.ocr_thr,
                batch_size=OCR_BATCH_SIZE,
                wait_ms=OCR_BATCH_WAIT_MS,
                on_plate=lambda pid, text, conf, plate_class: self.plate_found.emit(pid, text, conf, plate_class)
            )
            logger.start()

            fps_t = time.monotonic()
            fps_cnt = 0

            try:
                while cap.isOpened() and not self._stop:
                    t0 = time.monotonic()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    frame = cv2.resize(frame, (1280, 960))

                    ann, vehicles, plates, dets = process_frame(
                        frame, plate_tracker, vehicle_tracker, plate_memory, pending_ids, state_lock,
                        frame_count, self.conf_thr, self.ocr_thr, ocr_logger=logger,
                        detect_every_n_frames=DETECT_EVERY_N_FRAMES,
                        ocr_every_n_frames=OCR_EVERY_N_FRAMES,
                    )
                    self.frame_ready.emit(ann.copy())

                    fps_cnt += 1
                    elapsed = time.monotonic() - fps_t
                    if elapsed >= 1.0:
                        self.fps_update.emit(fps_cnt / elapsed)
                        fps_cnt = 0
                        fps_t = time.monotonic()

                    sleep = 0.001 - (time.monotonic() - t0)
                    if sleep > 0:
                        self.msleep(int(sleep * 1000))

            finally:
                cap.release()
                logger.stop()

            self.done.emit()
        except Exception:
            self.error.emit(traceback.format_exc())

# ================= MAIN WINDOW =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LPR Integrated UI v4")
        self.resize(1500, 900)

        try:
            init_models(DET_MODEL_PATH, OCR_MODEL_PATH)
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))
            raise

        tabs = QTabWidget()
        tabs.addTab(ImageTab(), "Image")
        tabs.addTab(OfflineTab(), "Offline Video")
        tabs.addTab(RealtimeTab(), "Realtime / Webcam")
        self.setCentralWidget(tabs)

        self.statusBar().showMessage(f"Using device: {DEVICE}")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()