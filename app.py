"""
app_v9.py  —  LPR Desktop Application
Requires:  PyQt5, opencv-python, torch, ultralytics, numpy
Optional:  tensorrt (.engine), onnxruntime (.onnx)

Run:  python app_v9.py
"""

import sys
import os
import csv
import time
import traceback

import cv2
import numpy as np
import torch
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, QRect
)
from PyQt5.QtGui import (
    QPixmap, QImage, QFont, QFontDatabase, QColor, QPalette,
    QIcon, QPainter, QPen, QBrush, QLinearGradient
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QFileDialog,
    QTabWidget, QFrame, QScrollArea, QSizePolicy, QSpacerItem,
    QProgressBar, QComboBox, QLineEdit, QCheckBox, QGroupBox,
    QSplitter, QListWidget, QListWidgetItem, QMessageBox,
    QStatusBar, QToolButton, QStackedWidget, QTextEdit
)

os.environ["YOLO_VERBOSE"] = "False"

# ═══════════════════════════════════════════════════════════════════
#  STYLESHEET — dark surveillance / industrial aesthetic
# ═══════════════════════════════════════════════════════════════════
QSS = """
QMainWindow, QWidget {
    background-color: #070a0f;
    color: #c8d8e8;
    font-family: 'Segoe UI', 'SF Pro Display', Arial;
    font-size: 13px;
}

/* ── Tab Bar ──────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #1c2a3a;
    border-top: none;
    background: #070a0f;
}
QTabBar {
    background: #0d1117;
}
QTabBar::tab {
    background: #0d1117;
    color: #5a7a96;
    padding: 12px 28px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    min-width: 140px;
}
QTabBar::tab:hover {
    color: #c8d8e8;
    background: #111820;
}
QTabBar::tab:selected {
    color: #00d4ff;
    border-bottom: 2px solid #00d4ff;
    background: #0d1117;
}

/* ── Frames / Panels ──────────────────────────────────────── */
QFrame#panel {
    background: #0d1117;
    border: 1px solid #1c2a3a;
    border-radius: 8px;
}
QFrame#dark_panel {
    background: #040608;
    border: 1px solid #1c2a3a;
    border-radius: 8px;
}
QFrame#header_bar {
    background: #131b25;
    border-bottom: 1px solid #1c2a3a;
    border-radius: 0px;
    max-height: 44px;
    min-height: 44px;
}

/* ── Buttons ──────────────────────────────────────────────── */
QPushButton {
    background: #131b25;
    color: #c8d8e8;
    border: 1px solid #243447;
    border-radius: 5px;
    padding: 9px 20px;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 1px;
}
QPushButton:hover {
    background: #182230;
    border-color: #007fa6;
    color: #00d4ff;
}
QPushButton:pressed {
    background: #0d1a28;
}
QPushButton:disabled {
    color: #2a3f55;
    border-color: #1a2535;
}
QPushButton#btn_primary {
    background: rgba(0,212,255,0.10);
    border-color: #007fa6;
    color: #00d4ff;
}
QPushButton#btn_primary:hover {
    background: rgba(0,212,255,0.20);
    box-shadow: 0 0 12px rgba(0,212,255,0.3);
}
QPushButton#btn_green {
    background: rgba(0,255,135,0.08);
    border-color: #00cc6a;
    color: #00ff87;
}
QPushButton#btn_green:hover { background: rgba(0,255,135,0.18); }
QPushButton#btn_red {
    background: rgba(255,71,87,0.08);
    border-color: #cc3344;
    color: #ff4757;
}
QPushButton#btn_red:hover { background: rgba(255,71,87,0.18); }
QPushButton#btn_amber {
    background: rgba(255,184,0,0.08);
    border-color: #cc9400;
    color: #ffb800;
}

/* ── Sliders ──────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #1c2a3a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #00d4ff;
    border: 2px solid #004a5e;
    width: 14px; height: 14px;
    border-radius: 7px;
    margin: -5px 0;
}
QSlider::handle:horizontal:hover {
    background: #33ddff;
    border-color: #00d4ff;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #007fa6, stop:1 #00d4ff);
    border-radius: 2px;
}

/* ── Progress bar ─────────────────────────────────────────── */
QProgressBar {
    background: #131b25;
    border: 1px solid #1c2a3a;
    border-radius: 4px;
    text-align: center;
    color: #5a7a96;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    height: 10px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #007fa6, stop:1 #00d4ff);
    border-radius: 3px;
}

/* ── List widget ──────────────────────────────────────────── */
QListWidget {
    background: #0d1117;
    border: 1px solid #1c2a3a;
    border-radius: 6px;
    color: #c8d8e8;
    outline: none;
}
QListWidget::item {
    padding: 10px 14px;
    border-bottom: 1px solid #131b25;
}
QListWidget::item:selected {
    background: rgba(0,212,255,0.10);
    color: #00d4ff;
    border-left: 3px solid #00d4ff;
}
QListWidget::item:hover {
    background: #131b25;
}

/* ── Labels ───────────────────────────────────────────────── */
QLabel#title_label {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 3px;
}
QLabel#subtitle_label {
    color: #5a7a96;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 3px;
}
QLabel#section_label {
    color: #007fa6;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
}
QLabel#value_label {
    color: #00d4ff;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    min-width: 48px;
    text-align: right;
}
QLabel#plate_text {
    color: #00d4ff;
    font-family: 'Courier New', monospace;
    font-size: 22px;
    letter-spacing: 4px;
    font-weight: bold;
}
QLabel#conf_text {
    color: #00ff87;
    font-family: 'Courier New', monospace;
    font-size: 11px;
}
QLabel#status_label {
    color: #5a7a96;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    letter-spacing: 1px;
}
QLabel#video_label {
    background: #020408;
    border: 1px solid #1c2a3a;
    border-radius: 8px;
    color: #2a3f55;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    qproperty-alignment: AlignCenter;
}

/* ── Combo box ────────────────────────────────────────────── */
QComboBox {
    background: #131b25;
    border: 1px solid #243447;
    border-radius: 5px;
    padding: 7px 14px;
    color: #c8d8e8;
    font-size: 12px;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #131b25;
    border: 1px solid #243447;
    selection-background-color: rgba(0,212,255,0.12);
    selection-color: #00d4ff;
}

/* ── Scrollbar ────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #0d1117; width: 8px; border: none;
}
QScrollBar::handle:vertical {
    background: #243447; border-radius: 4px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #2e4560; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* ── Text Edit (log) ──────────────────────────────────────── */
QTextEdit {
    background: #040608;
    border: 1px solid #1c2a3a;
    border-radius: 6px;
    color: #5a7a96;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    padding: 8px;
}

/* ── Status bar ───────────────────────────────────────────── */
QStatusBar {
    background: #0d1117;
    border-top: 1px solid #1c2a3a;
    color: #5a7a96;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    letter-spacing: 1px;
}

/* ── Group box ────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #1c2a3a;
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px;
    font-size: 10px;
    font-family: 'Courier New', monospace;
    color: #007fa6;
    letter-spacing: 2px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px; top: -1px;
    padding: 0 6px;
    background: #070a0f;
}
"""


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VEHICLE_CLASSES  = {"car", "motorcycle", "bus", "truck"}
PLATE_CLASS_NAME = "license_plate"

_TRT_LOGGER = None   # must be module-level global


def cv2_to_qpixmap(frame: np.ndarray) -> QPixmap:
    h, w, ch = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def fit_pixmap(pixmap: QPixmap, container: QLabel) -> QPixmap:
    return pixmap.scaled(container.width(), container.height(),
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ═══════════════════════════════════════════════════════════════════
#  OCR BACKENDS  (identical interface: __call__ returns (B,C,T))
# ═══════════════════════════════════════════════════════════════════
from util import CHARS, safe_crop, match_plate_to_vehicle, batch_read_license_plates


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
        opts = ort.SessionOptions(); opts.log_severity_level = 3
        self._s  = ort.InferenceSession(path, sess_options=opts, providers=providers)
        self._in = self._s.get_inputs()[0].name
        self._out = self._s.get_outputs()[0].name
        self._gpu = torch.cuda.is_available()

    def __call__(self, x):
        out = torch.from_numpy(
            self._s.run([self._out], {self._in: x.cpu().numpy().astype(np.float32)})[0])
        if self._gpu: out = out.cuda()
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
        self._ctx   = self._eng.create_execution_context()
        self._in_n  = self._eng.get_tensor_name(0)
        self._out_n = self._eng.get_tensor_name(1)

    def __call__(self, x):
        x = x.contiguous().cuda().float()
        self._ctx.set_input_shape(self._in_n, tuple(x.shape))
        out = torch.empty(
            tuple(self._ctx.get_tensor_shape(self._out_n)),
            dtype=torch.float32, device="cuda")
        self._ctx.set_tensor_address(self._in_n,  x.data_ptr())
        self._ctx.set_tensor_address(self._out_n, out.data_ptr())
        self._ctx.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.permute(1, 2, 0)

    def eval(self): return self
    def to(self, *a, **k): return self
    def parameters(self): return iter([])


def load_ocr_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pth":    return _PTHBackend(path)
    if ext == ".onnx":   return _ONNXBackend(path)
    if ext == ".engine": return _TRTBackend(path)
    raise ValueError(f"Unsupported extension '{ext}'")


# ── Detection helper ─────────────────────────────────────────────
from ultralytics import YOLO as _YOLO

_det_model  = None
_ocr_model  = None
_models_ok  = False


def init_models(det_path: str, ocr_path: str):
    global _det_model, _ocr_model, _models_ok
    _det_model = _YOLO(det_path, task="detect")
    _ocr_model = load_ocr_model(ocr_path)
    _models_ok = True


def _detect(frame, conf_thr=0.30):
    res  = _det_model.predict(
        frame, conf=conf_thr, verbose=False,
        device=0 if torch.cuda.is_available() else "cpu", imgsz=640)
    dets = []
    if res and res[0].boxes is not None:
        for b in res[0].boxes:
            dets.append({
                "class_name": str(_det_model.names.get(int(b.cls[0]), "")).lower(),
                "bbox":       tuple(map(int, b.xyxy[0])),
                "conf":       float(b.conf[0]),
            })
    return dets


def _split(dets):
    return ([d for d in dets if d["class_name"] in VEHICLE_CLASSES],
            [d for d in dets if d["class_name"] == PLATE_CLASS_NAME])


class SimpleTracker:
    def __init__(self, iou_thr=0.3, max_lost=15):
        self.next_id = 0; self.tracks = {}
        self.iou_thr = iou_thr; self.max_lost = max_lost

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
        iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
        inter=iw*ih
        union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/union if union>0 else 0.0

    def update(self, dets):
        updated, assigned = {}, set()
        for tid, tr in self.tracks.items():
            best, bi = 0.0, -1
            for i, d in enumerate(dets):
                if i in assigned: continue
                v = self._iou(tr["bbox"], d["bbox"])
                if v > best: best, bi = v, i
            if best > self.iou_thr:
                updated[tid] = {"bbox": dets[bi]["bbox"], "lost": 0}
                dets[bi]["track_id"] = tid; assigned.add(bi)
            else:
                tr["lost"] += 1
                if tr["lost"] < self.max_lost: updated[tid] = tr
        for i, d in enumerate(dets):
            if i not in assigned:
                updated[self.next_id] = {"bbox": d["bbox"], "lost": 0}
                d["track_id"] = self.next_id; self.next_id += 1
        self.tracks = updated
        return dets


def _process_frame(frame, tracker, plate_memory, conf_thr=0.30, ocr_conf_thr=0.30):
    """
    Runs detection + OCR on one frame.
    Returns (annotated_frame, list[{"text":str,"conf":float}])
    """
    dets             = _detect(frame, conf_thr)
    vehicles, plates = _split(dets)
    vehicles         = tracker.update(vehicles)

    crops, plate_objs = [], []
    for pl in plates:
        c = safe_crop(frame, pl["bbox"])
        if c is not None and c.size > 0:
            crops.append(c); plate_objs.append(pl)

    new_detections = []
    if crops:
        results = batch_read_license_plates(crops, _ocr_model)
        for pl, (text, conf) in zip(plate_objs, results):
            if text and conf >= ocr_conf_thr:
                mv = match_plate_to_vehicle(pl["bbox"], vehicles)
                if mv:
                    vid  = mv["track_id"]
                    prev = plate_memory.get(vid)
                    if prev is None or conf > prev["conf"]:
                        plate_memory[vid] = {"text": text, "conf": conf}
                        new_detections.append({"text": text, "conf": round(float(conf), 3)})

    # ── Draw ──────────────────────────────────────────────────────
    for pl in plates:
        mv   = match_plate_to_vehicle(pl["bbox"], vehicles)
        info = plate_memory.get(mv["track_id"]) if mv else None
        txt  = info["text"] if info else "PLATE"
        x1,y1,x2,y2 = pl["bbox"]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,180,0), 2)
        cv2.putText(frame, txt, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,180,0), 2)

    for v in vehicles:
        x1,y1,x2,y2 = v["bbox"]
        info  = plate_memory.get(v.get("track_id"))
        label = v["class_name"].upper() + (f" | {info['text']}" if info else "")
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,80,220), 2)
        cv2.putText(frame, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,180,255), 2)

    return frame, new_detections


# ═══════════════════════════════════════════════════════════════════
#  WORKER THREADS
# ═══════════════════════════════════════════════════════════════════

class ImageWorker(QThread):
    result  = pyqtSignal(np.ndarray, list)   # frame, plates
    error   = pyqtSignal(str)

    def __init__(self, frame, conf_thr, ocr_conf_thr):
        super().__init__()
        self.frame       = frame
        self.conf_thr    = conf_thr
        self.ocr_thr     = ocr_conf_thr

    def run(self):
        try:
            tracker  = SimpleTracker()
            memory   = {}
            ann, pls = _process_frame(self.frame.copy(), tracker, memory,
                                      self.conf_thr, self.ocr_thr)
            self.result.emit(ann, pls)
        except Exception as e:
            self.error.emit(traceback.format_exc())


class OfflineVideoWorker(QThread):
    progress    = pyqtSignal(int)           # 0-100
    frame_ready = pyqtSignal(np.ndarray)   # preview frame
    plate_found = pyqtSignal(str, float)   # text, conf
    finished_ok = pyqtSignal(str, str)     # video_path, csv_path
    error       = pyqtSignal(str)

    def __init__(self, src, out_video, out_csv, conf_thr, ocr_conf_thr):
        super().__init__()
        self.src        = src
        self.out_video  = out_video
        self.out_csv    = out_csv
        self.conf_thr   = conf_thr
        self.ocr_thr    = ocr_conf_thr
        self._stop      = False

    def stop(self): self._stop = True

    def run(self):
        cap     = cv2.VideoCapture(self.src)
        fps     = cap.get(cv2.CAP_PROP_FPS) or 25
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        W, H    = 1280, 960
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer  = cv2.VideoWriter(self.out_video, fourcc, fps, (W, H))
        tracker = SimpleTracker()
        memory  = {}
        rows    = []
        fc      = 0

        try:
            while cap.isOpened() and not self._stop:
                ret, frame = cap.read()
                if not ret: break
                fc += 1
                frame = cv2.resize(frame, (W, H))
                ann, plates = _process_frame(frame, tracker, memory,
                                             self.conf_thr, self.ocr_thr)
                writer.write(ann)

                if fc % 8 == 0:
                    self.frame_ready.emit(ann.copy())

                for p in plates:
                    rows.append([fc, p["text"], f"{p['conf']:.3f}"])
                    self.plate_found.emit(p["text"], p["conf"])

                self.progress.emit(min(99, int(fc / total * 100)))

        except Exception as e:
            self.error.emit(traceback.format_exc())
            return
        finally:
            cap.release(); writer.release()

        if not self._stop:
            with open(self.out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Frame","Plate","Confidence"])
                w.writerows(rows)
            self.progress.emit(100)
            self.finished_ok.emit(self.out_video, self.out_csv)


class RealtimeVideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    plate_found = pyqtSignal(str, float)
    fps_update  = pyqtSignal(float)
    finished    = pyqtSignal()
    error       = pyqtSignal(str)

    def __init__(self, src, conf_thr, ocr_conf_thr, target_fps=25):
        super().__init__()
        self.src        = src          # path or 0 for webcam
        self.conf_thr   = conf_thr
        self.ocr_thr    = ocr_conf_thr
        self.target_fps = target_fps
        self._stop      = False

    def stop(self): self._stop = True

    def run(self):
        cap     = cv2.VideoCapture(self.src)
        tracker = SimpleTracker()
        memory  = {}
        frame_delay = 1.0 / self.target_fps
        fps_t   = time.monotonic()
        fps_cnt = 0

        try:
            while cap.isOpened() and not self._stop:
                t0  = time.monotonic()
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.resize(frame, (1280, 960))
                ann, plates = _process_frame(frame, tracker, memory,
                                             self.conf_thr, self.ocr_thr)
                self.frame_ready.emit(ann.copy())

                for p in plates:
                    self.plate_found.emit(p["text"], p["conf"])

                fps_cnt += 1
                elapsed_fps = time.monotonic() - fps_t
                if elapsed_fps >= 1.0:
                    self.fps_update.emit(fps_cnt / elapsed_fps)
                    fps_cnt = 0; fps_t = time.monotonic()

                sleep = frame_delay - (time.monotonic() - t0)
                if sleep > 0:
                    self.msleep(int(sleep * 1000))

        except Exception as e:
            self.error.emit(traceback.format_exc())
        finally:
            cap.release()
            self.finished.emit()


# ═══════════════════════════════════════════════════════════════════
#  REUSABLE WIDGETS
# ═══════════════════════════════════════════════════════════════════

class SectionLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(f"// {text}", parent)
        self.setObjectName("section_label")
        self.setContentsMargins(0, 8, 0, 8)


class VideoLabel(QLabel):
    def __init__(self, placeholder="AWAITING INPUT ···", parent=None):
        super().__init__(parent)
        self.setObjectName("video_label")
        self.placeholder = placeholder
        self.setText(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None

    def set_frame(self, frame: np.ndarray):
        px = cv2_to_qpixmap(frame)
        self._pixmap = px
        self._update_display()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._pixmap:
            self._update_display()

    def _update_display(self):
        scaled = self._pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def clear_display(self):
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(self.placeholder)


class PlateCard(QFrame):
    """Single detected plate entry card."""
    def __init__(self, text, conf, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.setStyleSheet("""
            QFrame#panel {
                border-left: 3px solid #00d4ff;
                border-radius: 4px;
                background: #131b25;
                padding: 6px;
            }
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)

        lbl = QLabel(text)
        lbl.setObjectName("plate_text")
        lbl.setFont(QFont("Courier New", 18, QFont.Bold))
        lbl.setStyleSheet("color: #00d4ff; letter-spacing: 4px;")

        conf_lbl = QLabel(f"{conf*100:.1f}%")
        conf_lbl.setObjectName("conf_text")
        conf_lbl.setStyleSheet("color: #00ff87; font-family: 'Courier New'; font-size: 12px;")
        conf_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lay.addWidget(lbl)
        lay.addStretch()
        lay.addWidget(conf_lbl)


class ControlSlider(QWidget):
    """Labelled slider: section label + slider + live value display."""
    valueChanged = pyqtSignal(float)

    def __init__(self, label, min_v, max_v, default, scale=1.0, fmt="{:.0f}", parent=None):
        super().__init__(parent)
        self.scale = scale
        self.fmt   = fmt

        row  = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        lbl = QLabel(label)
        lbl.setObjectName("section_label")
        lbl.setMinimumWidth(160)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_v / scale), int(max_v / scale))
        self.slider.setValue(int(default / scale))

        self.val_lbl = QLabel(self.fmt.format(default))
        self.val_lbl.setObjectName("value_label")
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_lbl.setMinimumWidth(52)

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


class StatusDot(QLabel):
    def __init__(self, parent=None):
        super().__init__("● OFFLINE", parent)
        self.setStyleSheet("color: #5a7a96; font-family: 'Courier New'; "
                           "font-size: 10px; letter-spacing: 2px;")

    def set_online(self, label="ONLINE"):
        self.setText(f"● {label}")
        self.setStyleSheet("color: #00ff87; font-family: 'Courier New'; "
                           "font-size: 10px; letter-spacing: 2px;")

    def set_offline(self, label="OFFLINE"):
        self.setText(f"● {label}")
        self.setStyleSheet("color: #5a7a96; font-family: 'Courier New'; "
                           "font-size: 10px; letter-spacing: 2px;")

    def set_busy(self, label="PROCESSING"):
        self.setText(f"◌ {label}")
        self.setStyleSheet("color: #ffb800; font-family: 'Courier New'; "
                           "font-size: 10px; letter-spacing: 2px;")


# ═══════════════════════════════════════════════════════════════════
#  TAB 1 — IMAGE
# ═══════════════════════════════════════════════════════════════════

class ImageTab(QWidget):
    log_msg = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker     = None
        self._orig_frame = None
        self._ann_frame  = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(20)

        # ── Left: controls ────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(14)

        left.addWidget(SectionLabel("IMAGE CONTROLS"))

        self.btn_open = QPushButton("📂  OPEN IMAGE")
        self.btn_open.setObjectName("btn_primary")
        self.btn_open.setMinimumHeight(42)
        self.btn_open.clicked.connect(self.open_image)
        left.addWidget(self.btn_open)

        # Sliders
        self.sl_conf = ControlSlider(
            "DET CONFIDENCE", 0.10, 0.90, 0.30, scale=0.01, fmt="{:.2f}")
        self.sl_ocr = ControlSlider(
            "OCR THRESHOLD", 0.10, 0.90, 0.30, scale=0.01, fmt="{:.2f}")
        self.sl_scale = ControlSlider(
            "DISPLAY SCALE", 10, 200, 100, scale=1.0, fmt="{:.0f}%")
        self.sl_scale.valueChanged.connect(self._apply_scale)

        left.addWidget(self.sl_conf)
        left.addWidget(self.sl_ocr)
        left.addWidget(self.sl_scale)

        self.btn_run = QPushButton("▶  RUN DETECTION")
        self.btn_run.setObjectName("btn_green")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_detection)
        left.addWidget(self.btn_run)

        self.btn_save = QPushButton("💾  SAVE RESULT")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_result)
        left.addWidget(self.btn_save)

        self.status_dot = StatusDot()
        left.addWidget(self.status_dot)

        left.addWidget(SectionLabel("DETECTED PLATES"))

        self.plates_scroll = QScrollArea()
        self.plates_scroll.setWidgetResizable(True)
        self.plates_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        plates_container = QWidget()
        self.plates_layout = QVBoxLayout(plates_container)
        self.plates_layout.setSpacing(8)
        self.plates_layout.setContentsMargins(0, 0, 0, 0)
        self.plates_layout.addStretch()
        self.plates_scroll.setWidget(plates_container)
        self.plates_scroll.setMinimumHeight(180)
        left.addWidget(self.plates_scroll, 1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(320)

        # ── Right: image display ───────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(SectionLabel("PREVIEW"))
        self.img_label = VideoLabel("DROP AN IMAGE OR CLICK OPEN")
        right.addWidget(self.img_label, 1)

        right_widget = QWidget()
        right_widget.setLayout(right)

        root.addWidget(left_widget)
        root.addWidget(right_widget, 1)

    # ── drag & drop ───────────────────────────────────────────────
    def setAcceptDrops(self, v):
        super().setAcceptDrops(v)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.accept()

    def dropEvent(self, e):
        path = e.mimeData().urls()[0].toLocalFile()
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            self._load_image(path)

    # ── slots ─────────────────────────────────────────────────────
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.jpg *.jpeg *.png)")
        if path: self._load_image(path)

    def _load_image(self, path):
        frame = cv2.imread(path)
        if frame is None: return
        self._orig_frame = frame
        self._ann_frame  = None
        self.img_label.set_frame(frame)
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.status_dot.set_offline("IMAGE LOADED")
        self._clear_plates()
        self.log_msg.emit(f"Image loaded: {os.path.basename(path)}")

    def run_detection(self):
        if self._orig_frame is None or self._worker is not None: return
        self.status_dot.set_busy("PROCESSING")
        self.btn_run.setEnabled(False)

        self._worker = ImageWorker(
            self._orig_frame.copy(),
            self.sl_conf.value(),
            self.sl_ocr.value()
        )
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_result(self, frame, plates):
        self._ann_frame = frame
        self._apply_scale(self.sl_scale.value())
        self._clear_plates()
        for p in plates:
            self._add_plate_card(p["text"], p["conf"])
        if not plates:
            no_lbl = QLabel("NO PLATES DETECTED")
            no_lbl.setObjectName("status_label")
            no_lbl.setAlignment(Qt.AlignCenter)
            self.plates_layout.insertWidget(0, no_lbl)
        self.btn_save.setEnabled(True)
        self.status_dot.set_online(f"DONE · {len(plates)} PLATE(S)")
        self.log_msg.emit(f"Detection complete: {len(plates)} plate(s)")

    def _apply_scale(self, scale_pct):
        frame = self._ann_frame if self._ann_frame is not None else self._orig_frame
        if frame is None: return
        pct = scale_pct / 100.0
        h, w = frame.shape[:2]
        new_w, new_h = max(1, int(w * pct)), max(1, int(h * pct))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.img_label.set_frame(resized)

    def _on_error(self, msg):
        self.status_dot.set_offline("ERROR")
        self.log_msg.emit("ERROR: " + msg[:120])

    def _on_finished(self):
        self._worker = None
        self.btn_run.setEnabled(True)

    def save_result(self):
        if self._ann_frame is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "result.jpg", "Images (*.jpg *.png)")
        if path:
            cv2.imwrite(path, self._ann_frame)
            self.log_msg.emit(f"Saved: {path}")

    def _clear_plates(self):
        while self.plates_layout.count() > 1:
            item = self.plates_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def _add_plate_card(self, text, conf):
        card = PlateCard(text, conf)
        self.plates_layout.insertWidget(
            self.plates_layout.count() - 1, card)


# ═══════════════════════════════════════════════════════════════════
#  TAB 2 — OFFLINE VIDEO
# ═══════════════════════════════════════════════════════════════════

class OfflineTab(QWidget):
    log_msg = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._out_video = None
        self._out_csv   = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(20)

        # ── Left: controls ────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(14)
        left.addWidget(SectionLabel("OFFLINE PROCESSING"))

        self.btn_open = QPushButton("📂  OPEN VIDEO")
        self.btn_open.setObjectName("btn_primary")
        self.btn_open.setMinimumHeight(42)
        self.btn_open.clicked.connect(self.open_video)
        left.addWidget(self.btn_open)

        self.src_lbl = QLabel("NO FILE SELECTED")
        self.src_lbl.setObjectName("status_label")
        self.src_lbl.setWordWrap(True)
        left.addWidget(self.src_lbl)

        self.sl_conf = ControlSlider(
            "DET CONFIDENCE", 0.10, 0.90, 0.30, scale=0.01, fmt="{:.2f}")
        self.sl_ocr = ControlSlider(
            "OCR THRESHOLD", 0.10, 0.90, 0.30, scale=0.01, fmt="{:.2f}")
        left.addWidget(self.sl_conf)
        left.addWidget(self.sl_ocr)

        self.btn_start = QPushButton("▶  START PROCESSING")
        self.btn_start.setObjectName("btn_green")
        self.btn_start.setMinimumHeight(42)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_processing)
        left.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btn_red")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_processing)
        left.addWidget(self.btn_stop)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setMaximumHeight(14)
        left.addWidget(self.progress)

        self.status_dot = StatusDot()
        left.addWidget(self.status_dot)

        # Download buttons
        self.btn_dl_video = QPushButton("⬇  SAVE VIDEO")
        self.btn_dl_video.setObjectName("btn_amber")
        self.btn_dl_video.setEnabled(False)
        self.btn_dl_video.clicked.connect(self.save_video)
        left.addWidget(self.btn_dl_video)

        self.btn_dl_csv = QPushButton("⬇  SAVE CSV")
        self.btn_dl_csv.setObjectName("btn_amber")
        self.btn_dl_csv.setEnabled(False)
        self.btn_dl_csv.clicked.connect(self.save_csv)
        left.addWidget(self.btn_dl_csv)

        left.addWidget(SectionLabel("PLATES LOG"))
        self.plates_list = QListWidget()
        self.plates_list.setMinimumHeight(160)
        left.addWidget(self.plates_list, 1)

        left_w = QWidget(); left_w.setLayout(left)
        left_w.setMaximumWidth(320)

        # ── Right: preview ─────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(SectionLabel("PROCESSING PREVIEW"))
        self.vid_label = VideoLabel("SELECT A VIDEO FILE TO BEGIN")
        right.addWidget(self.vid_label, 1)

        right_w = QWidget(); right_w.setLayout(right)

        root.addWidget(left_w)
        root.addWidget(right_w, 1)

        self._src_path = None

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self._src_path = path
            self.src_lbl.setText(os.path.basename(path))
            self.btn_start.setEnabled(True)
            self.status_dot.set_offline("FILE READY")
            self.log_msg.emit(f"Video selected: {os.path.basename(path)}")

    def start_processing(self):
        if not self._src_path or self._worker: return
        base = os.path.splitext(self._src_path)[0]
        self._out_video = base + "_processed.mp4"
        self._out_csv   = base + "_plates.csv"
        self.plates_list.clear()
        self.progress.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_dl_video.setEnabled(False)
        self.btn_dl_csv.setEnabled(False)
        self.status_dot.set_busy()

        self._worker = OfflineVideoWorker(
            self._src_path, self._out_video, self._out_csv,
            self.sl_conf.value(), self.sl_ocr.value())
        self._worker.progress.connect(self.progress.setValue)
        self._worker.frame_ready.connect(self.vid_label.set_frame)
        self._worker.plate_found.connect(self._on_plate)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def stop_processing(self):
        if self._worker: self._worker.stop()

    def _on_plate(self, text, conf):
        item = QListWidgetItem(f"  {text}   {conf*100:.1f}%")
        item.setFont(QFont("Courier New", 12))
        item.setForeground(QColor("#00d4ff"))
        self.plates_list.insertItem(0, item)

    def _on_done(self, vpath, cpath):
        self.status_dot.set_online("COMPLETE")
        self.btn_dl_video.setEnabled(True)
        self.btn_dl_csv.setEnabled(True)
        self.log_msg.emit(f"Offline processing done: {vpath}")

    def _on_error(self, msg):
        self.status_dot.set_offline("ERROR")
        self.log_msg.emit("ERROR: " + msg[:120])

    def _on_finished(self):
        self._worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def save_video(self):
        if not self._out_video or not os.path.exists(self._out_video): return
        dst, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "processed.mp4", "Video (*.mp4)")
        if dst:
            import shutil; shutil.copy2(self._out_video, dst)
            self.log_msg.emit(f"Video saved: {dst}")

    def save_csv(self):
        if not self._out_csv or not os.path.exists(self._out_csv): return
        dst, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "plates.csv", "CSV (*.csv)")
        if dst:
            import shutil; shutil.copy2(self._out_csv, dst)
            self.log_msg.emit(f"CSV saved: {dst}")


# ═══════════════════════════════════════════════════════════════════
#  TAB 3 — REALTIME  (video file  OR  webcam)
# ═══════════════════════════════════════════════════════════════════

class RealtimeTab(QWidget):
    log_msg = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker     = None
        self._all_plates = {}    # text → best_conf
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(20)

        # ── Left ───────────────────────────────────────────────────
        left = QVBoxLayout(); left.setSpacing(14)
        left.addWidget(SectionLabel("REALTIME CONTROLS"))

        # Source selection
        src_grp = QGroupBox("INPUT SOURCE")
        src_lay = QVBoxLayout(src_grp)
        src_lay.setSpacing(8)

        self.btn_open_vid = QPushButton("📂  OPEN VIDEO FILE")
        self.btn_open_vid.setObjectName("btn_primary")
        self.btn_open_vid.clicked.connect(self.open_video)
        src_lay.addWidget(self.btn_open_vid)

        cam_row = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["Webcam 0", "Webcam 1", "Webcam 2"])
        self.btn_open_cam = QPushButton("🎥  OPEN WEBCAM")
        self.btn_open_cam.setObjectName("btn_primary")
        self.btn_open_cam.clicked.connect(self.open_webcam)
        cam_row.addWidget(self.cam_combo, 1)
        cam_row.addWidget(self.btn_open_cam)
        src_lay.addLayout(cam_row)

        self.src_lbl = QLabel("NO SOURCE")
        self.src_lbl.setObjectName("status_label")
        src_lay.addWidget(self.src_lbl)
        left.addWidget(src_grp)

        self.sl_conf = ControlSlider(
            "DET CONFIDENCE", 0.10, 0.90, 0.30, scale=0.01, fmt="{:.2f}")
        self.sl_ocr = ControlSlider(
            "OCR THRESHOLD", 0.10, 0.90, 0.30, scale=0.01, fmt="{:.2f}")
        left.addWidget(self.sl_conf)
        left.addWidget(self.sl_ocr)

        ctrl_row = QHBoxLayout()
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btn_red")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_stream)
        ctrl_row.addWidget(self.btn_stop)
        left.addLayout(ctrl_row)

        # FPS / status
        info_row = QHBoxLayout()
        self.status_dot = StatusDot()
        self.fps_lbl = QLabel("0.0 FPS")
        self.fps_lbl.setObjectName("value_label")
        self.fps_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_row.addWidget(self.status_dot)
        info_row.addStretch()
        info_row.addWidget(self.fps_lbl)
        left.addLayout(info_row)

        # Clear plates
        self.btn_clear = QPushButton("🗑  CLEAR PLATES")
        self.btn_clear.clicked.connect(self.clear_plates)
        left.addWidget(self.btn_clear)

        left.addWidget(SectionLabel("PLATES DETECTED"))
        self.plates_list = QListWidget()
        left.addWidget(self.plates_list, 1)

        left_w = QWidget(); left_w.setLayout(left)
        left_w.setMaximumWidth(320)

        # ── Right: video display ───────────────────────────────────
        right = QVBoxLayout(); right.setSpacing(8)
        right.addWidget(SectionLabel("LIVE FEED"))
        self.vid_label = VideoLabel("SELECT SOURCE TO BEGIN STREAM")
        right.addWidget(self.vid_label, 1)

        right_w = QWidget(); right_w.setLayout(right)

        root.addWidget(left_w)
        root.addWidget(right_w, 1)

        self._src = None

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self._src = path
            self.src_lbl.setText(os.path.basename(path))
            self._start_stream()

    def open_webcam(self):
        idx = self.cam_combo.currentIndex()
        self._src = idx
        self.src_lbl.setText(f"Webcam {idx}")
        self._start_stream()

    def _start_stream(self):
        if self._worker: self.stop_stream()
        self.clear_plates()
        self.status_dot.set_busy("STREAMING")
        self.btn_stop.setEnabled(True)

        self._worker = RealtimeVideoWorker(
            self._src, self.sl_conf.value(), self.sl_ocr.value())
        self._worker.frame_ready.connect(self.vid_label.set_frame)
        self._worker.plate_found.connect(self._on_plate)
        self._worker.fps_update.connect(self._on_fps)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def stop_stream(self):
        if self._worker:
            self._worker.stop()

    def _on_plate(self, text, conf):
        prev = self._all_plates.get(text)
        if prev is None or conf > prev:
            self._all_plates[text] = conf
            # remove old entry if exists
            for i in range(self.plates_list.count()):
                if self.plates_list.item(i).data(Qt.UserRole) == text:
                    self.plates_list.takeItem(i); break
            item = QListWidgetItem(f"  {text}   {conf*100:.1f}%")
            item.setFont(QFont("Courier New", 13))
            item.setForeground(QColor("#00ff87"))
            item.setData(Qt.UserRole, text)
            self.plates_list.insertItem(0, item)
        self.log_msg.emit(f"Plate: {text} ({conf*100:.1f}%)")

    def _on_fps(self, fps):
        self.fps_lbl.setText(f"{fps:.1f} FPS")

    def _on_error(self, msg):
        self.status_dot.set_offline("ERROR")
        self.log_msg.emit("ERROR: " + msg[:120])

    def _on_finished(self):
        self._worker = None
        self.btn_stop.setEnabled(False)
        self.status_dot.set_offline("STOPPED")
        self.vid_label.clear_display()

    def clear_plates(self):
        self._all_plates.clear()
        self.plates_list.clear()


# ═══════════════════════════════════════════════════════════════════
#  MODEL CONFIG DIALOG
# ═══════════════════════════════════════════════════════════════════

class ModelConfigWidget(QWidget):
    models_loaded = pyqtSignal(bool, str)   # ok, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(16)

        lay.addWidget(SectionLabel("MODEL PATHS"))

        # YOLO path
        yolo_row = QHBoxLayout()
        yolo_lbl = QLabel("YOLO MODEL")
        yolo_lbl.setObjectName("section_label")
        yolo_lbl.setMinimumWidth(120)
        self.yolo_edit = QLineEdit("models/best_of_both.engine")
        self.yolo_edit.setStyleSheet(
            "background:#131b25;border:1px solid #243447;border-radius:4px;"
            "padding:7px;color:#c8d8e8;font-family:'Courier New';font-size:11px;")
        yolo_browse = QPushButton("···")
        yolo_browse.setMaximumWidth(40)
        yolo_browse.clicked.connect(
            lambda: self._browse(self.yolo_edit, "Model Files (*.pt *.engine *.onnx)"))
        yolo_row.addWidget(yolo_lbl)
        yolo_row.addWidget(self.yolo_edit, 1)
        yolo_row.addWidget(yolo_browse)
        lay.addLayout(yolo_row)

        # OCR path
        ocr_row = QHBoxLayout()
        ocr_lbl = QLabel("OCR MODEL")
        ocr_lbl.setObjectName("section_label")
        ocr_lbl.setMinimumWidth(120)
        self.ocr_edit = QLineEdit("models/license_plate_detection_dynamic.engine")
        self.ocr_edit.setStyleSheet(self.yolo_edit.styleSheet())
        ocr_browse = QPushButton("···")
        ocr_browse.setMaximumWidth(40)
        ocr_browse.clicked.connect(
            lambda: self._browse(self.ocr_edit, "Model Files (*.pth *.onnx *.engine)"))
        ocr_row.addWidget(ocr_lbl)
        ocr_row.addWidget(self.ocr_edit, 1)
        ocr_row.addWidget(ocr_browse)
        lay.addLayout(ocr_row)

        self.btn_load = QPushButton("⚡  LOAD MODELS")
        self.btn_load.setObjectName("btn_primary")
        self.btn_load.setMinimumHeight(44)
        self.btn_load.clicked.connect(self.load_models)
        lay.addWidget(self.btn_load)

        self.load_status = QLabel("MODELS NOT LOADED")
        self.load_status.setObjectName("status_label")
        self.load_status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.load_status)

        lay.addStretch()

    def _browse(self, edit, filt):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", filt)
        if path: edit.setText(path)

    def load_models(self):
        self.btn_load.setEnabled(False)
        self.load_status.setText("LOADING ···")
        det_path = self.yolo_edit.text().strip()
        ocr_path = self.ocr_edit.text().strip()

        class _Loader(QThread):
            done = pyqtSignal(bool, str)
            def __init__(self, dp, op): super().__init__(); self.dp=dp; self.op=op
            def run(self):
                try:
                    init_models(self.dp, self.op)
                    self.done.emit(True, "MODELS LOADED SUCCESSFULLY")
                except Exception as e:
                    self.done.emit(False, str(e)[:120])

        self._loader = _Loader(det_path, ocr_path)
        self._loader.done.connect(self._on_loaded)
        self._loader.start()

    def _on_loaded(self, ok, msg):
        self.btn_load.setEnabled(True)
        self.load_status.setText(msg)
        if ok:
            self.load_status.setStyleSheet("color:#00ff87;font-family:'Courier New';font-size:11px;")
        else:
            self.load_status.setStyleSheet("color:#ff4757;font-family:'Courier New';font-size:11px;")
        self.models_loaded.emit(ok, msg)


# ═══════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LPR SYSTEM v9  ·  License Plate Recognition")
        self.setMinimumSize(1280, 780)
        self.resize(1400, 860)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("header_bar")
        header.setFixedHeight(54)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(24, 0, 24, 0)

        logo_lbl = QLabel("LPR <span style='color:#00d4ff'>SYS</span>")
        logo_lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        logo_lbl.setStyleSheet("color:white;letter-spacing:3px;")
        logo_lbl.setTextFormat(Qt.RichText)

        sub_lbl = QLabel("GENERALISED LPRNET  ·  v9")
        sub_lbl.setStyleSheet(
            "color:#5a7a96;font-family:'Courier New';font-size:9px;letter-spacing:3px;")

        self.model_status_dot = StatusDot()

        logo_col = QVBoxLayout()
        logo_col.setSpacing(0)
        logo_col.addWidget(logo_lbl)
        logo_col.addWidget(sub_lbl)

        h_lay.addLayout(logo_col)
        h_lay.addStretch()
        h_lay.addWidget(self.model_status_dot)

        main_lay.addWidget(header)

        # ── Stacked: config vs main tabs ─────────────────────────
        self.stack = QStackedWidget()
        main_lay.addWidget(self.stack, 1)

        # --- Page 0: model config --------------------------------
        self.cfg_widget = ModelConfigWidget()
        self.cfg_widget.models_loaded.connect(self._on_models_loaded)
        self.stack.addWidget(self.cfg_widget)

        # --- Page 1: main tabs -----------------------------------
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.North)
        tab_widget.setDocumentMode(True)

        self.img_tab  = ImageTab()
        self.off_tab  = OfflineTab()
        self.rt_tab   = RealtimeTab()

        tab_widget.addTab(self.img_tab,  "🖼   IMAGE")
        tab_widget.addTab(self.off_tab,  "📼   OFFLINE VIDEO")
        tab_widget.addTab(self.rt_tab,   "⚡   REALTIME / WEBCAM")

        # ── Log panel below tabs ──────────────────────────────────
        page1 = QWidget()
        p1_lay = QVBoxLayout(page1)
        p1_lay.setContentsMargins(0, 0, 0, 0)
        p1_lay.setSpacing(0)

        p1_lay.addWidget(tab_widget, 1)

        log_frame = QFrame()
        log_frame.setFixedHeight(80)
        log_frame.setStyleSheet("background:#040608;border-top:1px solid #1c2a3a;")
        log_lay = QHBoxLayout(log_frame)
        log_lay.setContentsMargins(12, 6, 12, 6)

        log_hdr = QLabel("LOG")
        log_hdr.setObjectName("section_label")
        log_hdr.setFixedWidth(36)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(62)
        self.log_edit.setStyleSheet(
            "background:transparent;border:none;color:#5a7a96;"
            "font-family:'Courier New';font-size:10px;padding:0;")

        log_lay.addWidget(log_hdr)
        log_lay.addWidget(self.log_edit, 1)
        p1_lay.addWidget(log_frame)

        self.stack.addWidget(page1)

        # Connect log signals
        self.img_tab.log_msg.connect(self._log)
        self.off_tab.log_msg.connect(self._log)
        self.rt_tab.log_msg.connect(self._log)

        # ── Status bar ─────────────────────────────────────────────
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.sb_left = QLabel(f"  DEVICE: {DEVICE.upper()}")
        self.sb_left.setObjectName("status_label")
        self.sb_right = QLabel("MODELS NOT LOADED  ")
        self.sb_right.setObjectName("status_label")
        sb.addWidget(self.sb_left)
        sb.addPermanentWidget(self.sb_right)

        # Enable drag & drop on image tab
        self.img_tab.setAcceptDrops(True)

        # Start on config page
        self.stack.setCurrentIndex(0)

    def _on_models_loaded(self, ok, msg):
        if ok:
            self.model_status_dot.set_online("MODELS READY")
            self.sb_right.setText(f"MODELS LOADED  ")
            self.sb_right.setStyleSheet(
                "color:#00ff87;font-family:'Courier New';font-size:10px;")
            self.stack.setCurrentIndex(1)
            self._log("Models loaded OK — ready.")
        else:
            self.model_status_dot.set_offline("LOAD FAILED")
            self._log("Model load FAILED: " + msg)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_edit.append(f"[{ts}]  {msg}")
        # auto-scroll
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Apply dark palette so system widgets match
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor("#070a0f"))
    pal.setColor(QPalette.WindowText,      QColor("#c8d8e8"))
    pal.setColor(QPalette.Base,            QColor("#0d1117"))
    pal.setColor(QPalette.AlternateBase,   QColor("#131b25"))
    pal.setColor(QPalette.ToolTipBase,     QColor("#131b25"))
    pal.setColor(QPalette.ToolTipText,     QColor("#c8d8e8"))
    pal.setColor(QPalette.Text,            QColor("#c8d8e8"))
    pal.setColor(QPalette.Button,          QColor("#131b25"))
    pal.setColor(QPalette.ButtonText,      QColor("#c8d8e8"))
    pal.setColor(QPalette.Highlight,       QColor("#007fa6"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)
    app.setStyleSheet(QSS)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()