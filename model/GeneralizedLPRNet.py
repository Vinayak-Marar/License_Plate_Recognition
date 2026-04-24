# class GeneralizedLPRNet(nn.Module):
#     def __init__(self, num_classes, hidden_size=256):
#         super().__init__()

#         # Original LPRNet feature extractor — unchanged
#         self.backbone = nn.Sequential(
#             nn.Conv2d(3, 64, 3, stride=1, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(),
#             nn.MaxPool2d(3, stride=1),
#             nn.Conv2d(64, 128, 3, stride=1, padding=1),  nn.BatchNorm2d(128), nn.ReLU(),
#             nn.MaxPool2d(3, stride=2, padding=1),
#             nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
#             nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
#             nn.MaxPool2d(3, stride=2, padding=1),
#             nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
#             nn.MaxPool2d(3, stride=1),
#             nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
#         )
#         # output: (B, 512, H', W') — W' is your sequence dimension

#         # NEW: BiGRU for sequence modeling — this is the generalization upgrade
#         self.gru = nn.GRU(
#             input_size=512,
#             hidden_size=hidden_size,
#             num_layers=2,
#             bidirectional=True,
#             dropout=0.3,
#             batch_first=True
#         )

#         self.classifier = nn.Linear(hidden_size * 2, num_classes)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         f = self.backbone(x)             # (B, 512, H', W')
#         f = f.mean(dim=2)                # pool H' → (B, 512, W')
#         f = f.permute(0, 2, 1)           # (B, W', 512)
#         f = self.dropout(f)
#         out, _ = self.gru(f)             # (B, W', hidden*2)
#         out = self.classifier(out)       # (B, W', num_classes)
#         return out.permute(1, 0, 2)      # (T, B, num_classes) for CTC

# Summary
# Original LPRNet          →  CNN only, fixed-length bias, no sequence model
# LPRNet + BiGRU (this)   →  plate-optimized features + proper sequence modeling
# MobileNet + BiGRU        →  overkill, general-purpose features, heavier
# Stick with LPRNet backbone + add BiGRU. MobileNet would only make sense if you were also doing detection or multi-task learning. For pure recognition, LPRNet backbone is the right call.
# Want me to write the full training loop for this now?lets stick with lprnback bone in generalized lprnet but also make sure it have batch inference because i want to convert then into tensorrt12:21Architected TensorRT-compatible LPRNet with batch inference supportArchitected TensorRT-compatible LPRNet with batch inference supportGood call — TensorRT has specific requirements. Let me write the full implementation with that in mind.

# TensorRT Constraints to Design Around

# No dynamic control flow in forward() — no if/else on tensor shapes
# GRU must export cleanly via ONNX — it does, but needs fixed input size
# CTC decoding stays outside TensorRT — TRT handles inference only
# Dynamic batch size — handled via ONNX dynamic axes
# No .item() calls inside forward


# Full Implementation
# pythonimport torch



import torch.nn as nn
# import torch.onnx
import numpy as np

# ── LPRNet Backbone (original, unchanged) ─────────────────────────────────────

class LPRNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1),

            # Block 2
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            # Block 3
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            # Block 4
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1),

            # Block 5
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # output: (B, 512, H', W')

    def forward(self, x):
        return self.features(x)


# ── Generalized LPRNet (TensorRT Ready) ───────────────────────────────────────

class GeneralizedLPRNet(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_gru_layers=2, dropout=0.3):
        super().__init__()

        self.backbone = LPRNetBackbone()   # (B, 512, H', W')

        # BiGRU — exports cleanly to ONNX/TRT
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            batch_first=True                # (B, T, C) — stable for ONNX export
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 3, H, W)
        f = self.backbone(x)             # (B, 512, H', W')
        f = f.mean(dim=2)                # (B, 512, W')  — pool height, no dynamic shape
        f = f.permute(0, 2, 1)           # (B, W', 512)
        f = self.dropout(f)
        out, _ = self.gru(f)             # (B, W', hidden*2) — ignore hidden state
        out = self.classifier(out)       # (B, W', num_classes)
        out = out.permute(1, 0, 2)       # (T, B, num_classes) — CTC format
        return out                        # raw logits — softmax outside TRT

