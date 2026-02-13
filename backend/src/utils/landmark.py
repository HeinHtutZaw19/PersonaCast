#!/usr/bin/env python3
"""
Wide-shot talking detector using YOLO face detection + MediaPipe FaceMesh.

Pipeline:
  YOLO (faces) -> crop per face -> FaceMesh -> mouth-motion talking -> draw mesh green/gray

Install:
  pip install ultralytics mediapipe opencv-python numpy

Usage:
  python talk_from_video_yolo.py input.mp4 output.mp4
  python talk_from_video_yolo.py input.mp4 output.mp4 --fd-only
  python talk_from_video_yolo.py input.mp4 output.mp4 --yolo-model yolov8n-face.pt
"""

import sys
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp

# Ultralytics YOLO
from ultralytics import YOLO


# -------------------- Lip indices (MediaPipe FaceMesh) --------------------
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LIP_UPPER = 13
LIP_LOWER = 14


# -------------------- Utils --------------------
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def pad_bbox(b, pad, w, h):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    x1 = x1 - pad * bw
    y1 = y1 - pad * bh
    x2 = x2 + pad * bw
    y2 = y2 + pad * bh
    return clamp_bbox(x1, y1, x2, y2, w, h)


def lm_to_xy(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def mouth_ratio(lms, w, h) -> float:
    p_left = lm_to_xy(lms, MOUTH_LEFT, w, h)
    p_right = lm_to_xy(lms, MOUTH_RIGHT, w, h)
    p_upper = lm_to_xy(lms, LIP_UPPER, w, h)
    p_lower = lm_to_xy(lms, LIP_LOWER, w, h)
    mouth_width = np.linalg.norm(p_right - p_left) + 1e-6
    mouth_open = np.linalg.norm(p_lower - p_upper)
    return float(mouth_open / mouth_width)


# -------------------- Tracking + talking --------------------
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    ratio_ema: Optional[float] = None
    ratio_hist: deque = field(default_factory=lambda: deque(maxlen=12))
    talking_hold: int = 0
    last_seen_frame: int = 0


class TalkingDetector:
    def __init__(
        self,
        window_size=12,
        ema_alpha=0.25,
        motion_threshold=0.015,
        min_open_gate=0.05,
        cooldown_frames=6,
        max_missed=20,
        match_iou_thresh=0.25,
    ):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.motion_threshold = motion_threshold
        self.min_open_gate = min_open_gate
        self.cooldown_frames = cooldown_frames
        self.max_missed = max_missed
        self.match_iou_thresh = match_iou_thresh
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def _new_track(self, bbox, frame_idx):
        t = Track(
            track_id=self.next_id,
            bbox=bbox,
            ratio_hist=deque(maxlen=self.window_size),
            last_seen_frame=frame_idx,
        )
        self.tracks[self.next_id] = t
        self.next_id += 1
        return t

    def _cleanup(self, frame_idx):
        dead = [tid for tid, t in self.tracks.items() if frame_idx - t.last_seen_frame > self.max_missed]
        for tid in dead:
            del self.tracks[tid]

    def match_and_update(self, detections, frame_idx):
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]
        used = set()
        out = []

        for det_bbox, ratio, payload in detections:
            best_tid, best_i = None, 0.0
            for tid, tb in zip(track_ids, track_bboxes):
                if tid in used:
                    continue
                s = iou(tb, det_bbox)
                if s > best_i:
                    best_i, best_tid = s, tid

            if best_tid is None or best_i < self.match_iou_thresh:
                t = self._new_track(det_bbox, frame_idx)
            else:
                t = self.tracks[best_tid]
                used.add(best_tid)
                t.bbox = det_bbox
                t.last_seen_frame = frame_idx

            if t.ratio_ema is None:
                t.ratio_ema = ratio
            else:
                t.ratio_ema = self.ema_alpha * ratio + (1 - self.ema_alpha) * t.ratio_ema

            t.ratio_hist.append(t.ratio_ema)
            motion = 0.0
            if len(t.ratio_hist) >= 4:
                motion = float(max(t.ratio_hist) - min(t.ratio_hist))

            talking = (t.ratio_ema > self.min_open_gate) and (motion > self.motion_threshold)

            if talking:
                t.talking_hold = self.cooldown_frames
            elif t.talking_hold > 0:
                talking = True
                t.talking_hold -= 1

            out.append((t, float(t.ratio_ema), motion, talking, payload))

        self._cleanup(frame_idx)
        return out


# -------------------- YOLO Face detector wrapper --------------------
class YoloFaceDetector:
    """
    Ultralytics YOLO face detector wrapper.

    You must supply a face model file.
    Example:
      - yolov8n-face.pt
      - yolov5n-face.pt
      - a custom trained face detector
    """
    def __init__(self, model_path: str, conf: float = 0.25, iou_thr: float = 0.45, imgsz: int = 640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou_thr = iou_thr
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        """
        Returns list of (x1,y1,x2,y2,conf) in pixel coords on the input frame.
        """
        # Ultralytics expects BGR np array fine; it handles preprocessing internally.
        results = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou_thr, imgsz=self.imgsz, verbose=False)
        if not results:
            return []

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        b = r.boxes
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()

        out = []
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            out.append((int(x1), int(y1), int(x2), int(y2), float(c)))
        return out


# -------------------- Main --------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python talk_from_video_yolo.py input.mp4 [output.mp4] [--fd-only] [--yolo-model MODEL.pt]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = "output_talking.mp4"
    fd_only = False
    yolo_model = None

    # crude arg parse
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--fd-only":
            fd_only = True
        elif a == "--yolo-model" and i + 1 < len(args):
            yolo_model = args[i + 1]
        elif not a.startswith("--") and out_path == "output_talking.mp4":
            out_path = a

    if yolo_model is None:
        raise SystemExit(
            "Missing --yolo-model. Example:\n"
            "  python talk_from_video_yolo.py input.mp4 out.mp4 --yolo-model yolov8n-face.pt\n"
        )

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # FaceMesh on crops
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    detector = TalkingDetector()
    yolo = YoloFaceDetector(yolo_model, conf=0.25, iou_thr=0.45, imgsz=640)

    # tuning
    PAD = 0.25
    CROP_UPSCALE = 2.0
    MIN_FACE_PX = 28

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # 1) YOLO face detect
        boxes = yolo.detect(frame_bgr)  # (x1,y1,x2,y2,conf)

        face_bboxes = []
        for x1, y1, x2, y2, c in boxes:
            x1, y1, x2, y2 = pad_bbox((x1, y1, x2, y2), PAD, W, H)
            if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
                continue
            face_bboxes.append((x1, y1, x2, y2, c))

        # fd-only mode
        if fd_only:
            for (x1, y1, x2, y2, c) in face_bboxes:
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame_bgr, f"Face {c:.2f}", (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            out.write(frame_bgr)
            frame_idx += 1
            continue

        # 2) FaceMesh per bbox
        detections = []
        for (x1, y1, x2, y2, c) in face_bboxes:
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            if CROP_UPSCALE != 1.0:
                crop_bgr_u = cv2.resize(crop_bgr, (0, 0), fx=CROP_UPSCALE, fy=CROP_UPSCALE, interpolation=cv2.INTER_LINEAR)
            else:
                crop_bgr_u = crop_bgr

            ch, cw = crop_bgr_u.shape[:2]
            crop_rgb = cv2.cvtColor(crop_bgr_u, cv2.COLOR_BGR2RGB)

            mesh_res = face_mesh.process(crop_rgb)
            if not mesh_res.multi_face_landmarks:
                continue

            fl = mesh_res.multi_face_landmarks[0]
            r = mouth_ratio(fl.landmark, cw, ch)

            # payload includes landmarks + crop geometry
            detections.append(((x1, y1, x2, y2), r, (fl, (x1, y1, x2, y2), (cw, ch), CROP_UPSCALE, c)))

        tracked = detector.match_and_update(detections, frame_idx)

        # 3) Draw
        for t, ratio_ema, motion, talking, payload in tracked:
            fl, (x1, y1, x2, y2), (cw, ch), crop_up, conf = payload
            color = (0, 255, 0) if talking else (180, 180, 180)
            status = "Talking" if talking else "Not talking"

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            # draw mesh lines on full frame by mapping crop coords back
            # since crop was upscaled by crop_up, divide by crop_up to go back to original crop pixels
            for a, b in mp_face_mesh.FACEMESH_TESSELATION:
                pa = fl.landmark[a]
                pb = fl.landmark[b]
                ax = int(x1 + (pa.x * cw) / crop_up)
                ay = int(y1 + (pa.y * ch) / crop_up)
                bx = int(x1 + (pb.x * cw) / crop_up)
                by = int(y1 + (pb.y * ch) / crop_up)
                cv2.line(frame_bgr, (ax, ay), (bx, by), color, 1)

            yy = max(20, y1 - 8)
            cv2.putText(frame_bgr, f"ID {t.track_id}: {status} ({conf:.2f})", (x1, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"ratio={ratio_ema:.3f} motion={motion:.3f}", (x1, yy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame_bgr)

        frame_idx += 1
        if frame_idx % 100 == 0:
            if total > 0:
                print(f"Processed {frame_idx}/{total}", end="\r")
            else:
                print(f"Processed {frame_idx}", end="\r")

    cap.release()
    out.release()
    face_mesh.close()
    print(f"\nDone. Wrote: {out_path}")


if __name__ == "__main__":
    main()
