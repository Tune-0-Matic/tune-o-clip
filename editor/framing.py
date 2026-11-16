"""9:16 framing from real face detections + burned-in caption detection. Replaces the old Haar cascade + overlay-edge
webcam guesser, which cropped walls / neon signs as "webcams" and cut heads off.

Faces: OpenCV YuNet (face_detection_yunet_2023mar.onnx, 230 KB, MIT; the detector shortsflow uses). Chosen over
MediaPipe: its published Tasks face model is short-range (selfie distance) and misses small webcam / wide-room faces,
and the mediapipe wheel pulls opencv-contrib-python 5 next to our opencv-python-headless (two packages owning cv2).
Text: PP-OCRv3 detector (text_detection_en_ppocrv3_2023may.onnx, 2.4 MB, Apache-2.0) through cv2.dnn.

Layouts (decide()):
  split_stack  one small, steady face near a frame edge and no other comparable face = a webcam overlay on gameplay.
               cam_box = a crop centred on that face with generous headroom, in the top panel's aspect.
  face_crop    the faces fit in one 9:16 column: a full-height 9:16 crop that smoothly follows them.
  blur_fill    faces too far apart for one column, or no face most of the time: the whole 16:9 over a blurred copy.
"""
import subprocess
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent / "models"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
TEXT_DB = MODELS / "text_detection_en_ppocrv3_2023may.onnx"
SAMPLE_W, FPS = 640, 2.0
MIN_SCORE, MIN_FACE_H = 0.6, 0.035  # smaller "faces" are posters, chat avatars, emotes
CAM_AREA, OVERLAY = 0.25, 2.4  # Case A: the webcam overlay (head + shoulders, ~2.4x the face each way) < 25% of the frame
CAM_SHARE, CAM_STEADY = 0.5, 0.05  # ...in >= 50% of the samples, its centre within 5% of the frame of its median spot
CAM_HEAD = 2.5  # top panel height = 2.5 face heights
HEAD_TOP = 0.9  # the panel starts 0.9 face heights above YuNet's box: it starts at the brows, hair / headsets sit ~0.6 above
COLUMN_FILL = 0.85  # faces fit the 9:16 column when their span is <= 85% of its width (margins for heads / shoulders)
FIT_SHARE, FACE_SHARE = 0.7, 0.5  # a face seen in < half the samples is not worth following


def probe_size(src):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout
    w, h = (int(v) for v in out.strip().split(",")[:2])
    return w, h


def sample_frames(src, fps=FPS, width=SAMPLE_W):
    """-> (times s, [BGR uint8 frames], source aspect w/h): one ffmpeg pass at `fps`, scaled to `width`."""
    sw, sh = probe_size(src)
    h = int(round(width * sh / sw / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"fps={fps},scale={width}:{h}", "-pix_fmt", "bgr24",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    n = len(raw) // (width * h * 3)
    frames = [np.frombuffer(raw, np.uint8, width * h * 3, i * width * h * 3).reshape(h, width, 3) for i in range(n)]
    return [round((i + 0.5) / fps, 2) for i in range(n)], frames, sw / sh


def detect_faces(frames, model=YUNET):
    """Per frame [(x, y, w, h, score)] normalized 0..1, weak / tiny / background faces dropped."""
    import cv2
    out = []
    if not frames:
        return out
    h, w = frames[0].shape[:2]
    det = cv2.FaceDetectorYN.create(str(model), "", (w, h), MIN_SCORE, 0.3, 5000)
    for f in frames:
        _, found = det.detect(f)
        faces = [(float(x) / w, float(y) / h, float(fw) / w, float(fh) / h, float(s))
                 for x, y, fw, fh, *_, s in (found if found is not None else [])]
        faces = [q for q in faces if q[3] >= MIN_FACE_H]
        top = max((q[3] for q in faces), default=0)
        out.append([q for q in faces if q[3] >= 0.45 * top])  # a face much smaller than the biggest one is background
    return out


def cam_box(face, aspect, panel_aspect, top=None, bottom=None):
    """Normalized (x, y, w, h) source crop for the top panel: the face centred across, the head HEAD_TOP face heights
    below the top, inside the frame. top / bottom = the highest face top / lowest chin seen (a streamer leaning up or
    down stays whole; live: Jynxzi's forehead left a median-placed crop). aspect = source w/h, panel = panel w/h."""
    fx, fy, fw, fh = face[:4]
    top = fy if top is None else min(top, fy)
    bottom = fy + fh if bottom is None else max(bottom, fy + fh)
    fy = top
    ch = min(1.0, max(fh * CAM_HEAD, bottom - top + (HEAD_TOP + 0.4) * fh))
    cw = ch * panel_aspect / aspect
    if cw > 1.0:
        ch, cw = ch / cw, 1.0
    x = min(max(0.0, fx + fw / 2 - cw / 2), 1 - cw)
    y = min(max(0.0, fy - HEAD_TOP * fh), 1 - ch)  # the whole head, hair and headset included, never the panel edge
    return tuple(round(float(v), 4) for v in (x, y, cw, ch))


def smooth(xs, k=5):
    """Fill gaps (None) from the nearest known value, then a centred moving average over k samples."""
    known = [i for i, x in enumerate(xs) if x is not None]
    if not known:
        return []
    filled = np.array([xs[min(known, key=lambda j: abs(j - i))] for i in range(len(xs))], dtype=float)
    pad = np.pad(filled, k // 2, mode="edge")
    return list(np.convolve(pad, np.ones(k) / k, mode="valid"))


MAX_ZOOM = 1.35  # the strongest punch-in the director asks for


def zoom_room(faces, panel=(0.0, 0.0, 1.0, 1.0), keep_bottom=None):
    """-> [ax, ay, max zoom] for a panel showing source region `panel` (x, y, w, h). The renderer zooms about the
    anchor (it stays put on screen). The protected box = every face seen (+ hair, chin, a margin) and, when the
    stream has its own captions, their line (keep_bottom, source y, centred). Per axis, the anchor
    lo / (1 - (hi - lo)) lets the zoom reach 1 / (hi - lo) with the box still inside: at that zoom its edges sit
    exactly on the panel's, so below it a head never leaves the frame."""
    px, py, pw, ph = panel
    f = [((x - px) / pw, (y - py) / ph, w / pw, h / ph) for x, y, w, h, *_ in faces]
    if f:
        left, right = np.percentile([q[0] - 0.2 * q[2] for q in f], 5), np.percentile([q[0] + 1.2 * q[2] for q in f], 95)
        top, bottom = np.percentile([q[1] - 0.6 * q[3] for q in f], 5), np.percentile([q[1] + 1.15 * q[3] for q in f], 95)
    else:  # gameplay only: nothing to protect, zoom about the centre
        left = right = 0.5
        top = bottom = 0.5
    if keep_bottom is not None:
        left, right = min(left, (0.25 - px) / pw), max(right, (0.75 - px) / pw)
        bottom = max(bottom, (keep_bottom - py) / ph)
        top = min(top, bottom)
    out, z = [], MAX_ZOOM
    for lo, hi in ((left, right), (top, bottom)):
        # a margin past the panel edge (a webcam hugging the frame edge) can't be lost: the anchor sits on that edge
        # (live: CaseOh's corner cam got no zoom at all)
        lo, hi = max(0.0, lo), min(1.0, hi)
        if hi - lo >= 1:  # the box fills the panel: no zoom at all
            out.append(0.5)
            z = 1.0
            continue
        out.append(float(lo / (1 - (hi - lo))))
        if hi - lo > 1e-6:
            z = min(z, 1 / (hi - lo))
    return [round(out[0], 3), round(out[1], 3), round(max(1.0, float(z)), 3)]


def decide(times, faces, aspect, panel_aspect=1080 / 690, keep_bottom=None):
    """-> {"layout", "cam_box", "track" [(t, centre x 0..1)], "zoom" [ax, ay, max] for that layout's zoomable panel,
    "zoom_full" (the same for the whole frame: blur_fill), "why"} from per-sample detections (pure: tested)."""
    res = _decide(times, faces, aspect, panel_aspect, keep_bottom)
    every = [q for f in faces for q in f]
    res["zoom_full"] = zoom_room(every, keep_bottom=keep_bottom)
    res.setdefault("zoom", res["zoom_full"])
    return res


def _decide(times, faces, aspect, panel_aspect, keep_bottom):
    n = len(faces)
    with_face = [f for f in faces if f]
    if not n or len(with_face) < FACE_SHARE * n:
        return {"layout": "blur_fill", "cam_box": None, "track": [], "why": f"a face in {len(with_face)}/{n} samples"}
    # Case A: one small face near an edge, steady, alone
    solo = [f[0] for f in faces if len(f) == 1 and OVERLAY ** 2 * f[0][2] * f[0][3] < CAM_AREA
            and not (0.3 < f[0][0] + f[0][2] / 2 < 0.7 and 0.3 < f[0][1] + f[0][3] / 2 < 0.7)]
    if len(solo) >= CAM_SHARE * n:
        cx, cy = np.median([q[0] + q[2] / 2 for q in solo]), np.median([q[1] + q[3] / 2 for q in solo])
        steady = [q for q in solo if abs(q[0] + q[2] / 2 - cx) < CAM_STEADY and abs(q[1] + q[3] / 2 - cy) < CAM_STEADY]
        if len(steady) >= CAM_SHARE * n:
            med = tuple(float(np.median([q[i] for q in steady])) for i in range(4))
            top = float(np.percentile([q[1] for q in solo], 5))  # where the head goes when they lean, not just its median
            bottom = float(np.percentile([q[1] + q[3] for q in solo], 95))
            box = cam_box(med, aspect, panel_aspect, top, bottom)
            return {"layout": "split_stack", "cam_box": box, "track": [], "zoom": zoom_room(solo, box),
                    "why": f"webcam face at ({cx:.2f}, {cy:.2f}) in {len(steady)}/{n} samples"}
    # Case B: one 9:16 column following the faces, unless they stand too far apart
    column = (9 / 16) / aspect
    spans = [(min(q[0] for q in f), max(q[0] + q[2] for q in f)) for f in with_face]
    fit = sum(b - a <= COLUMN_FILL * column for a, b in spans)
    if column >= 1 or fit < FIT_SHARE * len(with_face):
        return {"layout": "blur_fill", "cam_box": None, "track": [],
                "why": f"faces fit one 9:16 column in {fit}/{len(with_face)} samples"}
    centres = []
    for f in faces:
        if not f:
            centres.append(None)
            continue
        a, b = min(q[0] for q in f), max(q[0] + q[2] for q in f)
        if b - a > COLUMN_FILL * column:  # too wide this time: follow the biggest face
            q = max(f, key=lambda q: q[3])
            a, b = q[0], q[0] + q[2]
        centres.append((a + b) / 2)
    xs = [float(min(max(x, column / 2), 1 - column / 2)) for x in smooth(centres)]
    track = [(t, round(x, 4)) for t, x in zip(times, xs)][::2]  # 1 per s
    in_column = [((q[0] - x + column / 2) / column, q[1], q[2] / column, q[3]) for f, x in zip(faces, xs) for q in f]
    return {"layout": "face_crop", "cam_box": None, "track": track, "zoom": zoom_room(in_column),
            "why": f"faces fit one 9:16 column in {fit}/{len(with_face)} samples"}


# --- burned-in captions --------------------------------------------------------------------------------------------

def caption_boxes(frames, model=TEXT_DB):
    """Per frame [(cx, cy, w, h)] normalized text boxes (PP-OCRv3 detector)."""
    import cv2
    det = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(model)))
    det.setBinaryThreshold(0.3).setPolygonThreshold(0.5).setMaxCandidates(200).setUnclipRatio(2.0)
    det.setInputParams(1.0 / 255.0, (640, 384), (122.67891434, 116.66876762, 104.00698793), True)
    out = []
    for f in frames:
        h, w = f.shape[:2]
        polys, _ = det.detect(f)
        boxes = []
        for p in polys:
            p = np.asarray(p, dtype=float)
            x0, y0, x1, y1 = p[:, 0].min() / w, p[:, 1].min() / h, p[:, 0].max() / w, p[:, 1].max() / h
            boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
        out.append(boxes)
    return out


def has_source_captions(boxes):
    """Burned-in captions (a streamer's live-caption overlay, a re-upload's subtitles): during speech a wide text line
    sits low and centred in most samples, and its width keeps changing (new words). A game HUD line stays the same
    width; chat / alerts sit at the sides. boxes = caption_boxes() of frames taken while someone is talking."""
    widths = []
    for frame in boxes:
        line = [b for b in frame if b[1] >= 0.55 and 0.25 <= b[0] <= 0.75 and b[2] >= 0.12 and b[3] <= 0.15]
        if line:
            widths.append(max(b[2] for b in line))
    if len(boxes) < 4 or len(widths) < 0.5 * len(boxes):
        return False
    return float(np.std(widths) / np.mean(widths)) >= 0.08


def analyze(src, words=(), cam_override=None, panel_aspect=1080 / 690):
    """Everything framing needs for one clip: decide() + "source_captions". cam_override (a hand-set webcam box)
    forces split_stack with that box."""
    times, frames, aspect = sample_frames(src)
    talking = [f for t, f in zip(times, frames) if any(w["start"] <= t <= w["end"] + 0.3 for w in words)]
    pick = talking[::max(1, len(talking) // 20)][:20]  # 12 missed a live-caption stream once (2 of 12 frames between lines)
    captions = has_source_captions(caption_boxes(pick)) if pick else False
    faces = detect_faces(frames)
    res = decide(times, faces, aspect, panel_aspect, keep_bottom=0.97 if captions else None)  # zooms keep their line whole
    if cam_override is not None:
        res.update(layout="split_stack", cam_box=tuple(cam_override), track=[], zoom=zoom_room([], tuple(cam_override)),
                   why=f"webcam box set by hand {tuple(cam_override)}")
    res["source_captions"] = captions
    return res
