"""Face capture: ARKit-style face values per frame from a zoomed head crop.

Worker stage ``face`` (python -m mocap.face_capture face --project P). Runs in the
separate face runtime (MediaPipe, OpenCV, NumPy; see mocap/face_runtime.py), on the
remote GPU host like the other model stages.

On a full-body plate the face is ~60-120 px wide, too small for MediaPipe's face
detector. Each frame is therefore cropped around the head, upscaled and read by
MediaPipe Face Landmarker (adapted from mixamo-llm-mocap's detect_face.py):

  crop placement  COCO nose/eyes/ears from the model's 2D keypoints when present
                  (ExMR: exmr_keypoints.pt), else the face found in the previous
                  frame, else the top of the tracked person box.
  output          P/face.npz:
                    names     (51,)       ARKit names, as MediaPipe orders them
                    raw       (T, 51)     scores per frame, 0 where no face was read
                    blend     (T, 51)     gap-filled and lightly smoothed scores
                    found     (T,)        1 where a face was read
                    weight    (T,)        1 where blend holds a face, ramping to 0 (neutral) over long gaps
                    landmarks (T, 478, 3) image px (x, y) and relative depth, float16
                    crop      (T, 3)      crop centre x, y and half size, px
                    fps
                  P/face_receipt.json: input identity, so unchanged inputs are reused.

"Left" channels are the performer's own left, as in ARKit.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

VERSION = 1
CROP_PX = 384
GAP_FILL_SECONDS = 0.25  # Shorter dropouts are interpolated; longer ones fade to neutral.
RAMP_SECONDS = 0.15
MODEL = 'face/face_landmarker.task'


def emit(kind, **value):
    print('@@MOCAP@@' + json.dumps(dict(type=kind, **value)), flush=True)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_stat(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns]


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


# ------------------------------------------------------------------ crop placement

def load_tensor(path) -> np.ndarray:
    """Read a tensor saved with torch.save without importing torch.

    Restricted unpickler: only tensor storage and the tensor rebuild call are allowed.
    """
    dtypes = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16,
              'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
              'ByteStorage': np.uint8, 'BoolStorage': np.bool_}
    with zipfile.ZipFile(path) as archive:
        prefix = archive.namelist()[0].split('/')[0]

        def rebuild(storage, offset, size, stride, *_):
            item = storage.itemsize
            return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                                   strides=tuple(s * item for s in stride)).copy()

        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                    return rebuild
                if module == 'torch' and name in dtypes:
                    return name
                if module == 'collections' and name == 'OrderedDict':
                    return collections.OrderedDict
                raise pickle.UnpicklingError(f'Unexpected object in tensor file: {module}.{name}')

            def persistent_load(self, pid):
                _, storage, key, _location, _count = pid
                return np.frombuffer(archive.read(f'{prefix}/data/{key}'), dtype=dtypes[storage])

        with archive.open(f'{prefix}/data.pkl') as data:
            return Reader(data).load()


def smooth_track(values: np.ndarray, valid: np.ndarray, sigma=1.5) -> np.ndarray:
    good = np.flatnonzero(valid)
    if len(good) < 3:
        return values
    out = values.copy()
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / sigma) ** 2)
    kernel /= kernel.sum()
    for column in range(values.shape[1]):
        filled = np.interp(np.arange(len(values)), good, values[good, column])
        smoothed = np.convolve(np.pad(filled, 4, mode='edge'), kernel, mode='valid')
        out[good, column] = smoothed[good]
    return out


def keypoint_boxes(kp: np.ndarray, min_conf=0.3) -> np.ndarray:
    """(T, 3) centre and half size from COCO nose/eyes/ears; NaN where the face is not seen."""
    face = kp[:, :5]
    ok = (face[:, :3, 2] >= min_conf).all(1)
    cx = face[:, :3, 0].mean(1)
    cy = face[:, :3, 1].mean(1)
    span = np.maximum(np.linalg.norm(face[:, 1, :2] - face[:, 2, :2], axis=1),
                      0.6 * np.linalg.norm(face[:, 3, :2] - face[:, 4, :2], axis=1))
    boxes = np.stack([cx, cy, np.maximum(2.2 * span, 48.0)], 1)
    boxes = smooth_track(boxes, ok)
    boxes[~ok] = np.nan
    return boxes


def tracking_boxes(boxes: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Head guess from a person box: top-centre, about one head tall."""
    x0, y0, x1, y1 = boxes.T
    height = np.maximum(y1 - y0, 1.0)
    out = np.stack([(x0 + x1) / 2, y0 + 0.08 * height, np.maximum(0.12 * height, 48.0)], 1)
    out = smooth_track(out, observed)
    out[~observed] = np.nan
    return out


def crop_sources(project: Path, frames: int):
    sources = {}
    keypoints = project / 'exmr_keypoints.pt'
    if keypoints.is_file():
        kp = np.asarray(load_tensor(keypoints), dtype=np.float64)
        if kp.ndim == 3 and kp.shape[0] == frames and kp.shape[1] >= 5:
            sources['keypoints'] = (keypoints, keypoint_boxes(kp))
    tracking = project / 'tracking.npz'
    if tracking.is_file():
        data = np.load(tracking)
        if len(data['boxes']) == frames:
            observed = data['observed'].astype(bool) if 'observed' in data.files else np.ones(frames, bool)
            sources['tracking'] = (tracking, tracking_boxes(data['boxes'].astype(np.float64), observed))
    if not sources:
        raise ValueError('Face capture needs a tracked person: run Reconstruct 3D (or Track) first.')
    return sources


def crop(image: np.ndarray, cx: float, cy: float, half: float) -> np.ndarray:
    import cv2
    x0, y0, size = int(round(cx - half)), int(round(cy - half)), max(2, int(round(2 * half)))
    height, width = image.shape[:2]
    pad = max(0, -x0, -y0, x0 + size - width, y0 + size - height)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    return cv2.resize(image[y0:y0 + size, x0:x0 + size], (CROP_PX, CROP_PX), interpolation=cv2.INTER_CUBIC)


# ------------------------------------------------------------------ capture

def landmark_box(points: np.ndarray) -> tuple[float, float, float]:
    """Next frame's crop from this frame's landmarks: face centre, 1.25x its larger extent."""
    low, high = points[:, :2].min(0), points[:, :2].max(0)
    return float((low[0] + high[0]) / 2), float((low[1] + high[1]) / 2), float(0.62 * (high - low).max() * 2.0)


class NoFace(ValueError):
    """The face was never readable; the motion is still valid."""


def capture(video: Path, sources: dict, frames: int, model: Path):
    # MediaPipe's C++ start-up messages would otherwise fill the Console.
    os.environ.setdefault('GLOG_minloglevel', '2')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
    reader = cv2.VideoCapture(str(video))
    names = None
    raw = np.zeros((frames, 51), np.float32)
    landmarks = np.zeros((frames, 478, 3), np.float16)
    found = np.zeros(frames, np.uint8)
    crops = np.full((frames, 3), np.nan, np.float32)
    used = collections.Counter()
    previous, previous_t = None, -99
    started = last = time.monotonic()
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for t in range(frames):
            ok, bgr = reader.read()
            if not ok:
                raise ValueError(f'The project video ended at frame {t} of {frames}.')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            candidates = []
            if 'keypoints' in sources and np.isfinite(sources['keypoints'][1][t]).all():
                candidates.append(('keypoints', sources['keypoints'][1][t]))
            if previous is not None and t - previous_t <= 3:
                candidates.append(('previous', np.array(previous)))
            if 'tracking' in sources and np.isfinite(sources['tracking'][1][t]).all():
                box = sources['tracking'][1][t]
                candidates += [('tracking', box), ('tracking', box * [1, 1, 1.8])]
            for label, (cx, cy, half) in candidates:
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                    data=np.ascontiguousarray(crop(rgb, cx, cy, half))))
                if not result.face_landmarks or not result.face_blendshapes:
                    continue
                categories = [c for c in result.face_blendshapes[0] if c.category_name != '_neutral']
                names = names or [c.category_name for c in categories]
                raw[t] = [c.score for c in categories]
                points = np.array([(cx - half + p.x * 2 * half, cy - half + p.y * 2 * half, p.z * 2 * half)
                                   for p in result.face_landmarks[0][:478]])
                landmarks[t] = points
                found[t] = 1
                crops[t] = (cx, cy, half)
                previous, previous_t = landmark_box(points), t
                used[label] += 1
                break
            if time.monotonic() - last > 1.0 or t == frames - 1:
                emit('progress', stage='face', current=t + 1, total=frames, message='Reading the face')
                last = time.monotonic()
    reader.release()
    if names is None:
        raise NoFace('No face was found in this clip. Face capture needs the face visible to the camera.')
    return dict(names=np.array(names), raw=raw, landmarks=landmarks, found=found, crop=crops,
                crop_sources=dict(used), seconds=time.monotonic() - started)


def clean(raw: np.ndarray, found: np.ndarray, fps: float):
    """Fill short dropouts, fade long ones to neutral, and smooth lightly (blinks survive)."""
    frames = len(raw)
    good = np.flatnonzero(found)
    gap = max(1, int(round(GAP_FILL_SECONDS * fps)))
    ramp = max(1, int(round(RAMP_SECONDS * fps)))
    blend = raw.astype(np.float64).copy()
    weight = found.astype(np.float64).copy()
    for a, b in zip(good[:-1], good[1:]):
        if 1 < b - a <= gap + 1:
            s = (np.arange(a + 1, b) - a) / (b - a)
            blend[a + 1:b] = (1 - s)[:, None] * raw[a] + s[:, None] * raw[b]
            weight[a + 1:b] = 1
    # Long gaps: the last/next seen expression fades to neutral over the ramp.
    index = np.arange(frames)
    valid = weight > 0
    before = np.maximum.accumulate(np.where(valid, index, -10 ** 9))
    after = np.minimum.accumulate(np.where(valid, index, 10 ** 9)[::-1])[::-1]
    nearest = np.where(index - before <= after - index, before, after)
    fade = np.clip(1 - np.minimum(index - before, after - index) / ramp, 0, 1)
    missing = ~valid
    blend[missing] = blend[nearest[missing]]
    weight = np.where(missing, fade, weight)
    blend *= weight[:, None]
    # Median-of-3 removes single-frame spikes; a narrow Gaussian keeps 3-4 frame blinks.
    padded = np.pad(blend, ((1, 1), (0, 0)), mode='edge')
    blend = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    sigma = 0.6 * fps / 30.0
    kernel = np.exp(-0.5 * (np.arange(-3, 4) / max(sigma, 1e-3)) ** 2)
    kernel /= kernel.sum()
    blend = np.stack([np.convolve(np.pad(c, 3, mode='edge'), kernel, mode='valid') for c in blend.T], 1)
    return np.clip(blend, 0, 1).astype(np.float32), weight.astype(np.float32)


# ------------------------------------------------------------------ stage

def identity(project: Path, model: Path, sources: dict):
    import mediapipe
    return dict(version=VERSION, input=file_stat(project / 'input.mp4'),
                sources={k: file_stat(path) for k, (path, _) in sources.items()},
                model=file_stat(model), code=sha256(Path(__file__)), mediapipe=mediapipe.__version__)


def run(project: Path) -> int:
    from .paths import MODELS
    project = project.resolve()
    started = time.monotonic()
    state_path = project / 'project.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    frames = int(state.get('frames') or 0)
    fps = float(state.get('fps') or 30.0)
    if frames < 2:
        raise ValueError('The project has no prepared frames.')
    model = MODELS / MODEL
    if not model.is_file():
        raise FileNotFoundError('Face model missing: download "Face / MediaPipe Face Landmarker" in Model Manager.')
    sources = crop_sources(project, frames)
    ident = identity(project, model, sources)
    output, receipt = project / 'face.npz', project / 'face_receipt.json'
    try:
        prior = json.loads(receipt.read_text(encoding='utf-8'))
        if prior['identity'] == ident and prior['output'] == file_stat(output):
            emit('complete', stage='face', output=str(output), cached=True, elapsed=0,
                 found_ratio=prior.get('found_ratio'))
            return 0
    except (OSError, ValueError, KeyError):
        pass
    emit('info', message=f'Face capture: {frames} frames, crops from {", ".join(sources)}.')
    try:
        result = capture(project / 'input.mp4', sources, frames, model)
    except NoFace as warning:
        output.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
        emit('info', message=f'{warning} The motion is unaffected; exports carry no face animation.')
        emit('complete', stage='face', output=None, cached=False, elapsed=time.monotonic() - started, found_ratio=0.0)
        return 0
    blend, weight = clean(result['raw'], result['found'], fps)
    temp = project / f'.face.{os.getpid()}.npz'
    with open(temp, 'wb') as handle:
        np.savez_compressed(handle, names=result['names'], raw=result['raw'], blend=blend, weight=weight,
                            found=result['found'], landmarks=result['landmarks'], crop=result['crop'],
                            fps=np.float32(fps), version=np.int32(VERSION))
    temp.replace(output)
    ratio = float(result['found'].mean())
    elapsed = time.monotonic() - started
    atomic_json(receipt, dict(identity=ident, output=file_stat(output), found_ratio=ratio,
                              crop_sources=result['crop_sources'], elapsed=elapsed))
    state = json.loads(state_path.read_text(encoding='utf-8'))
    state.setdefault('stages', {})['face'] = {'status': 'complete', 'elapsed': elapsed}
    atomic_json(state_path, state)
    emit('info', message=f'Face read on {ratio * 100:.0f}% of frames '
                         f'({", ".join(f"{k} {v}" for k, v in result["crop_sources"].items())}); '
                         f'{frames / max(result["seconds"], 1e-6):.0f} frames/s.')
    emit('complete', stage='face', output=str(output), cached=False, elapsed=elapsed, found_ratio=ratio)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m mocap.face_capture')
    parser.add_argument('stage', choices=['face'])
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--result')  # Accepted for queue compatibility; capture does not depend on it.
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    try:
        return run(args.project)
    except Exception as error:  # Reported to the GUI like other worker failures.
        emit('failed', stage='face', message=str(error))
        try:
            path = args.project / 'project.json'
            state = json.loads(path.read_text(encoding='utf-8'))
            state.setdefault('stages', {})['face'] = {'status': 'failed', 'error': str(error)}
            atomic_json(path, state)
        except (OSError, ValueError):
            pass
        raise


if __name__ == '__main__':
    sys.exit(main())
