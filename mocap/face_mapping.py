"""Map captured face values (mocap/face_capture.py) onto a character's face targets.

NumPy only: used inside the character Blender runtime and testable outside it.

Targets, in order of preference per channel:
  shape keys  ARKit names (any case/separators) or common aliases such as Blink_L
  bones       Jaw (jawOpen), Left/RightEye (eyeLook*), eyelid bones (eyeBlink*) - used
              only for channels no shape key covers, so a face is never driven twice.

Scores are remapped per channel. MediaPipe reads a fully applied shape as about 0.6-0.78
(measured by rendering the MakeHuman test character at known weights; see docs/FACE.md),
and it carries per-performer biases: bangs over the brows read as raised brows, a head
turned toward the camera reads as a constant side glance. So each channel is measured from
the performer's own resting level (5th percentile of the clip) and scaled by the measured
span from neutral to full. Eye look is centred per eye and axis on the clip's median, so
the character keeps its gaze straight on average and still follows glances.
"""
from __future__ import annotations
import re
import numpy as np

# MediaPipe span from a neutral to a fully applied shape, by channel prefix.
SPAN = (('eyeBlink', 0.57), ('eyeLook', 0.67), ('eyeSquint', 0.55), ('eyeWide', 0.55),
        ('jawOpen', 0.73), ('', 0.70))
REST_PERCENTILE = 5
GAZE_AXES = (('eyeLookOut', 'eyeLookIn'), ('eyeLookUp', 'eyeLookDown'))

ALIASES = {
    'blinkl': 'eyeBlinkLeft', 'blinkr': 'eyeBlinkRight', 'blinkleft': 'eyeBlinkLeft', 'blinkright': 'eyeBlinkRight',
    'eyeblinkl': 'eyeBlinkLeft', 'eyeblinkr': 'eyeBlinkRight', 'eyeclosedl': 'eyeBlinkLeft', 'eyeclosedr': 'eyeBlinkRight',
    'eyecloseleft': 'eyeBlinkLeft', 'eyecloseright': 'eyeBlinkRight',
    'mouthopen': 'jawOpen', 'openmouth': 'jawOpen',
    'smilel': 'mouthSmileLeft', 'smiler': 'mouthSmileRight', 'smileleft': 'mouthSmileLeft', 'smileright': 'mouthSmileRight',
    # VRoid Studio exports
    'fcleyeclosel': 'eyeBlinkLeft', 'fcleyecloser': 'eyeBlinkRight', 'fclmtha': 'jawOpen',
}
# Eye-look shapes rotate the eye geometry itself; jaw/eye bones replace them when absent.
BONE_CHANNELS = {'jaw': ('jawOpen',), 'eye': ('eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeLookDownLeft',
                                             'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeLookDownRight'),
                 'lid': ('eyeBlinkLeft', 'eyeBlinkRight')}
JAW_DEGREES = 22.0
EYE_YAW_DEGREES = 30.0
EYE_PITCH_DEGREES = 20.0
UPPER_LID_DEGREES = 30.0
LOWER_LID_DEGREES = 8.0


def key(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower().split(':')[-1])


def channel_for(name: str, channels) -> str | None:
    """ARKit channel driven by a shape key name, or None."""
    lookup = {key(c): c for c in channels}
    k = key(name)
    if k in lookup:
        return lookup[k]
    alias = ALIASES.get(k)
    return alias if alias in channels else None


def span_of(name: str) -> float:
    return next(v for prefix, v in SPAN if name.startswith(prefix))


def remap(values: np.ndarray, names) -> np.ndarray:
    """(T, C) scores -> (T, C) shape weights in [0, 1], relative to the performer's rest."""
    values = np.asarray(values, dtype=np.float64)
    index = {n: i for i, n in enumerate(names)}
    out = np.zeros_like(values)
    done = set()
    for side in ('Left', 'Right'):
        for positive, negative in GAZE_AXES:
            a, b = index.get(positive + side), index.get(negative + side)
            if a is None or b is None:
                continue
            axis = values[:, a] - values[:, b]
            axis = axis - np.median(axis)
            span = span_of(positive)
            out[:, a] = np.clip(axis / span, 0.0, 1.0)
            out[:, b] = np.clip(-axis / span, 0.0, 1.0)
            done.update((a, b))
    for c, name in enumerate(names):
        if c in done:
            continue
        rest = float(np.percentile(values[:, c], REST_PERCENTILE))
        out[:, c] = np.clip((values[:, c] - rest) / span_of(name), 0.0, 1.0)
    return out


UPPER = re.compile(r'up|upper|top')
LOWER = re.compile(r'dwn|down|lower|bottom|bot')
NOT_EYE = ('squash', 'brow', 'lash', 'socket', 'target', 'aim', 'look', 'control', 'ctrl')


def _is_end(k: str) -> bool:
    return 'end' in k or k.endswith(('tip', 'nub', 'null'))


def _lid(k: str) -> str | None:
    probe = k.replace('pupil', '')
    if 'lid' in probe or 'orbicularis' in probe or ('eye' in probe and (UPPER.search(probe.replace('eye', ' ')) or LOWER.search(probe))):
        return 'lower' if LOWER.search(probe) or 'bottom' in probe else 'upper'
    return None


def _named_side(name: str) -> str | None:
    k = key(name)
    if 'left' in k:
        return 'Left'
    if 'right' in k:
        return 'Right'
    m = re.search(r'(^|[:_.\- ])([lr])([_.\- ]|$)', name.split(':')[-1].lower())
    return {'l': 'Left', 'r': 'Right'}[m.group(2)] if m else None


def _sided(names, lateral):
    """{side: bone}: by position across the character when known (names can be wrong), else by name."""
    out = {}
    for name in sorted(names, key=lambda n: len(key(n))):
        if lateral and name in lateral and abs(lateral[name]) > 1e-4:
            side = 'Left' if lateral[name] > 0 else 'Right'
        else:
            side = _named_side(name)
        if side:
            out.setdefault(side, name)
    return out


def face_bones(bone_names, lateral=None, hints=None, exclude=()) -> dict:
    """{'jaw': name, 'eye': {side: name}, 'lid_upper': {side: name}, 'lid_lower': {side: name}}.

    lateral: bone name -> offset toward the character's own left (sides come from position).
    hints:   Unity avatar {'Jaw', 'LeftEye', 'RightEye'} used only where names find nothing;
             a hint naming a body bone (exclude) or an eyelid is ignored.
    """
    names = [n for n in bone_names if n not in set(exclude)]
    keys = {n: key(n) for n in names}
    jaws = [n for n, k in keys.items() if 'jaw' in k and not _is_end(k)]
    jaw = min(jaws, key=lambda n: (keys[n] not in ('jaw', 'jawbone', 'lowerjaw'), len(keys[n])), default=None)
    lids = {n: _lid(k) for n, k in keys.items() if not _is_end(k)}
    eyes = [n for n, k in keys.items() if 'eye' in k and not lids.get(n) and not _is_end(k) and not any(w in k for w in NOT_EYE)]
    found = {'jaw': jaw, 'eye': _sided(eyes, lateral),
             'lid_upper': _sided([n for n, kind in lids.items() if kind == 'upper'], lateral),
             'lid_lower': _sided([n for n, kind in lids.items() if kind == 'lower'], lateral)}
    hints = hints or {}
    usable = lambda n: n in keys and not lids.get(n)  # noqa: E731
    if found['jaw'] is None and usable(hints.get('Jaw')):
        found['jaw'] = hints['Jaw']
    if not found['eye']:
        for side in ('Left', 'Right'):
            if usable(hints.get(side + 'Eye')) and hints[side + 'Eye'] != found['jaw']:
                found['eye'][side] = hints[side + 'Eye']
    return found


def bone_angles(weights: np.ndarray, names) -> dict:
    """Per-frame angles in degrees from remapped weights (character's own left/right).

    jaw: opening; eye[side]: (yaw toward the character's left, pitch up); lid[side]: closure 0..1.
    """
    index = {n: i for i, n in enumerate(names)}

    def channel(n):
        return weights[:, index[n]] if n in index else np.zeros(len(weights))
    out = {'jaw': JAW_DEGREES * channel('jawOpen'), 'eye': {}, 'lid': {}}
    for side, other in (('Left', 'Right'), ('Right', 'Left')):
        outward = channel(f'eyeLookOut{side}') - channel(f'eyeLookIn{side}')
        toward_left = outward if side == 'Left' else -outward
        pitch = channel(f'eyeLookUp{side}') - channel(f'eyeLookDown{side}')
        out['eye'][side] = (EYE_YAW_DEGREES * toward_left, EYE_PITCH_DEGREES * pitch)
        out['lid'][side] = channel(f'eyeBlink{side}')
    return out
