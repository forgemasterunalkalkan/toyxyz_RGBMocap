"""Key captured face values onto a retargeted character (character Blender runtime).

Called after the body bake, from the take folder's face.npz (copied from the source
project by retarget/worker.py). Shape keys whose names are ARKit channels (or common
aliases) receive their channel; jaw, eye and eyelid bones receive rotations only for
channels no shape key covers. Rotations are built in the character's own frame (left,
up, forward from the app's hip/head mapping), so bone roll does not matter.
"""
import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

from mocap.face_mapping import BONE_CHANNELS, LOWER_LID_DEGREES, UPPER_LID_DEGREES, bone_angles, channel_for, face_bones, remap
from .blender_bake import curves
from .file_paths import io_path


def _fill(fcurve, values):
    count = len(values)
    fcurve.keyframe_points.clear()
    fcurve.keyframe_points.add(count)
    fcurve.keyframe_points.foreach_set('co', np.column_stack((np.arange(1, count + 1), values)).reshape(-1))
    fcurve.keyframe_points.foreach_set('interpolation', [1] * count)  # LINEAR
    fcurve.update()


def _key_shapes(names, weights):
    """Key every mapped shape key; return {channel: [object/key, ...]}."""
    index = {n: i for i, n in enumerate(names)}
    covered = {}
    for obj in bpy.context.scene.objects:
        keys = obj.data.shape_keys if obj.type == 'MESH' else None
        if keys is None or not keys.use_relative:
            continue
        for block in keys.key_blocks[1:]:
            channel = channel_for(block.name, names)
            if channel is None:
                continue
            block.value = 0.0
            block.keyframe_insert('value', frame=1)
            path = block.path_from_id('value')
            fcurve = next(c for c in curves(keys.animation_data.action) if c.data_path == path)
            _fill(fcurve, weights[:, index[channel]])
            covered.setdefault(channel, []).append(f'{obj.name}/{block.name}')
    return covered


def _body_mapping(folder):
    kind = json.loads(Path(io_path(folder / 'import.json')).read_text(encoding='utf-8'))['kind']
    return [str(n) for n in np.load(io_path(folder / f'{kind}-canonical.npz'))['names']]


def _character_axes(folder, arm):
    """Armature-space unit vectors: the character's left, up and forward."""
    mapping = _body_mapping(folder)
    head = {i: Vector(arm.data.bones[mapping[i]].head_local) for i in (0, 1, 2, 15) if mapping[i] in arm.data.bones}
    left = (head[1] - head[2]).normalized()
    up = head[15] - head[0]
    up = (up - left * left.dot(up)).normalized()
    return left, up, left.cross(up)


def _face_rig(folder, arm, left):
    """Face bones by name, sides by position, Unity avatar hints as fallback; unskinned bones dropped."""
    mapping = _body_mapping(folder)
    head = Vector(arm.data.bones[mapping[15]].head_local)
    lateral = {b.name: (Vector(b.head_local) - head).dot(left) for b in arm.data.bones}
    try:
        hints = json.loads(Path(io_path(folder / 'unity-avatar.json')).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        hints = {}
    bones = face_bones(arm.data.bones.keys(), lateral=lateral, hints=hints, exclude=mapping)
    skinned = {g.name for o in bpy.context.scene.objects if o.type == 'MESH'
               for g in o.vertex_groups}
    weighted = set()
    for obj in (o for o in bpy.context.scene.objects if o.type == 'MESH' and o.vertex_groups):
        names = {g.index: g.name for g in obj.vertex_groups}
        for vertex in obj.data.vertices:
            weighted.update(names[g.group] for g in vertex.groups if g.weight > 0.05)
    keep = lambda n: n in skinned and n in weighted  # noqa: E731 - a bone that moves no vertex does nothing
    bones['jaw'] = bones['jaw'] if bones['jaw'] and keep(bones['jaw']) else None
    for part in ('eye', 'lid_upper', 'lid_lower'):
        bones[part] = {side: n for side, n in bones[part].items() if keep(n)}
    return bones


def _local_quaternions(arm, name, rotations):
    """Armature-space rotations about the bone's head -> pose-space quaternions (w, x, y, z)."""
    rest = arm.data.bones[name].matrix_local.to_3x3().normalized()
    inverse = rest.inverted()
    quats = np.array([tuple((inverse @ r @ rest).to_quaternion()) for r in rotations])
    # Keep consecutive quaternions in one hemisphere so linear keys do not flip.
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    return quats


def _key_bone(arm, name, quats):
    pose = arm.pose.bones[name]
    pose.rotation_mode = 'QUATERNION'
    pose.rotation_quaternion = quats[0]
    pose.keyframe_insert('rotation_quaternion', frame=1)
    path = pose.path_from_id('rotation_quaternion')
    for index in range(4):
        fcurve = next(c for c in curves(arm.animation_data.action) if c.data_path == path and c.array_index == index)
        _fill(fcurve, quats[:, index])


def apply_face(folder, arm, frames):
    """Key face.npz onto the character. Returns (keyed bone names, report or None)."""
    folder = Path(folder)
    source = folder / 'face.npz'
    if not Path(io_path(source)).is_file():
        return [], None
    data = np.load(io_path(source))
    names = [str(n) for n in data['names']]
    weights = remap(np.asarray(data['blend'], np.float64), names)
    if len(weights) >= frames:
        weights = weights[:frames]
    else:  # A shorter capture holds neutral at the end.
        weights = np.vstack([weights, np.zeros((frames - len(weights), weights.shape[1]))])

    covered = _key_shapes(names, weights)
    angles = bone_angles(weights, names)
    left, up, _forward = _character_axes(folder, arm)
    bones = _face_rig(folder, arm, left)
    keyed, driven = [], {}

    def rotations(axis, degrees):
        return [Matrix.Rotation(np.radians(d), 3, axis) for d in degrees]

    if bones['jaw'] and 'jawOpen' not in covered:
        _key_bone(arm, bones['jaw'], _local_quaternions(arm, bones['jaw'], rotations(left, angles['jaw'])))
        keyed.append(bones['jaw'])
        driven['jawOpen'] = bones['jaw']
    for side, name in bones['eye'].items():
        if any(c in covered for c in BONE_CHANNELS['eye'] if c.endswith(side)):
            continue
        yaw, pitch = angles['eye'][side]
        turns = [Matrix.Rotation(np.radians(y), 3, up) @ Matrix.Rotation(np.radians(p), 3, -left) for y, p in zip(yaw, pitch)]
        _key_bone(arm, name, _local_quaternions(arm, name, turns))
        keyed.append(name)
        driven[f'eyeLook ({side})'] = name
    for side in ('Left', 'Right'):
        if f'eyeBlink{side}' in covered:
            continue
        closure = angles['lid'][side]
        for part, degrees, axis in (('lid_upper', UPPER_LID_DEGREES, left), ('lid_lower', LOWER_LID_DEGREES, -left)):
            name = bones[part].get(side)
            if name:
                _key_bone(arm, name, _local_quaternions(arm, name, rotations(axis, degrees * closure)))
                keyed.append(name)
                driven[f'eyeBlink{side} ({part})'] = name

    report = dict(frames=frames, found_ratio=float(np.asarray(data['found']).mean()),
                  shape_channels=sorted(covered), shape_keys=sum(len(v) for v in covered.values()),
                  bone_channels=driven)
    return keyed, report
