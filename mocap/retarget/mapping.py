from collections import Counter
import numpy as np
A = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
NAMES = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot', 'Spine2', 'LeftToeBase', 'RightToeBase', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand']
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
VRM = ['hips', 'leftUpperLeg', 'rightUpperLeg', 'spine', 'leftLowerLeg', 'rightLowerLeg', 'chest', 'leftFoot', 'rightFoot', 'upperChest', 'leftToes', 'rightToes', 'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand']
PMX = ['下半身', '足.L', '足.R', '上半身', 'ひざ.L', 'ひざ.R', '上半身1', '足首.L', '足首.R', '上半身2', 'つま先.L', 'つま先.R', '首', '肩.L', '肩.R', '頭', '腕.L', '腕.R', 'ひじ.L', 'ひじ.R', '手首.L', '手首.R']

def body_trajectory_root(parents, mapped):
    """Use the body's existing root; independent control trees may coexist."""
    roots = set()
    for name in mapped:
        seen = set()
        while True:
            if name not in parents or name in seen:
                raise ValueError('Invalid humanoid bone hierarchy.')
            seen.add(name)
            parent = parents[name]
            if parent is None:
                roots.add(name)
                break
            name = parent
    if len(roots) != 1:
        raise ValueError('The mapped humanoid body must share one hierarchy root.')
    return next(iter(roots))

FINGERS = [side+'Hand'+finger+str(n) for side in ('Left','Right') for finger in ('Thumb','Index','Middle','Ring','Pinky') for n in (1,2,3)]
SEPARATORS = ':_|.- '


def _side(table, side):
    return {name.replace('Left', side): [c.replace('_l', '_'+side[0].lower()).replace('.l', '.'+side[0].lower()) for c in candidates]
            for name, candidates in table.items()}


def _convention(center, sided, finger):
    table = dict(center)
    for side in ('Left', 'Right'):
        table.update(_side(sided, side))
        for f in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
            for n in (1, 2, 3):
                table[side+'Hand'+f+str(n)] = [c.format(f=f.lower(), n=n).replace('_l', '_'+side[0].lower()).replace('.l', '.'+side[0].lower()) for c in finger]
    return table


# Other widespread naming conventions, mapped onto the Mixamo roles (case-insensitive exact names).
CONVENTIONS = {
    'Advanced Skeleton': _convention(
        {'Hips': ['root_m'], 'Spine': ['spine1_m'], 'Spine1': ['spine2_m', 'spine3_m'], 'Spine2': ['chest_m'], 'Neck': ['neck_m'], 'Head': ['head_m']},
        {'LeftUpLeg': ['hip_l'], 'LeftLeg': ['knee_l'], 'LeftFoot': ['ankle_l'], 'LeftToeBase': ['toes_l'], 'LeftShoulder': ['scapula_l'],
         'LeftArm': ['shoulder_l'], 'LeftForeArm': ['elbow_l'], 'LeftHand': ['wrist_l']},
        ['{f}finger{n}_l']),
    'Auto-Rig Pro': _convention(
        {'Hips': ['root.x'], 'Spine': ['spine_01.x'], 'Spine1': ['spine_02.x'], 'Spine2': ['spine_03.x'], 'Neck': ['neck.x'], 'Head': ['head.x']},
        {'LeftUpLeg': ['thigh_stretch.l', 'thigh.l'], 'LeftLeg': ['leg_stretch.l', 'leg.l'], 'LeftFoot': ['foot.l'], 'LeftToeBase': ['toes_01.l', 'toes.l'],
         'LeftShoulder': ['shoulder.l'], 'LeftArm': ['arm_stretch.l', 'arm.l'], 'LeftForeArm': ['forearm_stretch.l', 'forearm.l'], 'LeftHand': ['hand.l']},
        ['c_{f}{n}.l', '{f}{n}.l']),
}


def _mixamo_candidates(bones, role):
    exact = [b for b in bones if b.split(':')[-1] == role]
    if len(exact) == 1:
        return exact
    # Case-insensitive, after any prefix: Bind_Hips, mixamorig1:Hips, RIghtArm.
    target = role.lower()
    return [b for b in bones if b.lower() == target or (b.lower().endswith(target) and b[-len(target)-1] in SEPARATORS)]


def resolve_roles(bones, deforming=None, parents=None):
    """Mixamo role -> bone name for the 22 body joints and 30 finger bones, and the naming used.

    Mixamo names (exact, then case-insensitive after any prefix), else a whole known convention.
    Body joints must all resolve; fingers are optional. When a rig carries control copies
    (Bind_Hips and FK_Hips), the bone that deforms meshes wins, then the prefix most roles use.
    """
    bones = list(bones)
    candidates = {role: _mixamo_candidates(bones, role) for role in NAMES + FINGERS}
    if deforming is not None:
        candidates = {role: ([c for c in found if c in deforming] or found) if len(found) > 1 else found
                      for role, found in candidates.items()}
    prefixes = Counter(found[0][:len(found[0]) - len(role)].lower() for role, found in candidates.items() if len(found) == 1)
    roles = {}
    problems = {}
    for role in NAMES + FINGERS:
        found = candidates[role]
        if len(found) > 1:
            ranked = sorted(found, key=lambda c: -prefixes.get(c[:len(c) - len(role)].lower(), 0))
            if prefixes.get(ranked[0][:len(ranked[0]) - len(role)].lower(), 0) > prefixes.get(ranked[1][:len(ranked[1]) - len(role)].lower(), 0):
                found = ranked[:1]
        if len(found) == 1:
            roles[role] = found[0]
        elif role in NAMES:
            problems[role] = 'ambiguous' if found else 'missing'
    if parents:
        # A hand with an unusual name (e.g. LeftHandHand) is the bone its fingers hang from, below the forearm.
        for side in ('Left', 'Right'):
            hand, forearm = side + 'Hand', side + 'ForeArm'
            roots = {parents.get(roles.get(side + 'Hand' + f + '1')) for f in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')} - {None}
            if hand in problems and forearm in roles and len(roots) == 1:
                candidate, up = next(iter(roots)), next(iter(roots))
                while up is not None and up != roles[forearm]:
                    up = parents.get(up)
                if up == roles[forearm]:
                    roles[hand] = candidate
                    problems.pop(hand)
    if not problems:
        return roles, 'Mixamo'
    lower = {b.lower(): b for b in bones}
    for convention, table in CONVENTIONS.items():
        found = {role: next((lower[c] for c in candidates if c in lower), None) for role, candidates in table.items()}
        if all(found[role] for role in NAMES):
            return {role: bone for role, bone in found.items() if bone}, convention
    missing = [r for r, p in problems.items() if p == 'missing']
    ambiguous = [r for r, p in problems.items() if p == 'ambiguous']
    detail = ('Missing: ' + ', '.join(missing) if missing else '') + ('; ' if missing and ambiguous else '') + ('Ambiguous: ' + ', '.join(ambiguous) if ambiguous else '')
    raise ValueError('Expected a humanoid rig with Mixamo, Advanced Skeleton or Auto-Rig Pro bone names. ' + detail)


def body_mapping(kind,bones,semantic,deforming=None,parents=None):
    names=set(bones)
    if kind=='mixamo':
        roles,_=resolve_roles(bones,deforming,parents)
        mapping=[roles[n] for n in NAMES]
        if len(set(mapping))!=22:raise ValueError('Humanoid bones must have distinct mappings')
        return mapping
    mapping=[semantic.get(n) for n in VRM] if kind=='vrm' else PMX
    missing=[NAMES[i] for i,n in enumerate(mapping) if n not in names]
    if missing:raise ValueError('Unsupported humanoid mapping (manual mapping is not available yet): '+', '.join(missing))
    if len(set(mapping))!=22:raise ValueError('Humanoid bones must have distinct mappings')
    return mapping
