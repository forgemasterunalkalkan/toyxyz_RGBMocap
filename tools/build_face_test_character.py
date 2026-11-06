"""Build a CC0 MakeHuman test character for face capture evaluation (run inside Blender 4.5).

BLENDER_USER_RESOURCES=<isolated profile dir> runtime_character/blender -b --factory-startup \
    --python tools/build_face_test_character.py -- MPFB_ZIP PACKS_COMMA_SEPARATED OUT.fbx [--eyes] [--face-texture-only]

Inputs (docs/FACE.md#test-character): MPFB 2.0.18 from extensions.blender.org, MakeHuman
faceunits01.zip and makehuman_system_assets_cc0.zip. Use an isolated BLENDER_USER_RESOURCES:
the script installs MPFB into that Blender profile.

Result: MPFB "mixamo_unity" rig (the 52 Mixamo bones plus Root, Jaw, eyes and eyelids) + all 52
ARKit face shape keys (MakeHuman faceunits01), propagated to eyes, brows and lashes, exported as FBX.
"""
import json
import sys
import zipfile
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index('--') + 1:]
mpfb_zip, packs, out_fbx = Path(args[0]), [Path(p) for p in args[1].split(',') if p], Path(args[2])
with_eyes = '--eyes' in args
face_texture_only = '--face-texture-only' in args

if 'bl_ext.user_default.mpfb' not in bpy.context.preferences.addons:
    bpy.ops.extensions.package_install_files(filepath=str(mpfb_zip), repo='user_default', enable_on_install=True)

from bl_ext.user_default.mpfb.services import (AssetService, ExportService, FaceService, HumanService,  # noqa: E402
                                               LocationService, ObjectService, TargetService)

data = Path(LocationService.get_user_data())
for pack in packs:
    marker = data / f'.installed-{pack.name}'
    if not marker.exists():
        with zipfile.ZipFile(pack) as archive:
            archive.extractall(data)
        marker.touch()
AssetService.rescan_pack_metadata()
print('RESULT faceunits installed:', FaceService.is_faceunits01_installed(force_recheck=True))

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

basemesh = HumanService.create_human(feet_on_ground=True, scale=0.1)
# mixamo_unity: the 52 Mixamo bones plus Root, Jaw, eyes and eyelids, with complete skin weights
# (the plain 'mixamo' rig gives the hand bone at most 0.2 weight).
armature = HumanService.add_builtin_rig(basemesh, 'mixamo_unity', import_weights=True)
if with_eyes:
    for fragment, subdir, kind in (('high-poly/high-poly.mhclo', 'eyes', 'Eyes'),
                                   ('eyebrow001/eyebrow001.mhclo', 'eyebrows', 'Eyebrows'),
                                   ('eyelashes01/eyelashes01.mhclo', 'eyelashes', 'Eyelashes')):
        path = AssetService.find_asset_absolute_path(fragment, subdir)
        print('RESULT asset', kind, path)
        if path:
            HumanService.add_mhclo_asset(path, basemesh, asset_type=kind, subdiv_levels=0, material_type='GAMEENGINE')
    skin = AssetService.find_asset_absolute_path('young_caucasian_male/young_caucasian_male.mhmat', 'skins')
    print('RESULT skin', skin)
    if skin:
        HumanService.set_character_skin(str(skin), basemesh, skin_type='GAMEENGINE')

collection = basemesh.users_collection[0]
copy = ExportService.create_character_copy(basemesh, name_suffix='_export', place_in_collection=collection)
new_basemesh = ObjectService.find_object_of_type_amongst_nearest_relatives(copy)
TargetService.bake_targets(new_basemesh)
FaceService.load_targets(new_basemesh, load_microsoft_visemes=False, load_meta_visemes=False, load_arkit_faceunits=True)
FaceService.interpolate_targets(new_basemesh)
ExportService.bake_modifiers_remove_helpers(new_basemesh, bake_masks=True, bake_subdiv=False,
                                            remove_helpers=True, also_proxy=True)

# Keep only the export copy: its armature and the meshes parented to it.
keep = {copy} | set(copy.children_recursive)
for obj in list(bpy.data.objects):
    if obj not in keep:
        bpy.data.objects.remove(obj, do_unlink=True)
copy.name = 'Armature'

# Normalize skin weights (a few MPFB vertices sum to 2.0); engines and the app expect 1.0.
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.vertex_groups:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.vertex_group_normalize_all(group_select_mode='BONE_DEFORM', lock_active=False)


def untexture_body(armature):
    """Keep the skin texture on the head only; everything below the neck gets a plain material.

    Head polygons are those whose vertices are all mostly skinned to the head, jaw, eye or
    eyelid bones. Eyes, brows and lashes are separate meshes and stay textured.
    """
    import numpy as np
    face_bones = {b.name for b in armature.data.bones
                  if any(k in b.name for k in ('Head', 'Jaw', 'Eye', 'Orbicularis')) and not b.name.endswith('_End')}
    body = next(o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('Human_export'))
    groups = {g.index for g in body.vertex_groups if g.name in face_bones}
    head = np.array([sum(g.weight for g in v.groups if g.group in groups) for v in body.data.vertices]) >= 0.5
    plain = bpy.data.materials.new('Human.body_untextured')
    plain.use_nodes = True
    shader = plain.node_tree.nodes['Principled BSDF']
    shader.inputs['Base Color'].default_value = (0.55, 0.55, 0.55, 1.0)
    shader.inputs['Roughness'].default_value = 0.6
    body.data.materials.append(plain)
    index = len(body.data.materials) - 1
    moved = 0
    for polygon in body.data.polygons:
        if not all(head[i] for i in polygon.vertices):
            polygon.material_index = index
            moved += 1
    print('RESULT untextured body polygons', moved, 'of', len(body.data.polygons), '| head vertices', int(head.sum()))


if face_texture_only:
    untexture_body(copy)

report = dict(armature=copy.name, bones=len(copy.data.bones),
              bone_sample=[b.name for b in copy.data.bones][:6], meshes={})
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        keys = [k.name for k in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else []
        report['meshes'][obj.name] = dict(vertices=len(obj.data.vertices), shape_keys=len(keys),
                                          modifiers=[m.type for m in obj.modifiers])
print('RESULT', json.dumps(report))

bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.data.objects:
    obj.select_set(True)
out_fbx.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.fbx(filepath=str(out_fbx), use_selection=True, object_types={'ARMATURE', 'MESH'},
                         use_mesh_modifiers=False, add_leaf_bones=False, bake_anim=False,
                         mesh_smooth_type='FACE', path_mode='COPY', embed_textures=True)
print('RESULT exported', out_fbx, out_fbx.stat().st_size)
