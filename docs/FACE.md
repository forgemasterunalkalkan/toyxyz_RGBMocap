# Face capture (Linux branch)

**Capture Face (eyes and mouth)** reads blinks, gaze, brows and mouth shapes from the
video and keys them onto retargeted characters. It is an optional add-on to ExMR,
FlowHMR and GVHMR; the body motion is unaffected either way. Linux only for now.

## Setup

1. Build the face runtime where the GPU stages run:
   - remote workers: `.venv-gui/bin/python -m mocap.remote_workers --setup face`
   - local workers: `./install.sh --profiles face`

   It is a separate, hash-pinned environment (`.venv-face`: MediaPipe 1.1.1, OpenCV,
   NumPy; no torch), from `installer/face-requirements-linux.txt`.
2. In **Model Manager**, download **Face / MediaPipe Face Landmarker (Optional: Face)**
   (3.6 MB, Apache-2.0, pinned to model version 1 and its SHA-256).

## Use

1. Check **Capture Face (eyes and mouth)** in the **Face** panel and run **Reconstruct 3D**.
   Completed stages are reused, so on an existing project only face capture runs.
2. Retarget the character (**Retarget > person**). Face data is keyed when a character is
   retargeted; after recapturing a face, retarget again.
3. **Export Character… > FBX.**

The 3D viewport plays the face on the retarget preview, both shape-key and bone-driven
faces. Characters retargeted before this feature, or before recapturing, need **Retarget**
again to show it.

## How it works

`face` worker stage (`mocap/face_capture.py`, remote like the other model stages):

- Each frame is cropped around the head and upscaled to 384 px, because a full-body
  shot leaves the face ~60-120 px wide. The crop comes from the model's COCO head
  keypoints (ExMR), else the face found in the previous frame, else the top of the
  tracked person box.
- MediaPipe Face Landmarker returns 51 ARKit-named values (no `tongueOut`) and 478
  landmarks. "Left" is the performer's own left, as in ARKit.
- Dropouts up to 0.25 s are interpolated; longer ones fade to neutral. A median-of-3
  and a narrow Gaussian smooth the values without erasing 3-4 frame blinks.
- Output: `face.npz` and `face_receipt.json` in the model folder. A clip where no face
  is visible finishes with a warning and no face data; it does not fail the run.

Character mapping (`mocap/face_mapping.py`, `mocap/retarget/blender_face.py`, during the bake):

| Target on the character | Driven by | Notes |
|---|---|---|
| Shape keys named like ARKit channels (any case/separators), or aliases such as `Blink_L`, `EyeBlink.R`, `MouthOpen`, VRoid `Fcl_EYE_Close_L` | that channel | Preferred: a channel driving a shape key is not also sent to a bone |
| Jaw bone (`Jaw`, `mixamorig:Jaw`, …) | `jawOpen` | 22° opening about the character's left-right axis |
| Left/Right eye bones | `eyeLookIn/Out/Up/Down` | 30° yaw, 20° pitch |
| Upper/lower eyelid bones (`Orbicularis Top/Bottom`, `UpperLid`, …) | `eyeBlink` | 30° / 8° closure; not yet validated visually |

Face bones are found by name: a jaw is any bone with "jaw" in its name (`Jaw_M`,
`c_jawbone.x`, `JawMiddle`, end bones excluded); eyes are "eye" bones that are not
lids, ends, squash, brow or lash bones; eyelids are "eye" bones marked up/top/upper or
dwn/down/lower, or named `lid`/`Orbicularis`. Left and right come from each bone's
position across the face, not its name. When names find no jaw or eyes, a Unity
Humanoid avatar in the FBX's `.meta` is used as a hint, unless it names a body bone or
an eyelid. Bones that move no vertices are skipped.

Rotations are built in the character's own frame (left/up from the app's hip and head
mapping), so bone roll and axis conventions do not matter. The bake report
(`bake.json` → `face`) lists which channels went to shape keys and which to bones.

### Calibration

MediaPipe does not read a fully applied shape as 1.0. Rendering the test character at
known shape weights and reading the renders back gave:

| Rendered | MediaPipe score |
|---|---|
| neutral | blink 0.04-0.07, eye look 0.04-0.09 |
| `eyeBlinkLeft` 1.0 | `eyeBlinkLeft` 0.66 (right 0.09) |
| `mouthSmileLeft` 1.0 | `mouthSmileLeft` 0.78 (right 0.19) |
| look to own left 1.0 | `eyeLookOutLeft` 0.74, `eyeLookInRight` 0.75 |
| `jawOpen` 0.5 / 1.0 | 0.52 / 0.73 |

MediaPipe also carries per-performer biases: bangs over the brows read as raised brows
(0.70-0.97 on every frame of one test clip), and a head turned while the eyes hold the
camera reads as a constant side glance. So each channel is measured from the performer's
own resting level, the 5th percentile of the clip, and scaled by the measured span from
neutral to full (blink 0.57, eye look 0.67, jaw 0.73, others 0.70). Eye look is centred
per eye and axis on the clip's median, so gaze is straight on average and follows glances.
The same renders confirmed MediaPipe's left/right convention.

### Viewport

The retarget bake stores every animated face shape as sparse per-vertex offsets in the
take's `preview.npz` (`morph`, `morph_rows`) and packs the per-frame shape weights into
extra 4x4 blocks after the bone palette in `preview-frames.npy` (`morph_block`). The
character skinning shader adds the weighted offsets before skinning. Checked against a
NumPy reference on an offscreen GL 4.3 context: identical to 0.0005 mm.

MediaPipe's blink channel also rises with squinting: a performer who scowls through a
clip keeps narrowed eyes on the character.

## Test character

A CC0 MakeHuman character with both kinds of face targets: all 52 ARKit shape keys
(propagated to eyes, brows and lashes) and Mixamo-named jaw, eye and eyelid bones.
`tools/build_face_test_character.py` builds it headless in the app's Blender from
[MPFB](https://extensions.blender.org/add-ons/mpfb/) 2.0.18 (GPL code; MakeHuman
assets CC0) with the `faceunits01` and `makehuman_system_assets_cc0` packs. Hard-won
details:

- Use MPFB's `mixamo_unity` rig. The plain `mixamo` rig gives the hand bone at most 0.2
  weight and the app rejects it (`AssertionError ('mixamo', 0, 'palm')`).
- Add the rig before eyes, eyebrows and eyelashes, or MPFB's export copy drops them.
- Use the `GAMEENGINE` material type; `MAKESKIN` exports the skin texture into alpha.
- `--face-texture-only` keeps the skin texture on the head (polygons skinned to the head,
  jaw, eye and eyelid bones) and gives the rest of the body a plain grey material.
