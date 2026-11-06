"""Prepare RGB MoCap's pinned runtimes and support tools on Linux x86_64.

Builds the same separated profiles as the Windows setup, from
installer/runtime-lock-linux.json (the Windows pins mapped to manylinux wheels,
see tools/linux_lock.py), in the developer layout the Linux app uses:

  shared -> .venv        Python 3.10.22  (GVHMR stages, tracking, correction, export)
  gui    -> .venv-gui    overlay on shared (PySide6)
  flow   -> .venv-flow   overlay on shared (FlowHMR)
  hand   -> .venv-hand   overlay on shared (HandFlow)
  htd    -> .venv-htd    overlay on shared (HTD-Refine)
  exmr   -> .venv-exmr   Python 3.12.10  (ExMR)

Every wheel is installed --no-deps with its pinned SHA-256. Python and uv are
fetched by the pinned uv into .runtime/ and .bootstrap/, caches stay in .cache/.
It never touches models/, projects/, cache/ or other user data.

Standard library only (system python3 >= 3.9).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'installer/runtime-lock-linux.json'
SOURCE_LOCK = ROOT / 'installer/runtime-lock.json'
STATE = ROOT / '.setup/linux-runtime-state.json'
FOLDERS = {'shared': '.venv', 'gui': '.venv-gui', 'flow': '.venv-flow', 'hand': '.venv-hand',
           'htd': '.venv-htd', 'exmr': '.venv-exmr'}
CHECKS = {
    'shared': 'import torch,torchvision,numpy,smplx;print("torch",torch.__version__,"cuda",torch.cuda.is_available())',
    'gui': 'import PySide6,shiboken6;print("PySide6",PySide6.__version__)',
    'flow': 'import torch,timm,numpy;print("torch",torch.__version__,"timm",timm.__version__,"numpy",numpy.__version__)',
    'hand': 'import torch,pyrender,trimesh;print("trimesh",trimesh.__version__)',
    'htd': 'import torch,timm;print("timm",timm.__version__)',
    'exmr': 'import torch,torchvision,numpy;print("torch",torch.__version__,"cuda",torch.cuda.is_available())',
    'face': 'import mediapipe,cv2,numpy;from mediapipe.tasks.python import vision;print("mediapipe",mediapipe.__version__,"opencv",cv2.__version__)',
}
FACE_FOLDER, FACE_PYTHON, FACE_REQUIREMENTS = '.venv-face', '3.12.10', 'installer/face-requirements-linux.txt'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def say(message):
    print(message, flush=True)


def environment():
    env = os.environ.copy()
    env.update(UV_PYTHON_INSTALL_DIR=str(ROOT / '.runtime/python'), UV_CACHE_DIR=str(ROOT / '.cache/uv'),
               UV_NO_CONFIG='1', UV_PYTHON_PREFERENCE='only-managed', PYTHONUTF8='1')
    env.pop('VIRTUAL_ENV', None)
    return env


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], cwd=ROOT, env=environment(), check=True, **kwargs)


def download(url, target, digest, size=None):
    target = Path(target)
    if target.is_file() and sha256(target) == digest:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    say(f'Downloading {target.name}' + (f' ({size / 2**20:.0f} MiB)' if size else ''))
    request = urllib.request.Request(url, headers={'User-Agent': 'rgbmocap-setup'})
    with urllib.request.urlopen(request, timeout=120) as response, open(part, 'wb') as out:
        shutil.copyfileobj(response, out, 1 << 20)
    if sha256(part) != digest:
        part.unlink()
        raise RuntimeError(f'Checksum mismatch for {target.name}')
    part.replace(target)
    return target


def load_lock():
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    if lock['source_lock_sha256'] != sha256(SOURCE_LOCK):
        raise SystemExit('installer/runtime-lock-linux.json does not match installer/runtime-lock.json. '
                         'Regenerate it with: python3 tools/linux_lock.py')
    return lock


def ensure_uv(lock):
    entry = lock['uv']
    binary = ROOT / f".bootstrap/uv-{entry['version']}/uv"
    if binary.is_file():
        return binary
    wheel = download(entry['url'], ROOT / '.cache/wheels' / entry['filename'], entry['sha256'], entry['bytes'])
    with zipfile.ZipFile(wheel) as archive:
        member = next(n for n in archive.namelist() if n.endswith('/scripts/uv'))
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(archive.read(member))
    binary.chmod(0o755)
    return binary


def bundled_wheels():
    """Pure-Python wheels rebuilt from installer/runtime_sources (shared with Windows)."""
    sys.path.insert(0, str(ROOT / 'tools'))
    from runtime_support import prepared_lock
    work = ROOT / 'build/dependencies'
    prepared = prepared_lock(work)
    return {p['bundled']: (work / p['bundled'], p['sha256'])
            for g in prepared['groups'] for p in g['packages'] if p.get('bundled')}


def requirement_lines(group, bundled):
    lines = []
    for package in group['packages']:
        if package.get('bundled'):
            path, digest = bundled[package['bundled']]
            lines.append(f"{package['name']} @ {path.as_uri()} --hash=sha256:{digest}")
        else:
            lines.append(f"{package['name']} @ {package['url']} --hash=sha256:{package['sha256']}")
    return lines


def site_packages(folder, python_version):
    return ROOT / folder / f"lib/python{'.'.join(python_version.split('.')[:2])}/site-packages"


def build_profile(uv, group, bundled):
    key = group['key']
    folder = ROOT / FOLDERS[key]
    python = folder / 'bin/python'
    if not python.is_file():
        run(uv, 'venv', folder, '--python', group['python'], '--no-project')
    requirements = ROOT / f'build/linux-requirements-{key}.txt'
    requirements.parent.mkdir(parents=True, exist_ok=True)
    requirements.write_text('\n'.join(requirement_lines(group, bundled)) + '\n', encoding='utf-8')
    say(f"Installing {key}: {len(group['packages'])} pinned packages")
    run(uv, 'pip', 'install', '--python', python, '--no-deps', '--require-hashes', '-r', requirements)
    if group.get('base'):
        shared = site_packages(FOLDERS[group['base']], group['python'])
        (site_packages(FOLDERS[key], group['python']) / 'shared_runtime.pth').write_text(
            'import sys; sys.path.append(' + ascii(str(shared)) + ')\n', encoding='ascii')
    verify(key, python, group)


def verify(key, python, group):
    pins = {p['name']: p['version'] for p in group['packages']}
    script = ('import importlib.metadata as m,json,sys;pins=json.loads(sys.argv[1]);'
              'bad=[f"{n}=={v} (found {m.version(n)})" for n,v in pins.items() if m.version(n)!=v];'
              'print("\\n".join(bad));sys.exit(1 if bad else 0)')
    mismatch = subprocess.run([str(python), '-c', script, json.dumps(pins)], capture_output=True, text=True)
    if mismatch.returncode:
        raise SystemExit(f'{key}: installed versions differ from the lock:\n{mismatch.stdout}{mismatch.stderr}')
    check = subprocess.run([str(python), '-c', CHECKS[key]], capture_output=True, text=True)
    if check.returncode:
        raise SystemExit(f'{key}: import check failed:\n{check.stderr[-2000:]}')
    say(f'READY {key}: {check.stdout.strip()}')


def build_face(uv):
    """Separate MediaPipe runtime for face capture (mocap/face_runtime.py); no torch."""
    folder = ROOT / FACE_FOLDER
    python = folder / 'bin/python'
    if not python.is_file():
        run(uv, 'venv', folder, '--python', FACE_PYTHON, '--no-project')
    say('Installing face: hash-pinned MediaPipe runtime')
    run(uv, 'pip', 'install', '--python', python, '--no-deps', '--require-hashes', '-r', ROOT / FACE_REQUIREMENTS)
    check = subprocess.run([str(python), '-c', CHECKS['face']], capture_output=True, text=True)
    if check.returncode:
        raise SystemExit('face: import check failed:\n' + check.stderr[-2000:])
    say(f'READY face: {check.stdout.strip()}')


def build_correction():
    """Compile the Kimodo motion-correction extension for the shared Python 3.10 runtime."""
    python = ROOT / '.venv/bin/python'
    if not python.is_file():
        raise SystemExit('Build the shared profile before motion correction.')
    cmake = shutil.which('cmake')
    if not cmake or not (shutil.which('c++') or shutil.which('g++')):
        raise SystemExit('Motion correction needs cmake and a C++17 compiler (apt install cmake g++).')
    base = subprocess.run([str(python), '-c', 'import sys;print(sys.base_prefix)'], capture_output=True,
                          text=True, check=True).stdout.strip()
    source = ROOT / 'third_party/KimodoMotionCorrection'
    build = ROOT / 'build/motion-correction-linux'
    say('Building motion correction (C++17, Release)')
    subprocess.run([cmake, '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                    '-DPython3_EXECUTABLE=' + str(Path(base) / 'bin/python3.10'),
                    '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON'], check=True)
    subprocess.run([cmake, '--build', build, '--config', 'Release', '--target', '_motion_correction',
                    '--parallel', str(min(8, os.cpu_count() or 2))], check=True)
    built = sorted(build.rglob('_motion_correction*.so'))
    if not built:
        raise SystemExit('The motion-correction build produced no module.')
    target = source / 'bin'
    target.mkdir(exist_ok=True)
    for old in target.glob('_motion_correction*.so'):
        old.unlink()
    shutil.copy2(built[0], target / built[0].name)
    probe = subprocess.run([str(python), '-c', 'from mocap.correction import backend;m=backend();print(m.__name__)'],
                           cwd=ROOT, capture_output=True, text=True)
    if probe.returncode:
        raise SystemExit('Motion correction does not import:\n' + probe.stderr[-2000:])
    say(f'READY correction: {target / built[0].name}')


def install_blender(lock):
    entry = lock['blender']
    target = ROOT / 'runtime_character'
    if (target / 'blender').is_file():
        say(f'READY blender: {target / "blender"}')
        return
    archive = download(entry['url'], ROOT / '.cache/downloads' / Path(entry['url']).name, entry['sha256'], entry['bytes'])
    say('Extracting Blender')
    stage = ROOT / '.cache/blender-stage'
    shutil.rmtree(stage, ignore_errors=True)
    with tarfile.open(archive) as tar:
        tar.extractall(stage, filter='tar') if sys.version_info >= (3, 12) else tar.extractall(stage)
    (stage / entry['archive_root']).replace(target)
    shutil.rmtree(stage, ignore_errors=True)
    version = subprocess.run([str(target / 'blender'), '--version'], capture_output=True, text=True)
    say('READY blender: ' + (version.stdout.splitlines() or ['unknown'])[0])


def check_ffmpeg():
    found = [shutil.which(n) for n in ('ffmpeg', 'ffprobe')]
    if not all(found):
        raise SystemExit('FFmpeg is missing. Install it (apt install ffmpeg) or place ffmpeg/ffprobe in tools/bin.')
    version = subprocess.run([found[0], '-version'], capture_output=True, text=True).stdout.splitlines()[0]
    say('READY ffmpeg: ' + version)


def main():
    parser = argparse.ArgumentParser(description='Prepare pinned RGB MoCap runtimes on Linux.')
    parser.add_argument('--profiles', nargs='+', choices=[*FOLDERS, 'face'], default=['shared', 'gui'],
                        help='Runtimes to build (overlays add shared automatically).')
    parser.add_argument('--correction', action='store_true', help='Build the native motion-correction module.')
    parser.add_argument('--blender', action='store_true', help='Install the pinned Blender 4.5 for characters.')
    parser.add_argument('--keep-cache', action='store_true', help='Keep the uv download cache afterwards.')
    args = parser.parse_args()
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise SystemExit('This setup supports Linux x86_64 only; use install.cmd on Windows.')
    lock = load_lock()
    groups = {g['key']: g for g in lock['groups']}
    face = 'face' in args.profiles
    profiles = [k for k in args.profiles if k != 'face']
    wanted = list(dict.fromkeys((['shared'] if any(groups[k].get('base') == 'shared' for k in profiles) else [])
                                + profiles))
    uv = ensure_uv(lock)
    pythons = sorted({groups[k]['python'] for k in wanted} | ({FACE_PYTHON} if face else set()))
    run(uv, 'python', 'install', *pythons)
    if wanted:
        bundled = bundled_wheels()
    for key in wanted:
        build_profile(uv, groups[key], bundled)
    if face:
        build_face(uv)
        wanted.append('face')
    check_ffmpeg()
    if args.correction:
        build_correction()
    if args.blender:
        install_blender(lock)
    if not args.keep_cache:
        # Installed files are hardlinked from the cache; pruning drops only unused entries.
        run(uv, 'cache', 'prune')
    STATE.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(STATE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    state.update(lock_sha256=sha256(LOCK), source_lock_sha256=lock['source_lock_sha256'],
                 profiles=sorted(set(state.get('profiles', [])) | set(wanted)))
    STATE.write_text(json.dumps(state, indent=2), encoding='utf-8')
    say('Linux runtimes ready: ' + ', '.join(wanted))


if __name__ == '__main__':
    main()
