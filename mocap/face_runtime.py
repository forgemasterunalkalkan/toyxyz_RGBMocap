"""Separate face capture runtime: MediaPipe, OpenCV and NumPy only (no torch).

Kept apart from the model runtimes, like MAMMA and PromptHMR: MediaPipe pins its own
NumPy/OpenCV, which must not leak into the pinned motion-model environments.
Built by tools/setup_linux.py (profile ``face``) from installer/face-requirements-linux.txt.
"""
from __future__ import annotations
import os
from pathlib import Path

STAGES = frozenset({'face'})
FOLDER = '.venv-face'
PYTHON = '3.12.10'
MODULE = 'mocap.face_capture'
REQUIREMENTS = 'installer/face-requirements-linux.txt'


def python(root) -> Path:
    return Path(root) / FOLDER / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def command(root, stage, args, check=True) -> list[str]:
    if os.name == 'nt':
        raise RuntimeError('Face capture is not available in the Windows distribution yet.')
    interpreter = python(root)
    if check and not interpreter.is_file():
        raise RuntimeError('The face capture runtime is missing. Run ./install.sh --profiles face.')
    return [str(interpreter), '-X', 'utf8', '-m', MODULE, stage, *map(str, args)]
