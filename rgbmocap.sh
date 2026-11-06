#!/bin/sh
# Start RGB MoCap on Linux with the pinned GUI runtime (see docs/LINUX.md).
root="$(cd "$(dirname "$0")" && pwd)"
python="$root/.venv-gui/bin/python"
if [ ! -x "$python" ]; then
    echo "The GUI runtime is missing. Run ./install.sh first." >&2
    exit 1
fi
# Keep application caches inside the app folder, as the Windows launcher does.
export RGB_MOCAP_DATA="${RGB_MOCAP_DATA:-$root}"
export HF_HOME="$root/cache/huggingface" TORCH_HOME="$root/cache/torch" XDG_CACHE_HOME="$root/cache"
export PYTHONDONTWRITEBYTECODE=1
# The 3D viewport's OpenGL 4.3 core context fails under native Wayland EGL (EGL_BAD_MATCH,
# NVIDIA, Qt 6.8.3); X11/XWayland works. Set QT_QPA_PLATFORM to override.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
unset PYTHONHOME PYTHONPATH
cd "$root" || exit 1
exec "$python" -X utf8 app.py "$@"
