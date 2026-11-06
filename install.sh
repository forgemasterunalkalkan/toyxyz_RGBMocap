#!/bin/sh
# Linux setup: pinned runtimes for the GUI and CPU stages, motion correction, Blender 4.5.
# GPU model runtimes are added with --profiles (e.g. shared gui exmr), or on a remote
# workstation with: python3 -m mocap.remote_workers --init && python3 -m mocap.remote_workers --setup
cd "$(dirname "$0")" || exit 1
if [ "$#" -eq 0 ]; then
    set -- --profiles shared gui --correction --blender
fi
exec python3 -u tools/setup_linux.py "$@"
