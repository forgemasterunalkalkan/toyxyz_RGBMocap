# RGB MoCap on Linux

The Linux port runs the same application code and the same pinned package
versions as the Windows Git/CMD distribution. It adds two things:

- **Linux runtimes**: `install.sh` builds the separated model runtimes from
  `installer/runtime-lock-linux.json`, which maps every Windows pin to its
  manylinux wheel at the same version (`tools/linux_lock.py`). Only torch's
  CUDA libraries and Triton are added, at the versions torch itself requires.
- **Remote workers** (optional): the GUI runs on one machine and the GPU stages
  run on a workstation over SSH. Video import, contact correction, Blender
  retargeting and export stay local.
- **Shared project path** (optional): a copy stored anywhere appears at the path
  its projects were made under, so projects move between machines and several
  machines share one workstation mirror. See [Shared project path](#shared-project-path).

Supported here: ExMR, FlowHMR and GVHMR runtimes. MAMMA, PromptHMR and PEAR
still use Windows-only setup and are not available on Linux yet.

## Requirements

- Linux x86_64, glibc 2.36 or newer (Debian 12+, Ubuntu 23.04+, Fedora 37+).
- NVIDIA driver with CUDA 12.8 support (driver 570+) for local inference.
  The CUDA toolkit is not needed; torch's wheels bring their own libraries.
- `python3`, `ffmpeg`, `rsync`, `cmake` and a C++17 compiler
  (`apt install python3 ffmpeg rsync cmake g++`). Minimal desktops may also
  lack Qt's X11 cursor library (`libxcb-cursor0`; Fedora `xcb-util-cursor`).
- Fedora Atomic/Bazzite (not yet tested here): `python3`, `rsync` and `ffmpeg`
  are normally present; add the compiler for motion correction with
  `rpm-ostree install cmake gcc-c++` and reboot.
- Disk: about 6 GB per torch runtime (shared, ExMR) plus model files.
- For a [shared project path](#shared-project-path): bubblewrap and
  unprivileged user namespaces (Debian: `apt install bubblewrap`; Fedora and
  Bazzite include both).

## Install and start

```sh
./install.sh          # shared + gui runtimes, motion correction, Blender 4.5
./rgbmocap.sh         # start the app
```

`rgbmocap.sh` starts the window through X11 (XWayland on Wayland desktops):
under native Wayland the 3D viewport's OpenGL 4.3 core context failed with
`EGL_BAD_MATCH` (NVIDIA driver 595, Qt 6.8.3). Set `QT_QPA_PLATFORM=wayland`
to try native Wayland anyway.

`install.sh` with no arguments prepares what the GUI and the local CPU stages
need. To run AI models on this machine too, add their runtimes:

```sh
./install.sh --profiles shared gui exmr flow hand htd --correction --blender
```

| Profile | Folder | Python | Used by |
|---|---|---|---|
| shared | `.venv` | 3.10.22 | GVHMR stages, tracking, correction, export |
| gui | `.venv-gui` | overlay on shared | the application window |
| exmr | `.venv-exmr` | 3.12.10 | ExMR (and FlowHMR's optional HaMeR) |
| flow | `.venv-flow` | overlay on shared | FlowHMR |
| hand | `.venv-hand` | overlay on shared | HandFlow |
| htd | `.venv-htd` | overlay on shared | HTD-Refine |

Every wheel is installed `--no-deps` with its pinned SHA-256, and each profile
is checked against the lock after installation. Model files are still obtained
through **Model Manager**, as on Windows.

After changing `installer/runtime-lock.json`, regenerate the Linux lock with
`python3 tools/linux_lock.py` (needs `packaging` and network access);
`setup_linux.py` refuses a stale Linux lock.

## Remote workers

Use this when the machine with the screen is not the machine with the GPU.

### How it works

Stages that run models (`track`, `people`, `person`, ExMR, FlowHMR and GVHMR
stages and their batches, and face capture; see `REMOTE_STAGES` in `mocap/remote_workers.py`)
start a local proxy instead of the worker. For each job the proxy:

1. copies the source code and `models/` to the host, where the worker sees
   them at the **same absolute path** as on this machine (stored at that path,
   or in another folder shown at that path, see
   [One workstation for several machines](#one-workstation-for-several-machines));
2. reconciles the job's project workspaces with the host (the whole
   `<video>_project` folder, since workers also read and write the workspace
   around the model folder they were given);
3. runs the identical worker command there and relays its output unchanged,
   so progress, cancellation and the Console behave as for a local job;
4. copies the results back before the GUI starts its next job.

Because the worker sees the same paths as this machine, receipts, cache
identities and model receipts stay valid without any path translation. Project folders are
reconciled both ways against the state of the last sync: changes on either side
are copied, deletions are propagated, and if both sides changed one file the
newer copy wins. Outputs of a cancelled job are therefore kept for the next
run, and files you delete locally are not brought back.

`exmr_frames/` folders (one JPEG per frame, at any depth) never leave the
host; they are rebuilt from `input.mp4` when needed.

Cancel closes the SSH connection; `tools/remote_exec.py` then stops the remote
worker and everything it started. One job runs on the host at a time.

### Set up

1. Key-based SSH to the host without prompts (`ssh <host> true` must work;
   load passphrase-protected keys with `ssh-add`).
2. The host needs a directory at the same absolute path as this checkout,
   writable by your SSH user. If the user names differ, create it once on the
   host, for example:
   `sudo mkdir -p /home/<local-user>/Workspace && sudo chown -R <ssh-user>: /home/<local-user>`.
   Use a real directory, not a symlink: the app resolves paths. Alternatively
   store the mirror in any folder with `"root"` (bubblewrap on the host, see
   [One workstation for several machines](#one-workstation-for-several-machines)).
3. On the host: `python3`, `rsync`, `flock` (util-linux), `ffmpeg`, an
   NVIDIA driver for CUDA 12.8.
4. Configure the host in `config/local.json` (local settings, never committed):

   ```json
   {"remote_workers": {"host": "workstation"}}
   ```

   or start the app with `RGB_MOCAP_REMOTE=workstation ./rgbmocap.sh`
   (`RGB_MOCAP_REMOTE=off` forces local workers).
5. Create the mirror and build the runtimes on the host:

   ```sh
   .venv-gui/bin/python -m mocap.remote_workers --init
   .venv-gui/bin/python -m mocap.remote_workers --setup shared exmr
   ```

`--setup` takes the same profiles as `install.sh`, plus `face` for
[face capture](FACE.md). Models stay managed
locally: import or download them in Model Manager here; the first job that
needs them uploads them to the host (the Console shows the upload progress).

### Limits

- Intermediate files from a running remote job (for example tracking previews)
  appear locally when the job finishes, not during it.
- **Resource Safety** monitors RAM on this machine; the worker applies its VRAM
  limits on the host as usual.
- **Help > Check System** reports this machine.
- Projects must live inside the checkout, or under a path that also exists on
  the host.

## Shared project path

Projects store absolute paths: the source video, results, linked folders and
models. A copy of the app and its projects therefore works only at the path the
projects were made under, and remote workers see the same path as the machine
that started them. `path_view` in `config/local.json` gives a copy stored
anywhere that path, without creating it on disk:

```json
{"path_view": {"/home/roth": "~/rgbmocap"}}
```

`rgbmocap.sh` then starts the app in a private mount namespace (bubblewrap)
in which the folder `~/rgbmocap` appears at `/home/roth`. Here that folder holds
`Workspace/toyxyz_RGBMocap/` (the app) and the source videos the projects
refer to, for example `Downloads/`. The GUI, its workers, Blender and the remote
proxy all run inside the view, so new projects record the shared path too.
Everything else, your own home included, looks as usual.

- Map the path the projects were made under or one of its parents. A mapping
  that would hide your own home folder is refused.
- Run `./install.sh` in the folder where the copy is stored, as usual.
- Folders along the shared path are rebuilt for the view with all their
  entries linked back, so `/home/<you>` and other homes stay reachable. A new
  entry created directly in such a folder while the app runs (for example a
  new folder in `/home`) is not kept.
- Fedora Atomic and Bazzite, whose `/home` is a symlink to `/var/home`, are
  handled: `/home` becomes a real folder inside the view.
- Inside the view, files owned by root appear owned by "nobody". ssh rejects
  such files in included system configuration, so remote workers use only
  `~/.ssh/config`; define the host alias there.
- Without bubblewrap, or where user namespaces are disabled, the app does not
  start and reports the error. Remove `path_view` to run without it.

### One workstation for several machines

Clients that use the same shared path can use one mirror on the workstation.
Its `"root"` stores the mirror in any folder there; workers then run in the
same kind of view on the host and still see it at the shared path:

```json
{"path_view": {"/home/roth": "~/rgbmocap"},
 "remote_workers": {"host": "workstation", "root": "/home/forgemaster/rgbmocap"}}
```

The host needs bubblewrap as well. `--init` and `--setup` honor `root`, and
`--setup` builds runtimes inside the view.

All clients of one mirror share its code, models, runtimes and projects, and
its job lock, so jobs from different machines run one after another. Keep the
clients on the same commit, since each job first copies its sender's code. A
project used from two machines syncs through the host and the newer file wins,
so do not work on one project from two machines at once.

To move an existing mirror, move its folder on the host and set `root` on every
client. Runtimes built at the old location keep working when the old location
is the shared path, because the view shows the mirror there; otherwise rebuild
them with `--setup`.
