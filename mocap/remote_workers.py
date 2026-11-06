"""Linux worker commands, optionally running GPU stages on a workstation over SSH.

Local Linux runtimes use the developer layout (``.venv*/bin/python``) built by
tools/setup_linux.py from installer/runtime-lock-linux.json.

With a remote host configured, every stage in REMOTE_STAGES starts this module
as a proxy instead of the worker. The proxy:

1. mirrors the source code and ``models/`` to the host. The worker there sees the
   mirror at the *same absolute path* as this checkout, so commands, receipts,
   cache identities and @@MOCAP@@ events need no path translation. By default the
   mirror is stored at that path; with ``"root"`` it is stored elsewhere and the
   worker runs in a private mount namespace that shows it at this checkout's path
   (mocap/path_view.py), so several clients can share one mirror;
2. reconciles the job's project folders with the host against the state of the
   last sync - changes on either side are copied, deletions are propagated and,
   when both sides changed a file, the newer one wins - so outputs of a
   cancelled job are kept and local deletions are not undone;
3. runs the identical worker command through tools/remote_exec.py, relaying
   its output unchanged; closing the connection stops the remote worker;
4. reconciles the project folders again so the results are local before the
   GUI starts its next job.

Everything else (GUI, video import, contact correction, Blender, export) stays
local. Configure with ``config/local.json``::

    {"remote_workers": {"host": "workstation", "root": "/home/me/rgbmocap"}}

or ``RGB_MOCAP_REMOTE=<ssh host>`` (``off`` disables). Without ``root`` the host
needs the same absolute path as this checkout. Prepare it with
``python3 -m mocap.remote_workers --init``.
"""
from __future__ import annotations
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import path_view
from .paths import HOME, ROOT
from .runtime_registry import PROFILES, profile_for

REMOTE_STAGES = frozenset({
    'track', 'people', 'person', 'features', 'gvhmr', 'pva', 'refine', 'footmr', 'handflow',
    'flow_camera', 'flow_features', 'flowhmr',
    'exmr_preprocess', 'exmr_features', 'exmr',
    'flow_batch', 'exmr_batch', 'gvhmr_batch',
    'face',
})
# Worker-only caches inside a project, rebuilt on the host from input.mp4.
HOST_ONLY = frozenset({'exmr_frames'})
FORWARD_ENV = ('PYTHONIOENCODING', 'PYTHONUNBUFFERED', 'PYTHONUTF8', 'PYTHONDONTWRITEBYTECODE',
               'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME')
MARKER = '.setup/remote-mirror.json'
LOCK = HOME / 'cache/remote_jobs/worker.lock'
STATE = HOME / 'cache/remote_sync'


# ---------------------------------------------------------------- commands

def settings() -> dict:
    value = os.environ.get('RGB_MOCAP_REMOTE')
    if value is not None:
        value = value.strip()
        return {'host': value} if value and value.lower() not in ('0', 'off', 'none', 'local') else {}
    try:
        from .resource_safety import path, read_settings
        config = read_settings(path()).get('remote_workers') or {}
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) and isinstance(config.get('host'), str) and config['host'] else {}


def remote_stages(config) -> frozenset:
    stages = config.get('stages')
    return frozenset(stages) if isinstance(stages, list) else REMOTE_STAGES


def local_command(root, stage, args, check=True) -> list[str]:
    from . import face_runtime
    if stage in face_runtime.STAGES:
        return face_runtime.command(root, stage, args, check=check)
    key = profile_for(stage, args)
    python = Path(root) / PROFILES[key]['dev'] / 'bin/python'
    if check and not python.is_file():
        raise RuntimeError(f'The {key} runtime is missing. Run ./install.sh to prepare it.')
    return [str(python), '-X', 'utf8', '-m', 'mocap.worker_progress', stage, *map(str, args)]


def command(root, stage, args) -> list[str]:
    config = settings()
    if config.get('host') and stage in remote_stages(config):
        return [sys.executable, '-X', 'utf8', '-m', 'mocap.remote_workers', stage, *map(str, args)]
    return local_command(root, stage, args)


# ---------------------------------------------------------------- transport

class RemoteError(RuntimeError):
    pass


def say(message):
    print(message, flush=True)


class Link:
    """One multiplexed SSH connection; every ssh/rsync call reuses it.

    ``root`` is where the mirror is stored on the host (default: this checkout's path).
    """

    def __init__(self, host, root=None):
        self.host = host
        if root is not None and not (isinstance(root, str) and root.startswith('/')):
            raise RemoteError('remote_workers.root must be an absolute path on the host.')
        self.root = Path(os.path.normpath(root)) if root else ROOT
        runtime = os.environ.get('XDG_RUNTIME_DIR') or tempfile.gettempdir()
        self.control = str(Path(runtime) / 'rgbmocap-ssh-%C')
        self.options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                        '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
                        '-o', 'ControlPath=' + self.control]
        if os.environ.get(path_view.ENV, '{}') != '{}':
            # Inside a path view root-owned files appear owned by "nobody", and ssh rejects the
            # system configuration's Include files for that. Read only the user's own config.
            user = Path.home() / '.ssh/config'
            self.options = ['-F', str(user) if user.is_file() else '/dev/null', *self.options]

    def open(self):
        # Start (or reuse) a background master with its stdio detached, so it never
        # holds the GUI's output pipes open after this proxy exits.
        check = subprocess.run(['ssh', *self.options, '-O', 'check', self.host],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode == 0:
            return
        # A file, not a pipe: the backgrounded master would keep a pipe open for minutes.
        with tempfile.TemporaryFile() as errors:
            start = subprocess.run(['ssh', *self.options, '-o', 'ControlMaster=yes', '-o', 'ControlPersist=300',
                                    '-f', '-N', self.host], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=errors)
            errors.seek(0)
            message = errors.read().decode('utf-8', 'replace')
        if start.returncode:
            detail = message.strip().splitlines()[-1:] or ['no response']
            raise RemoteError(f'Cannot connect to {self.host}: {detail[0]}. '
                              'Check that ssh-agent holds your key (ssh-add).')

    @property
    def view(self):
        """Path view for the remote worker: this checkout's path shows the stored mirror."""
        return {str(ROOT): str(self.root)} if self.root != ROOT else None

    def path(self, local) -> Path:
        """Where a path of this checkout is stored on the host (other paths are unchanged)."""
        local = Path(local)
        if self.root != ROOT and local.is_relative_to(ROOT):
            return self.root / local.relative_to(ROOT)
        return local

    def remote(self, local) -> str:
        return f'{self.host}:{self.path(local)}'

    def ssh(self, *remote, **kwargs):
        return ['ssh', *self.options, '-o', 'ControlMaster=no', '-T', self.host, *remote]

    def run(self, script, *args, input=None):
        """Run a shell command line on the host and return its stdout."""
        line = script + ''.join(' ' + shlex.quote(str(a)) for a in args)
        done = subprocess.run(self.ssh(line), input=input, capture_output=True, text=True)
        if done.returncode:
            raise RemoteError(f'{self.host}: {(done.stderr or done.stdout).strip()[-600:]}')
        return done.stdout

    def rsync(self, *args, progress=None, names=None, what='File'):
        """rsync over the shared connection; ``names`` feeds --files-from=-."""
        rsh = ' '.join(shlex.quote(a) for a in ['ssh', *self.options, '-o', 'ControlMaster=no'])
        cmd = ['rsync', '-a', '--mkpath', '-e', rsh, *args]
        if progress is None:
            listing = None if names is None else ''.join(n + '\n' for n in names)
            done = subprocess.run(cmd, input=listing, capture_output=True, text=True)
            if done.returncode:
                raise RemoteError(f'{what} transfer failed: ' + (done.stderr.strip()[-600:] or f'rsync {done.returncode}'))
            return done.stdout
        return stream_progress(cmd + ['--info=progress2', '--no-inc-recursive'], progress)


def stream_progress(cmd, label):
    """rsync with a progress line every few seconds (keeps the GUI stall monitor informed)."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    last = 0.0
    buffer = b''
    while True:
        chunk = process.stdout.read1(4096) if hasattr(process.stdout, 'read1') else process.stdout.read(4096)
        if not chunk:
            break
        buffer = (buffer + chunk)[-4096:]
        parts = buffer.replace(b'\r', b'\n').split(b'\n')
        buffer = parts.pop()
        if time.monotonic() - last > 5 and parts:
            text = parts[-1].decode('utf-8', 'replace').split()
            if len(text) >= 2 and text[1].endswith('%'):
                say(f'{label}: {text[1]} ({text[0]} bytes, {text[2] if len(text) > 2 else ""})')
                last = time.monotonic()
    errors = process.stderr.read().decode('utf-8', 'replace')
    if process.wait():
        raise RemoteError('File transfer failed: ' + (errors.strip()[-600:] or f'rsync {process.returncode}'))


# ---------------------------------------------------------------- mirroring

def code_files() -> list[str]:
    listed = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            capture_output=True, check=True).stdout
    return sorted({name for name in listed.decode('utf-8').split('\0') if name and (ROOT / name).is_file()})


def state_file(name) -> Path:
    STATE.mkdir(parents=True, exist_ok=True)
    return STATE / name


def push_code(link):
    """Tracked and new source files; files removed locally since the last push are removed remotely."""
    files = code_files()
    key = link.host if link.root == ROOT else f'{link.host}\0{link.root}'
    previous_path = state_file(hashlib.sha1(key.encode()).hexdigest()[:16] + '-code.json')
    try:
        previous = set(json.loads(previous_path.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        previous = set()
    removed = sorted(previous - set(files))
    link.rsync('--files-from=-', '--delete-missing-args', str(ROOT) + '/', link.remote(ROOT) + '/',
               names=files + removed, what='Code')
    previous_path.write_text(json.dumps(files), encoding='utf-8')


def push_models(link):
    models = HOME / 'models'
    if not models.is_dir():
        return
    plan = link.rsync('--dry-run', '--out-format=%l %n', '--exclude=*.part', '--exclude=*.partial',
                      str(models) + '/', link.remote(models) + '/')
    sizes = [int(line.split(' ', 1)[0]) for line in plan.splitlines() if line and not line.endswith('/')
             and line.split(' ', 1)[0].isdigit()]
    if not sizes:
        return
    say(f'Uploading {len(sizes)} model file(s), {sum(sizes) / 2**30:.2f} GiB, to {link.host}...')
    link.rsync('--exclude=*.part', '--exclude=*.partial', str(models) + '/', link.remote(models) + '/',
               progress='Uploading models')


def scan(folder: Path) -> dict:
    found = {}
    if not folder.is_dir():
        return found
    for base, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in HOST_ONLY]
        for name in names:
            path = Path(base, name)
            try:
                value = path.lstat()
            except OSError:
                continue
            if path.is_symlink():
                continue
            found[path.relative_to(folder).as_posix()] = [value.st_size, value.st_mtime_ns]
    return found


REMOTE_SCAN = r'''
import json,os,sys
skip=set(json.loads(sys.argv[1]));out={}
for folder in sys.argv[2:]:
    if not os.path.isdir(folder):out[folder]=None;continue
    found=out[folder]={}
    for base,dirs,names in os.walk(folder):
        dirs[:]=[d for d in dirs if d not in skip]
        for name in names:
            path=os.path.join(base,name)
            if os.path.islink(path):continue
            try:value=os.lstat(path)
            except OSError:continue
            found[os.path.relpath(path,folder)]=[value.st_size,value.st_mtime_ns]
print(json.dumps(out))
'''


def remote_scan(link, folders) -> dict:
    stored = {str(link.path(folder)): Path(folder) for folder in folders}
    out = link.run('python3 -c ' + shlex.quote(REMOTE_SCAN), json.dumps(sorted(HOST_ONLY)), *stored)
    return {stored[k]: (v or {}) for k, v in json.loads(out).items()}


def plan_sync(local, remote, base):
    push, pull, conflicts = [], [], []
    merged = {}
    for rel in sorted(set(local) | set(remote) | set(base)):
        mine, theirs, before = local.get(rel), remote.get(rel), base.get(rel)
        if mine == theirs:
            if mine is not None:
                merged[rel] = mine
            continue
        mine_changed, theirs_changed = mine != before, theirs != before
        if mine_changed and not theirs_changed:
            winner = 'push'
        elif theirs_changed and not mine_changed:
            winner = 'pull'
        else:
            if mine is not None and theirs is not None and before is not None:
                conflicts.append(rel)
            winner = ('pull' if mine is None else 'push' if theirs is None
                      else 'push' if mine[1] >= theirs[1] else 'pull')
        (push if winner == 'push' else pull).append(rel)
        value = mine if winner == 'push' else theirs
        if value is not None:
            merged[rel] = value
    return push, pull, conflicts, merged


def sync_folders(link, folders):
    """Two-way reconcile of project folders with the host (see module docstring)."""
    if not folders:
        return
    remote = remote_scan(link, folders)
    wait = f'flock -w 300 {shlex.quote(str(link.path(LOCK)))} rsync'
    for folder in folders:
        record = state_file(hashlib.sha1(f'{link.host}\0{folder}'.encode()).hexdigest() + '.json')
        try:
            base = json.loads(record.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            base = {}
        push, pull, conflicts, merged = plan_sync(scan(folder), remote.get(folder, {}), base)
        for rel in conflicts:
            say(f'Remote sync: {folder.name}/{rel} changed on both machines; kept the newer copy.')
        for names, source, target in ((push, f'{folder}/', link.remote(folder) + '/'),
                                      (pull, link.remote(folder) + '/', f'{folder}/')):
            if not names:
                continue
            link.rsync('--rsync-path', wait, '--files-from=-', '--delete-missing-args', source, target,
                       names=names, what='Project')
        record.write_text(json.dumps(merged), encoding='utf-8')


def workspace_root(folder: Path) -> Path:
    """The project workspace a model folder belongs to (see project_workspace.root_for)."""
    try:
        state = json.loads((folder / 'project.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return folder
    relative = state.get('workspace_root') if state.get('workspace_version') is None else None
    if not isinstance(relative, str) or not relative:
        return folder
    root = (folder / relative).resolve()
    if root != folder and folder.is_relative_to(root) and (root / 'project.json').is_file():
        return root
    return folder


def job_paths(args):
    """Project folders and request files a worker invocation reads or writes."""
    def value(flag):
        return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None
    folders, files = [], []
    if value('--project'):
        folders.append(Path(value('--project')).resolve())
    if value('--request'):
        request = Path(value('--request')).resolve()
        files.append(request)
        try:
            data = json.loads(request.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            data = {}
        if isinstance(data.get('root'), str) and data['root']:
            folders.append(Path(data['root']).resolve())
        for job in data.get('jobs') or []:
            if isinstance(job, dict) and isinstance(job.get('project'), str):
                folders.append(Path(job['project']).resolve())
    folders = [workspace_root(folder) for folder in folders]
    unique = []
    for folder in sorted(set(folders), key=lambda p: len(p.parts)):
        if not any(folder.is_relative_to(parent) for parent in unique):
            unique.append(folder)
    return unique, files


# ---------------------------------------------------------------- proxy

def remote_env():
    env = {k: v for k, v in os.environ.items()
           if k.startswith('RGB_MOCAP_') and k not in ('RGB_MOCAP_REMOTE', path_view.ENV)}
    env.update({k: os.environ[k] for k in FORWARD_ENV if k in os.environ})
    env['RGB_MOCAP_DATA'] = str(HOME)
    env['RGB_MOCAP_REMOTE'] = 'off'  # A remote worker never proxies again.
    return env


def check_mirror(link, python):
    # The interpreter is a symlink that may resolve only inside the worker's path view.
    stored = shlex.quote(str(link.path(python)))
    status = link.run('test -f ' + shlex.quote(str(link.path(ROOT / MARKER))) + ' && echo mirror; '
                      f'{{ test -L {stored} || test -x {stored}; }} && echo runtime; '
                      + ('command -v bwrap >/dev/null && echo bwrap; ' if link.view else '')
                      + 'mkdir -p ' + shlex.quote(str(link.path(LOCK.parent))) + '; true')
    if 'mirror' not in status:
        raise RemoteError(f'{link.host} has no RGB MoCap mirror at {link.path(ROOT)}. '
                          'Run: python3 -m mocap.remote_workers --init')
    if 'runtime' not in status:
        raise RemoteError(f'{link.host} lacks the runtime {python.parent.parent.name}. '
                          'Run: python3 -m mocap.remote_workers --setup')
    if link.view and 'bwrap' not in status:
        raise RemoteError(f'{link.host} needs bubblewrap to run workers from {link.root}. ' + path_view.INSTALL)


def proxy(stage, args) -> int:
    config = settings()
    host = config.get('host')
    if not host:
        raise RemoteError('No remote host configured.')
    worker = local_command(ROOT, stage, args, check=False)
    link = Link(host, config.get('root'))
    folders, files = job_paths(args)
    synced = False
    say(f'Remote worker: {stage} on {host}.')
    try:
        link.open()
        check_mirror(link, Path(worker[0]))
        push_code(link)
        push_models(link)
        for file in files:
            link.rsync(str(file), link.remote(file))
        sync_folders(link, folders)
        synced = True
        spec = dict(command=worker, cwd=str(ROOT), env=remote_env(), lock=str(link.path(LOCK)),
                    lock_wait=float(config.get('lock_wait', 120)), view=link.view)
        runner = subprocess.Popen(link.ssh('python3', '-u', shlex.quote(str(link.path(ROOT / 'tools/remote_exec.py')))),
                                  stdin=subprocess.PIPE)
        try:
            runner.stdin.write((json.dumps(spec) + '\n').encode('utf-8'))
            runner.stdin.flush()
            code = runner.wait()
        finally:
            if runner.poll() is None:
                runner.stdin.close()
                runner.wait()
        if code == 255:
            say(f'Lost the connection to {host}; the remote job was stopped.')
        return code
    finally:
        if synced:
            try:
                sync_folders(link, folders)
            except (RemoteError, OSError) as error:
                say(f'Could not copy results back from {host}: {error}')
                raise
            for file in files:
                link.run('rm -f', str(link.path(file)))


def init_mirror(link):
    """Create the mirror directory on the host and copy the source there."""
    link.open()
    try:
        link.run('mkdir -p', str(link.root))
    except RemoteError as error:
        hint = (f'The host needs the same absolute path as this checkout ({ROOT}). Create it once with sudo '
                'and give your user ownership, or store the mirror elsewhere with remote_workers.root.'
                if link.root == ROOT else 'Choose a writable remote_workers.root.')
        raise RemoteError(f'{error}\n{hint}') from None
    push_code(link)
    marker = json.dumps(dict(version=1, source=str(ROOT), root=str(link.root),
                             created=time.strftime('%Y-%m-%dT%H:%M:%S')))
    link.run(f'mkdir -p {shlex.quote(str(link.path(ROOT / ".setup")))} && '
             f'cat > {shlex.quote(str(link.path(ROOT / MARKER)))}', input=marker)
    say(f'Mirror ready at {link.host}:{link.root}' + (f' (workers see it at {ROOT})' if link.view else ''))


def setup_command(link, profiles) -> list[str]:
    """Build runtimes on the host where the workers will see them (inside their path view)."""
    build = ['python3', '-u', 'tools/setup_linux.py', '--profiles', *profiles]
    if not link.view:
        return link.ssh('cd', shlex.quote(str(ROOT)), '&&', *build)
    return link.ssh('python3', '-I', shlex.quote(str(link.path(ROOT / 'mocap/path_view.py'))), 'run',
                    '--map', shlex.quote(f'{ROOT}={link.root}'), '--chdir', shlex.quote(str(ROOT)), '--', *build)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        # Run where the GUI runs: inside the configured path view (no-op inside it or without one).
        path_view.reenter(ROOT, [sys.executable, '-X', 'utf8', '-m', 'mocap.remote_workers', *argv], cwd=ROOT)
    except path_view.ViewError as error:
        print(error, file=sys.stderr, flush=True)
        return 3
    if argv and argv[0] in ('--init', '--setup'):
        import argparse
        parser = argparse.ArgumentParser(prog='python3 -m mocap.remote_workers')
        parser.add_argument('--init', action='store_true', help='Create or refresh the mirror on the host.')
        parser.add_argument('--setup', nargs='*', metavar='PROFILE',
                            help='Build runtimes on the host (default: shared exmr).')
        parser.add_argument('--host', help='SSH host; defaults to the configured one.')
        parser.add_argument('--root', help='Mirror folder on the host; defaults to remote_workers.root, '
                                           'else this checkout\'s path.')
        options = parser.parse_args(argv)
        host = options.host or settings().get('host')
        if not host:
            parser.error('No host: pass --host or configure remote_workers.host in config/local.json.')
        try:
            link = Link(host, options.root or settings().get('root'))
            if options.init:
                init_mirror(link)
            if options.setup is not None:
                link.open()
                push_code(link)
                return subprocess.run(setup_command(link, options.setup or ['shared', 'exmr'])).returncode
        except RemoteError as error:
            print(f'Remote worker setup failed: {error}', file=sys.stderr, flush=True)
            return 3
        return 0
    if not argv:
        print(__doc__)
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(129))
    try:
        return proxy(argv[0], argv[1:])
    except RemoteError as error:
        print(f'Remote worker failed: {error}', file=sys.stderr, flush=True)
        return 3


if __name__ == '__main__':
    sys.exit(main())
