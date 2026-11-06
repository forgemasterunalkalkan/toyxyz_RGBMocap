"""Remote side of mocap.remote_workers: run one worker, stop it when the SSH link closes.

Started as ``python3 tools/remote_exec.py`` on the GPU host with the job spec as
the first stdin line (JSON: command, cwd, env, lock, lock_wait, view). The worker's
stdout/stderr go straight back through SSH. When stdin reaches EOF - the GUI
cancelled, the laptop proxy died or the connection dropped - the worker's whole
process group is stopped, so no GPU job keeps running unattended. With ``view``
({client path: mirror folder}) the worker runs inside mocap/path_view.py's mount
namespace, so it sees the mirror at the client's path.

Standard library only; runs with the host's system Python 3.
"""
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time


def stop(pid, grace=5.0):
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 0)):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.1)


def acquire(path, wait):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, 'a+')
    deadline = time.monotonic() + wait
    announced = False
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if not announced:
                print('Waiting for the previous remote job to stop...', flush=True)
                announced = True
            if time.monotonic() > deadline:
                handle.close()
                return None
            time.sleep(0.5)


def read_spec():
    # Raw descriptor reads: a buffered stdin shared with the watcher thread can abort
    # interpreter shutdown while that thread is blocked in read().
    data = b''
    while not data.endswith(b'\n'):
        chunk = os.read(0, 65536)
        if not chunk:
            break
        data += chunk
    return json.loads(data)


def path_view():
    """mocap/path_view.py of this mirror, loaded without importing the mocap package."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mocap', 'path_view.py')
    spec = importlib.util.spec_from_file_location('rgbmocap_path_view', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    spec = read_spec()
    command, cwd = spec['command'], spec['cwd']
    if spec.get('view'):
        view = path_view()
        try:
            command, cwd = view.wrap(view.normalize(spec['view']), command, cwd=cwd, die_with_parent=True)
        except view.ViewError as error:
            print(f'Remote worker: {error}', file=sys.stderr, flush=True)
            return 70
    lock = acquire(spec['lock'], float(spec.get('lock_wait', 120)))
    if lock is None:
        print('The GPU host is still busy with another RGB MoCap job.', file=sys.stderr, flush=True)
        return 75
    env = os.environ.copy()
    env.update(spec.get('env', {}))
    child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                             start_new_session=True)

    def watch():
        try:
            while os.read(0, 65536):
                pass
        finally:
            if child.poll() is None:
                stop(child.pid)

    threading.Thread(target=watch, daemon=True).start()
    code = child.wait()
    stop(child.pid, grace=1.0)  # Reap descendants such as FFmpeg that outlived the worker.
    lock.close()
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)  # Do not wait for the watcher thread blocked on stdin.
