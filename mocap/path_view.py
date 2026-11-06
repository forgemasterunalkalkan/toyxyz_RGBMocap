"""Show folders at fixed paths inside a private mount namespace (Linux, bubblewrap).

Projects record absolute paths (sources, outputs, models), and remote workers run
at the same path as the GUI. A ``path_view`` in config/local.json lets a copy that
lives anywhere appear at the path its projects were made under::

    {"path_view": {"/home/roth": "~/rgbmocap"}}

maps the folder ``~/rgbmocap`` to ``/home/roth`` for the application and everything
it starts. Nothing is created on disk at the virtual path and other paths keep their
meaning; only processes started inside the view see it. Folders along the virtual
path are rebuilt in memory with every existing entry linked back, so a new top-level
entry created inside such a folder while the app runs (for example directly in
``/home``) is not kept. This also handles hosts whose ``/home`` is a symlink
(Fedora Atomic, Bazzite: ``/home -> var/home``).

Needs bubblewrap and unprivileged user namespaces. Standard library only: it also
runs with the system Python from rgbmocap.sh and on GPU hosts (tools/remote_exec.py).

    python3 mocap/path_view.py enter ROOT -- COMMAND...   # ROOT's configured view
    python3 mocap/path_view.py run --map VIRTUAL=FOLDER [--chdir DIR] [--die-with-parent] -- COMMAND...
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ENV = 'RGB_MOCAP_VIEW'  # Set inside a view (JSON mapping); "{}" means checked, none needed.
INSTALL = 'Install bubblewrap (Debian/Ubuntu: sudo apt install bubblewrap; Fedora: sudo dnf install bubblewrap).'


class ViewError(RuntimeError):
    pass


def is_under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip('/') + '/')


def normalize(mapping) -> dict[str, str]:
    """Validated {virtual: physical}; entries already at their virtual path are dropped."""
    if not isinstance(mapping, dict):
        raise ViewError('path_view must map virtual folders to real ones, e.g. {"/home/roth": "~/rgbmocap"}.')
    out = {}
    for virtual, physical in mapping.items():
        if not isinstance(virtual, str) or not isinstance(physical, str):
            raise ViewError('path_view entries must be folder paths.')
        target = os.path.normpath(virtual)
        if not os.path.isabs(target) or target == '/':
            raise ViewError(f'path_view: {virtual!r} must be an absolute folder other than /.')
        source = os.path.realpath(os.path.expanduser(physical))
        if not os.path.isdir(source):
            raise ViewError(f'path_view: folder {physical} does not exist.')
        if os.path.realpath(target) != source:
            out[target] = source
    for a in out:
        for b in out:
            if a != b and is_under(b, a):
                raise ViewError(f'path_view: {b} lies inside {a}; map only {a}.')
    homes = {os.path.expanduser('~'), os.path.realpath(os.path.expanduser('~'))}
    for target in out:
        if any(is_under(home, target) for home in homes):
            raise ViewError(f'path_view: {target} would hide your home folder.')
    return out


def load(root) -> dict[str, str]:
    """The ``path_view`` of <root>/config/local.json, normalized; {} without one."""
    path = Path(root) / 'config/local.json'
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise ViewError(f'path_view: cannot read {path}: {error}') from None
    value = config.get('path_view') if isinstance(config, dict) else None
    return normalize(value) if value else {}


def to_virtual(path, mapping) -> str:
    """Where ``path`` appears inside the view. The last component is kept as given, so a
    virtual-environment interpreter (a symlink) still starts its environment."""
    path = os.path.abspath(path)
    real = os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))
    for virtual, physical in mapping.items():
        if is_under(real, physical):
            return virtual + real[len(physical):]
    return path


def bwrap_args(mapping) -> list[str]:
    """Mounts for a root where every host entry stays in place and each virtual path shows its folder."""
    args: list[str] = []

    def needed(folder):
        prefix = folder.rstrip('/') + '/'
        return sorted({t[len(prefix):].split('/')[0] for t in mapping if t.startswith(prefix)})

    def expand(folder):
        names = needed(folder)
        if folder != '/':
            args.extend(['--dir', folder])
        host = os.path.realpath(folder) if os.path.isdir(folder) else None
        if host is not None:
            try:
                entries = sorted(os.listdir(host))
            except OSError as error:
                raise ViewError(f'path_view: cannot list {host}: {error}') from None
            for entry in entries:
                if entry in names:
                    continue
                source, dest = os.path.join(host, entry), os.path.join(folder, entry)
                if os.path.islink(source):
                    link = os.readlink(source)
                    if not os.path.isabs(link) and host != folder:
                        link = os.path.normpath(os.path.join(host, link))
                    args.extend(['--symlink', link, dest])
                else:
                    args.extend(['--dev-bind', source, dest])
        for name in names:
            child = os.path.join(folder, name)
            if child in mapping:
                args.extend(['--bind', mapping[child], child])
            else:
                expand(child)

    expand('/')
    return args


def wrap(mapping, argv, cwd=None, die_with_parent=False):
    """(argv, cwd) to run ``argv`` inside the view; unchanged without one."""
    if not mapping:
        return list(argv), cwd
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise ViewError('path_view needs bubblewrap (bwrap). ' + INSTALL)
    command = [bwrap, *bwrap_args(mapping)]
    if die_with_parent:
        command.append('--die-with-parent')
    if cwd:
        command += ['--chdir', str(cwd)]
    command += ['--setenv', ENV, json.dumps(mapping, sort_keys=True), '--', *map(str, argv)]
    return command, '/'


def reenter(root, argv, cwd=None):
    """Re-run ``argv`` inside ROOT's view unless already inside one (returns when none is configured)."""
    if os.environ.get(ENV):
        return
    mapping = load(root)
    if not mapping:
        return
    argv = [to_virtual(a, mapping) if os.path.isabs(str(a)) else str(a) for a in argv]
    command, _ = wrap(mapping, argv, cwd=to_virtual(cwd or os.getcwd(), mapping))
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(command[0], command)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if '--' not in argv or not argv or argv[0] not in ('enter', 'run'):
        print(__doc__, file=sys.stderr)
        return 2
    split = argv.index('--')
    options, command = argv[1:split], argv[split + 1:]
    if not command:
        print('path_view: no command given.', file=sys.stderr)
        return 2
    try:
        if argv[0] == 'enter':
            if len(options) != 1:
                print('path_view: enter takes the application folder.', file=sys.stderr)
                return 2
            reenter(options[0], command)
            os.environ.setdefault(ENV, '{}')  # No view configured: run directly, once.
            os.execvp(command[0], command)
        mapping, cwd, die = {}, None, False
        while options:
            flag = options.pop(0)
            if flag == '--map' and options and '=' in options[0]:
                virtual, physical = options.pop(0).split('=', 1)
                mapping[virtual] = physical
            elif flag == '--chdir' and options:
                cwd = options.pop(0)
            elif flag == '--die-with-parent':
                die = True
            else:
                print(f'path_view: unexpected option {flag}', file=sys.stderr)
                return 2
        command, cwd = wrap(normalize(mapping), command, cwd=cwd, die_with_parent=die)
        if cwd:
            os.chdir(cwd)
        os.execvp(command[0], command)
    except ViewError as error:
        print(error, file=sys.stderr)
        return 70
    except OSError as error:
        print(f'path_view: cannot start {command[0]}: {error}', file=sys.stderr)
        return 70


if __name__ == '__main__':
    sys.exit(main())
