"""Small process helpers shared by Windows and Linux; Windows values are unchanged."""
from __future__ import annotations
import os
import signal
from pathlib import Path

WINDOWS = os.name == 'nt'
# CREATE_NO_WINDOW. POSIX subprocess rejects any non-zero creationflags.
NO_WINDOW = 0x08000000 if WINDOWS else 0


def venv_python(folder) -> Path:
    """Interpreter of a virtual environment or private runtime folder."""
    folder = Path(folder)
    return folder / 'Scripts/python.exe' if WINDOWS else folder / 'bin/python'


def isolate_process(process) -> None:
    """Give a QProcess its own session so cancellation reaches every descendant.

    Windows cancellation already uses taskkill /T; this only applies to POSIX.
    """
    if WINDOWS:
        return
    from PySide6.QtCore import QProcess
    flags = QProcess.UnixProcessFlag.CreateNewSession
    process.setUnixProcessParameters(flags)


def kill_tree(pid: int, sig=None) -> bool:
    """Signal the session started by ``isolate_process`` (POSIX only)."""
    if WINDOWS or not pid:
        return False
    try:
        os.killpg(pid, signal.SIGKILL if sig is None else sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def peak_rss(process=None) -> int:
    """Peak resident memory in bytes: Windows peak working set, Linux VmHWM."""
    import psutil
    process = process or psutil.Process()
    if WINDOWS:
        return process.memory_info().peak_wset
    try:
        with open(f'/proc/{process.pid}/status', encoding='ascii') as status:
            for line in status:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return process.memory_info().rss


def handle_count(process=None) -> int:
    """Open handles (Windows) or file descriptors (POSIX)."""
    import psutil
    process = process or psutil.Process()
    return process.num_handles() if WINDOWS else process.num_fds()
