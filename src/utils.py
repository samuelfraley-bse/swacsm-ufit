import subprocess

from . import config as C


def git_hash():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def run_info(seed):
    return dict(seed=seed, git=git_hash(), fs=C.FS, window_s=C.WINDOW_S, overlap=C.OVERLAP,
                edge_trim_s=C.EDGE_TRIM_S, rf_trees=C.RF_TREES)
