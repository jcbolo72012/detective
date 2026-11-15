"""Paths and settings shared by the whole app. Everything that reads a file or a key goes through here."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository folder
DATA_DIR = os.path.join(ROOT, "data")                                  # Giovanni case-file CSVs
CATALOG_DIR = os.path.join(DATA_DIR, "catalog")                        # catalogs and embedding vectors
ENV_FILE = os.path.join(ROOT, ".env")


def load_env(path=ENV_FILE):
    """Read KEY=value lines from .env into the environment. Real environment variables win; blanks are ignored."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v


def env(name, default=""):
    """A setting from the environment (loaded from .env first)."""
    load_env()
    return os.environ.get(name, default)


load_env()
