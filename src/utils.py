"""
CustomerIQ — Shared utilities.

Provides:
- Logging setup
- Progress printing
- Safe Parquet I/O
- Timing decorator
"""
import logging
import time
from pathlib import Path
from functools import wraps
from typing import Any, Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root CustomerIQ logger."""
    logger = logging.getLogger("customeriq")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

import sys

# Ensure UTF-8 output encoding on Windows terminals if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_step(step: int, total: int, description: str) -> None:
    """Print a clearly formatted pipeline step."""
    print(f"[{step}/{total}] {description}...")


def print_header(title: str) -> None:
    """Print a section header."""
    border = "=" * 60
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"\n[OK] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"\n[WARN] WARNING: {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"\n[FAIL] ERROR: {message}")


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timeit(func: Callable) -> Callable:
    """Decorator that logs the execution time of a function."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("'%s' completed in %.2f seconds.", func.__name__, elapsed)
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Parquet I/O helpers
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: Path, description: str = "") -> None:
    """Save a DataFrame to Parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    label = description or path.name
    logger.info("Saved %s: %d rows × %d cols → %s", label, len(df), len(df.columns), path)


def load_parquet(path: Path, description: str = "") -> pd.DataFrame:
    """Load a Parquet file and return a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    label = description or path.name
    logger.info("Loaded %s: %d rows × %d cols", label, len(df), len(df.columns))
    return df


def parquet_exists(path: Path) -> bool:
    """Check whether a Parquet output file exists."""
    return path.exists() and path.suffix == ".parquet"


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def safe_divide(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division that returns fill where denominator is zero."""
    return numerator.where(denominator == 0, other=numerator / denominator.replace(0, float("nan"))).fillna(fill)


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Return the memory usage of a DataFrame in megabytes."""
    return df.memory_usage(deep=True).sum() / (1024 ** 2)
