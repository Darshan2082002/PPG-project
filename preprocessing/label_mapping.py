"""
label_mapping.py
================
Maps raw MIMIC-IV rhythm strings to integer class labels and provides
utilities to inspect label distributions.

Classes:
  0 – Normal Sinus Rhythm
  1 – Atrial Fibrillation / Flutter
  2 – Bradycardia
  3 – Tachycardia
  4 – Other Arrhythmia
"""

from __future__ import annotations
import numpy as np

CLASS_NAMES = [
    "Normal",
    "Atrial Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Other Arrhythmia",
]

NUM_CLASSES = len(CLASS_NAMES)

# Ordered from most-specific to least-specific keys to avoid greedy mismatch
_RHYTHM_RULES: list[tuple[str, int]] = [
    # Atrial Fibrillation / Flutter – class 1
    ("atrial fib",    1),
    ("atrial flu",    1),
    ("afib",          1),
    (" af ",          1),
    ("a fib",         1),
    # Tachycardia – class 3  (must check before "sinus" for sinus tach)
    ("sinus tach",    3),
    ("tachycardia",   3),
    ("svt",           3),
    ("supraventricular tach", 3),
    ("ventricular tach",      3),
    # Bradycardia – class 2
    ("bradycardia",   2),
    ("sinus brady",   2),
    ("brady",         2),
    # Normal – class 0  (last, most general)
    ("normal sinus",  0),
    ("sinus rhythm",  0),
    ("nsr",           0),
    ("sinus",         0),
]


def map_rhythm(rhythm_str: str | None) -> int:
    """
    Map a raw MIMIC-IV Rhythm string to an integer class.

    Parameters
    ----------
    rhythm_str : str or None

    Returns
    -------
    int  –  class label (0–4)
    """
    if not isinstance(rhythm_str, str):
        return 0  # Unknown/missing → Normal default
    s = rhythm_str.strip().lower()
    if not s:
        return 0

    for key, cls in _RHYTHM_RULES:
        if key in s:
            return cls

    return 4  # Other


def map_rhythm_batch(rhythm_series) -> np.ndarray:
    """
    Vectorised mapping for a pandas Series or list of rhythm strings.

    Returns
    -------
    np.ndarray of int32
    """
    return np.array([map_rhythm(r) for r in rhythm_series], dtype=np.int32)


def class_distribution(labels: np.ndarray | list) -> dict:
    """Return a dict of {class_name: count} for a label array."""
    labels = np.asarray(labels)
    dist = {}
    for i, name in enumerate(CLASS_NAMES):
        dist[name] = int(np.sum(labels == i))
    return dist


def print_class_distribution(labels: np.ndarray | list, title: str = "Label Distribution") -> None:
    """Pretty-print class distribution."""
    dist = class_distribution(labels)
    total = sum(dist.values())
    print(f"\n{'='*40}")
    print(f"  {title}  (total: {total})")
    print(f"{'='*40}")
    for name, cnt in dist.items():
        pct = 100.0 * cnt / total if total > 0 else 0.0
        bar = "█" * int(pct / 2)
        print(f"  {name:<22} {cnt:>6}  ({pct:5.1f}%)  {bar}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    test_cases = [
        "Normal Sinus Rhythm", "Sinus Rhythm", "NSR",
        "Atrial Fibrillation", "AF", "Afib",
        "Bradycardia", "Sinus Brady",
        "Sinus Tach", "Tachycardia", "SVT",
        "Ventricular Fibrillation", None, "",
    ]
    for t in test_cases:
        print(f"  {str(t):<35} → {map_rhythm(t)} ({CLASS_NAMES[map_rhythm(t)]})")
