"""Agreement statistics between two sets of labels or two system orderings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confusion_matrix(reference: Sequence[int], predicted: Sequence[int], size: int) -> np.ndarray:
    """Counts with reference grades as rows and predicted grades as columns."""
    matrix = np.zeros((size, size), dtype=int)
    for truth, guess in zip(reference, predicted, strict=True):
        matrix[truth, guess] += 1
    return matrix


def cohen_kappa(matrix: np.ndarray, *, weighted: bool = False) -> float:
    """Cohen's kappa from a confusion matrix; ``weighted`` uses linear grade distance."""
    matrix = np.asarray(matrix, dtype=float)
    total = matrix.sum()
    if total == 0:
        raise ValueError("confusion matrix is empty")
    size = len(matrix)
    index = np.arange(size)
    disagreement = (
        np.abs(index[:, None] - index[None, :]) / max(size - 1, 1)
        if weighted
        else (index[:, None] != index[None, :]).astype(float)
    )
    observed = float((disagreement * matrix).sum() / total)
    expected = float(
        (disagreement * np.outer(matrix.sum(axis=1), matrix.sum(axis=0))).sum() / total**2
    )
    return 1.0 - observed / expected if expected > 0 else 1.0


def kendall_tau(left: Sequence[float], right: Sequence[float]) -> float:
    """Kendall's tau-a between two score lists over the same systems."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("need two equally long lists with at least two entries")
    concordant = 0
    count = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            concordant += int(np.sign(left[i] - left[j]) * np.sign(right[i] - right[j]))
            count += 1
    return concordant / count
