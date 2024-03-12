"""
Common retrieval metrics.
Each metric expects a set of predicted results, and a set of actual results.
"""

from typing import *

T = TypeVar("T")


def precision(pred: Set[T], actual: Set[T]) -> float:
    """
    What percent of the predictions are in the set of actual items.
    """
    tp = pred.intersection(actual)
    return len(tp) / len(pred)


def recall(pred: Set[T], actual: Set[T]) -> float:
    """
    What percent of the actual items are in the predicted set.
    """
    tp = pred.intersection(actual)
    return len(tp) / len(actual)


def f1(pred: Set[T], actual: Set[T]) -> float:
    """
    Balances both precision and recall.
    """
    p = precision(pred, actual)
    r = recall(pred, actual)
    return (2 * p * r) / (p + r)


def em(pred: Set[T], actual: Set[T]) -> float:
    """
    This is a 1 for exact matches, and 0 otherwise.
    """
    return pred == actual
