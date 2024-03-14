from typing import *

from goalbert.eval.metrics import *


def test_precision():
    pred = {1, 2, 3, 4, 5}
    actual = {1, 2, 3, 4, 5}
    assert precision(pred, actual) == 1

    pred = {1}
    actual = {1, 2, 3, 4, 5}
    assert precision(pred, actual) == 1

    pred = {1, 2, 3, 4, 5}
    actual = {1}
    assert precision(pred, actual) == 1 / 5


def test_recall():
    pred = {1, 2, 3, 4, 5}
    actual = {1, 2, 3, 4, 5}
    assert recall(pred, actual) == 1

    pred = {1}
    actual = {1, 2, 3, 4, 5}
    assert recall(pred, actual) == 1 / 5

    pred = {1, 2, 3, 4, 5}
    actual = {1}
    assert recall(pred, actual) == 1


def test_f1():
    pred = {1, 2, 3, 4, 5}
    actual = {1, 2, 3, 4, 5}
    assert f1(pred, actual) == 1

    pred = {1}
    actual = {1, 2, 3, 4, 5}
    assert f1(pred, actual) == (2 * 1 * (1 / 5)) / (1 + (1 / 5))

    pred = {1, 2, 3, 4, 5}
    actual = {1}
    assert f1(pred, actual) == (2 * (1 / 5) * 1) / ((1 / 5) + 1)


def test_em():
    pred = {1, 2, 3, 4, 5}
    actual = {1, 2, 3, 4, 5}
    assert em(pred, actual) == 1

    pred = {1}
    actual = {1, 2, 3, 4, 5}
    assert em(pred, actual) == 0

    pred = {1, 2, 3, 4, 5}
    actual = {1}
    assert em(pred, actual) == 0
