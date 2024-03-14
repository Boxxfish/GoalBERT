from typing import *

from goalbert.config import BaseConfig


class ChildConfig(BaseConfig):
    prop1: str = "Test"
    prop2: int = 5


class ParentConfig(BaseConfig):
    child: ChildConfig = ChildConfig()
    prop3: bool = True


def test_single():
    c = ChildConfig()
    assert c.prop1 == "Test"
    assert c.prop2 == 5


def test_nested():
    parent = ParentConfig()
    assert parent.child.prop1 == "Test"
    assert parent.child.prop2 == 5
    assert parent.prop3


def test_no_mut():
    c1 = ChildConfig()
    c2 = ChildConfig()
    c1.prop2 = 10
    assert c1.prop2 == 10
    assert c2.prop2 == 5


def test_flat_dict():
    c = ChildConfig()
    d = c.flat_dict()
    assert d == {
        "prop1": "Test",
        "prop2": 5,
    }


def test_flat_dict_nested():
    parent = ParentConfig()
    d = parent.flat_dict()
    assert d == {
        "prop1": "Test",
        "prop2": 5,
        "prop3": True,
    }
