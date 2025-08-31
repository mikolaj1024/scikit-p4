
import pytest

from skp4.exceptions import SklearnP4Exception
from skp4.metrics import p4_score


def test_empty_lists():
    with pytest.raises(SklearnP4Exception):
        p4_score([], [])


def test_different_lengths():
    with pytest.raises(SklearnP4Exception):
        p4_score([1,2,3], [1,2])


def test_single_label():
    with pytest.raises(SklearnP4Exception):
        p4_score([1], [1])