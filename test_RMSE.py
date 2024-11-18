import pytest

#
# DO NOT MODIFY THIS CODE
#
try:
    import RMSE
except ImportError:
    pytest.fail("Cannot import RMSE. Is RMSE.py present?")


def test_RMSE():
    try:
        RMSE.RMSE
    except TypeError:
        pytest.fail("Is the RMSE class defined?")


def test_RMSE_is_ABC():
    try:
        RMSE.RMSE()
        pytest.fail("Is RMSE an Abstract Base Class?")
    except TypeError:
        pass

