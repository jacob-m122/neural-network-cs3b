"""Testing for Asst Five"""
import pytest
import math

from Neurode import MultiLinkNode

#
# DO NOT MODIFY THIS CODE
#

try:
    import FFBPNeurode
except ImportError:
    pytest.fail("Cannot import FFBPNeurode. Is FFBPNeurode.py present?")

try:
    import FFNeurode
except ImportError:
    pytest.fail("Cannot import FFNeurode. Is FFNeurode.py present?")

try:
    import BPNeurode
except ImportError:
    pytest.fail("Cannot import BPNeurode. Is BPNeurode.py present?")


def test_FFBPNeurode_as_child_class():
    my_node = FFBPNeurode.FFBPNeurode()
    assert isinstance(my_node, FFNeurode.FFNeurode), \
        "Is FFBPNeurode inherited from FFNeurode?"
    assert isinstance(my_node, BPNeurode.BPNeurode), \
        "Is FFBPNeurode inherited from BPNeurode?"
