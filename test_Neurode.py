"""Testing for Asst Five"""
import pytest
from enum import Enum

#
# DO NOT MODIFY THIS CODE
#
try:
    import Neurode
except ImportError:
    pytest.fail("Cannot import Neurode. Is Neurode.py present?")


def test_Neurode():
    try:
        Neurode.Neurode()
    except TypeError:
        pytest.fail("Is the Neurode class defined?")

def test_Neurode_as_child_class():
    my_node = Neurode.Neurode()
    assert isinstance(my_node, Neurode.MultiLinkNode), \
        "Is Neurode inherited from MultiLinkNode?"

def test_learning_rate():
    try:
        Neurode.MultiLinkNode._learning_rate
        pytest.fail("_learning_rate should not be defined in "
                    "MultiLinkNode")
    except AttributeError:
        pass
    try:
        Neurode.Neurode._learning_rate
    except TypeError:
        pytest.fail("Is _learning_rate defined as a class attribute in "
                    "the neurode class?")
    assert Neurode.Neurode._learning_rate == .05, \
        ("Is _learning_rate initially set to .05?")
    my_neurode = Neurode.Neurode()
    my_neurode_2 = Neurode.Neurode()
    assert my_neurode.learning_rate == .05, \
        ("Is learning_rate coded as a @property?")
    my_neurode.learning_rate = .06
    assert my_neurode_2._learning_rate == .06, \
        ("Does learning_rate have a setter?")

def test_value():
    my_neurode = Neurode.Neurode()
    assert my_neurode._value == 0, \
        "Is _value initialized to 0 in __init__?"
    try:
        assert my_neurode.value == 0, \
        "value property is not returing the correct value"
    except AttributeError:
        "Is value coded as a @property?"
    try:
        my_neurode.value = 1
        pytest.fail("value should only have a getter, not a setter.")
    except AttributeError:
        pass


def test_weights():
    my_neurode = Neurode.Neurode()
    assert isinstance(my_neurode._weights, dict), \
        "is _weights initialized as an empty dictionary?"
    # create three immutable objects as placeholders for nodes
    item_a = (1,)
    item_b = (2,)
    item_c = (3,)
    node_list_one = [item_a, item_b]
    node_list_two = [item_a, item_b, item_c]
    my_neurode.reset_neighbors(node_list_one, my_neurode.Side.UPSTREAM)
    my_neurode.reset_neighbors(node_list_two, my_neurode.Side.DOWNSTREAM)
    assert len(my_neurode._weights) == 2, \
        "Is there one weight set for each UPSTREAM node?"
    assert my_neurode._weights[item_a] != my_neurode._weights[item_b], \
        "Is a different random weight generated for each neurode?"
    assert 0 < my_neurode._weights[item_a] < 1, \
        "Is each neurode weight generated as a random number between 0 and 1?"
    try:
        my_neurode._weights[item_c]
        pytest.fail("Downstream neurodes should not be assigned weights.")
    except KeyError:
        pass
    assert my_neurode.get_weight(item_b) == my_neurode._weights[item_b], \
        "get_weight is returning an incorrect value."

def test_check_in():
    my_neurode = Neurode.Neurode()
    item_a = Neurode.Neurode()
    item_b = Neurode.Neurode()
    item_c = Neurode.Neurode()
    node_list_one = [item_a, item_b]
    node_list_two = [item_a, item_b, item_c]
    my_neurode.reset_neighbors(node_list_one, my_neurode.Side.UPSTREAM)
    my_neurode.reset_neighbors(node_list_two, my_neurode.Side.DOWNSTREAM)
    assert not my_neurode._check_in(item_a, my_neurode.Side.DOWNSTREAM), \
        "Check in returned True when not all nodes had checked in."
    assert (my_neurode._reporting_nodes[my_neurode.Side.DOWNSTREAM] == 1
            or my_neurode._reporting_nodes[my_neurode.Side.DOWNSTREAM] == 4), \
        "Check in did not correctly adjust _reporting_nodes"
    temp_rep = my_neurode._reporting_nodes[my_neurode.Side.DOWNSTREAM]
    assert not my_neurode._check_in(item_a, my_neurode.Side.DOWNSTREAM), \
        "Check in returned True when not all nodes had checked in."
    assert (temp_rep ==
            my_neurode._reporting_nodes[my_neurode.Side.DOWNSTREAM]), \
        ("_reporting_nodes should not change if a node checks in twice "
         "before all nodes have checked in.")
    assert not my_neurode._check_in(item_b, my_neurode.Side.DOWNSTREAM), \
        "Check in returned True when not all nodes had checked in."
    assert not my_neurode._check_in(item_a, my_neurode.Side.UPSTREAM), \
        "Check in returned True when not all nodes had checked in."
    assert my_neurode._check_in(item_c, my_neurode.Side.DOWNSTREAM), \
        "Check in returned False after all nodes had checked in."
    assert my_neurode._reporting_nodes[my_neurode.Side.DOWNSTREAM] == 0, \
        ("_reporting nodes for a side should be reset to "
         "zero after all nodes have checked in.")
    assert my_neurode._check_in(item_b, my_neurode.Side.UPSTREAM), \
        "Check in returned False after all nodes had checked in."
    assert my_neurode._reporting_nodes[my_neurode.Side.UPSTREAM] == 0, \
        ("_reporting nodes for a side should be reset to "
         "zero after all nodes have checked in.")
