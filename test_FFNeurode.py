"""Testing for Asst Five"""
import pytest
import math

#
# DO NOT MODIFY THIS CODE
#
try:
    import FFNeurode
except ImportError:
    pytest.fail("Cannot import FFNeurode. Is FFNeurode.py present?")

try:
    import Neurode
except ImportError:
    pytest.fail("Cannot import Neurode. Is Neurode.py present?")


def test_FFNeurode():
    try:
        FFNeurode.FFNeurode()
    except TypeError:
        pytest.fail("Is the FFNeurode class defined?")


def test_FFNeurode_as_child_class():
    my_node = FFNeurode.FFNeurode()
    assert isinstance(my_node, Neurode.Neurode), \
        "Is FFNeurode inherited from Neurode?"


def test_sigmoid():
    try:
        FFNeurode.FFNeurode._sigmoid(.2)
    except TypeError:
        pytest.fail("Is _sigmoid defined as a static method in "
                    "the FFNeurode class?")
    assert math.isclose(FFNeurode.FFNeurode._sigmoid(.2), 0.549833997312478), \
        "_sigmoid is returning incorrect value."


def test_calculate_value():
    central_neurode = FFNeurode.FFNeurode()
    input_neurodes = [FFNeurode.FFNeurode() for _ in range(3)]
    central_neurode.reset_neighbors(input_neurodes,
                                    Neurode.MultiLinkNode.Side.UPSTREAM)
    input_neurodes[0]._value = .1
    input_neurodes[1]._value = .2
    input_neurodes[2]._value = .3
    central_neurode._weights[input_neurodes[0]] = .4
    central_neurode._weights[input_neurodes[1]] = .5
    central_neurode._weights[input_neurodes[2]] = .6
    central_neurode._calculate_value()
    assert math.isclose(central_neurode.value, .5793242521487495), \
        "_calculate_value is saving an incorrect value."


def test_calculate_value_inputs_zero():
    central_neurode = FFNeurode.FFNeurode()
    input_neurodes = [FFNeurode.FFNeurode() for _ in range(3)]
    central_neurode.reset_neighbors(input_neurodes,
                                    Neurode.MultiLinkNode.Side.UPSTREAM)
    input_neurodes[0]._value = 0
    input_neurodes[1]._value = 0
    input_neurodes[2]._value = 0
    central_neurode._weights[input_neurodes[0]] = .4
    central_neurode._weights[input_neurodes[1]] = .5
    central_neurode._weights[input_neurodes[2]] = .6
    central_neurode._calculate_value()
    assert math.isclose(central_neurode.value, .5), \
        ("_calculate_value is saving an incorrect value when all "
         "inputs are zero.")


def test_fire_downstream():
    class FakeDownstreamNode:
        def __init__(self):
            self._caught = False
            self._calling_node = None

        def data_ready_upstream(self, node):
            self._caught = True
            self._calling_node = node

    central_node = FFNeurode.FFNeurode()
    ds_node_zero = FakeDownstreamNode()
    ds_node_one = FakeDownstreamNode()
    central_node.reset_neighbors([ds_node_zero, ds_node_one],
                                 Neurode.MultiLinkNode.Side.DOWNSTREAM)
    central_node._fire_downstream()
    assert ds_node_zero._caught and ds_node_one._caught, \
        ("_fire_downstream() is not correctly alerting all downstream "
         "nodes that data is ready.")
    assert ds_node_zero._calling_node == central_node, \
        ("_fire_downstream() is not correctly identifying itself to "
         "the downstream node (node value is not correct.)")
    assert ds_node_one._calling_node == central_node, \
        ("_fire_downstream() is not correctly identifying itself to "
         "the downstream node (node value is not correct.)")


def test_data_ready_upstream():
    class FakeDownstreamNode:
        def __init__(self):
            self._caught = False
            self._calling_node = None

        def data_ready_upstream(self, node):
            self._caught = True
            self._calling_node = node

    central_neurode = FFNeurode.FFNeurode()
    input_neurodes = [FFNeurode.FFNeurode() for _ in range(2)]
    central_neurode.reset_neighbors(input_neurodes,
                                    Neurode.MultiLinkNode.Side.UPSTREAM)
    ds_node_zero = FakeDownstreamNode()
    central_neurode.reset_neighbors([ds_node_zero],
                                    Neurode.MultiLinkNode.Side.DOWNSTREAM)
    central_neurode.data_ready_upstream(input_neurodes[0])
    assert not ds_node_zero._caught, \
        ("data_ready_upstream is firing downstream before all nodes"
         "have checked in.")
    central_neurode.data_ready_upstream(input_neurodes[1])
    assert ds_node_zero._caught, \
        ("data_ready_upstream is not firing downstream after all nodes"
         "have checked in.")


def test_set_input():
    class FakeDownstreamNode:
        def __init__(self):
            self._caught = False
            self._calling_node = None

        def data_ready_upstream(self, node):
            self._caught = True
            self._calling_node = node

    central_node = FFNeurode.FFNeurode()
    ds_node_zero = FakeDownstreamNode()
    ds_node_one = FakeDownstreamNode()
    central_node.reset_neighbors([ds_node_zero, ds_node_one],
                                 Neurode.MultiLinkNode.Side.DOWNSTREAM)
    central_node.set_input(.4)
    assert central_node.value == .4, \
        "Node value is not correct after set_input()."
    assert ds_node_zero._caught and ds_node_one._caught, \
        ("set_input() is not correctly alerting all downstream "
         "nodes that data is ready.")
    assert ds_node_zero._calling_node == central_node, \
        ("set_input() is not correctly identifying itself to "
         "the downstream node (node value is not correct.)")
    assert ds_node_one._calling_node == central_node, \
        ("set_input() is not correctly identifying itself to "
         "the downstream node (node value is not correct.)")
