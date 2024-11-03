"""Testing for Asst Five"""
import pytest
import math

from Neurode import MultiLinkNode

#
# DO NOT MODIFY THIS CODE
#

try:
    import BPNeurode
except ImportError:
    pytest.fail("Cannot import BPNeurode. Is BPNeurode.py present?")

try:
    import Neurode
except:
    pytest.fail("Cannot import Neurode. Is Neurode.py present?")

def test_BPNeurode():
    try:
        BPNeurode.BPNeurode()
    except TypeError:
        pytest.fail("Is the BPNeurode class defined?")


def test_BPNeurode_as_child_class():
    my_node = BPNeurode.BPNeurode()
    assert isinstance(my_node, Neurode.Neurode), \
        "Is BPNeurode inherited from Neurode?"


def test_delta_property():
    my_node = BPNeurode.BPNeurode()
    my_node._delta = 10
    try:
        assert my_node.delta == 10, \
            "delta property is returning an incorrect value"
    except AttributeError:
        pytest.fail("Is there a @property method for delta?")


def test_sigmoid_derivative():
    try:
        BPNeurode.BPNeurode._sigmoid_derivative(.5)
    except AttributeError:
        pytest.fail("Is _sigmoid_derivative defined as a @staticmethod?")
    assert math.isclose(BPNeurode.BPNeurode._sigmoid_derivative(.1), .09), \
        ("_sigmoid_derivative is returning an incorrect value. This "
         "calculation may be simpler than it first appears.")


def test_calculate_delta_output_node():
    my_node = BPNeurode.BPNeurode()
    my_node._value = .6
    my_node._calculate_delta(.4)
    assert math.isclose(my_node.delta, -.048), \
        ("_caculate_delta is storing an incorrect delta when called with an "
         "argument")


def test_calculate_delta_hidden_node():
    my_node = BPNeurode.BPNeurode()
    ds_nodes = [BPNeurode.BPNeurode() for _ in range(3)]
    my_node.reset_neighbors(ds_nodes, BPNeurode.BPNeurode.Side.DOWNSTREAM)
    for node in ds_nodes:
        node.reset_neighbors([my_node], BPNeurode.BPNeurode.Side.UPSTREAM)
    ds_nodes[0]._delta = .3
    ds_nodes[1]._delta = .4
    ds_nodes[2]._delta = .5
    ds_nodes[0]._weights[my_node] = .6
    ds_nodes[1]._weights[my_node] = .7
    ds_nodes[2]._weights[my_node] = .8
    my_node._value = .6
    my_node._calculate_delta()
    assert math.isclose(my_node.delta, .2064), \
        ("_caculate_delta is storing an incorrect delta when called with no"
         "argument.")

def test_set_expected():
    class patched_BPNeurode(BPNeurode.BPNeurode):
        def __init__(self):
            self.delta_called = False
            #self.fire_called = False
            self.arg_passed = False
            self.order_correct = True
            super().__init__()

        def _calculate_delta(self, expected_value=0):
            self.delta_called = True
            #if self.fire_called:
            #    self.order_correct = False
            if expected_value == .3:
                self.arg_passed = True

        #def _fire_upstream(self):
        #    self.fire_called = True
        #    if not self.delta_called:
        #        self.order_correct = False

    my_node = patched_BPNeurode()
    my_node.set_expected(.3)
    assert my_node.delta_called, \
        "Did set_expected call _calculate_delta?"
    #assert my_node.fire_called, \
    #    "Did set_expected call _fire_upstream?"
    assert my_node.arg_passed, \
        "Did set_expected correctly pass an argument to calculate_delta?"
    assert my_node.order_correct, \
        "Did set_expected handle function calls in the correct order?"


def test_data_ready_downstream():
    class patched_BPNeurode(BPNeurode.BPNeurode):
        def __init__(self, in_node):
            self.in_node = in_node
            self.delta_called = False
            self.fire_called = False
            self.update_called = False
            self.check_in_return = False
            self.check_in_arg_passed = False
            self.calc_delta_arg_passed = False
            self.order_correct = True
            super().__init__()

        def _calculate_delta(self, expected_value=0):
            self.delta_called = True
            if self.fire_called or self.update_called:
                self.order_correct = False
            if expected_value == .3:
                self.calc_delta_arg_passed = True

        def _fire_upstream(self):
            self.fire_called = True
            if not self.delta_called or self.update_called:
                self.order_correct = False

        def _update_weights(self):
            self.update_called = True
            if not self.delta_called or not self.fire_called:
                self.order_correct = False

        def _check_in(self, node: Neurode, side: MultiLinkNode.Side):
            if self.in_node is node and side is MultiLinkNode.Side.DOWNSTREAM:
                self.check_in_arg_passed = True
            return self.check_in_return

    my_passed_node = BPNeurode.BPNeurode()
    my_node = patched_BPNeurode(my_passed_node)
    my_node.data_ready_downstream(my_passed_node)
    assert not my_node.delta_called, \
        "data_ready_downstream called _calculate_delta prematurely"
    assert not my_node.fire_called, \
        "data_ready_downstream called _fire_upstream prematurely"
    assert not my_node.update_called, \
        "data_ready_downstream called _update_weights prematurely"
    my_node.check_in_return = True
    my_node.data_ready_downstream(my_passed_node)
    assert my_node.delta_called, \
        "Did data_ready_downstream call _calculate_delta?"
    assert my_node.fire_called, \
        "Did data_ready_downstream call _fire_upstream?"
    assert my_node.update_called, \
        "Did data_ready_downstream call _update_weights?"
    assert my_node.order_correct, \
        "Did data_ready_upstream handle function calls in the correct order?"


def test_adjust_weights():
    my_node = BPNeurode.BPNeurode()
    other_node = BPNeurode.BPNeurode()
    my_node._weights = {other_node: .5}
    my_node.adjust_weights(other_node, .1)
    assert my_node.get_weight(other_node) == .6, \
        "Adjust weights did not alter _weights correctly."


def test_update_weights():
    class patched_BPNeurode(BPNeurode.BPNeurode):
        def __init__(self, node):
            self.up_node = node
            self.expected_adjustment = .5
            self.adjust_weights_called = False
            self.correct_arg_passed = False
            super().__init__()

        def adjust_weights(self, node, adjustment):
            if (math.isclose(adjustment, self.expected_adjustment)
                    and node is self.up_node):
                self.correct_arg_passed = True
            self.adjust_weights_called = True

    up_node = BPNeurode.BPNeurode()
    up_node._value = .3
    down_node_1 = patched_BPNeurode(up_node)
    down_node_1._delta = .1
    down_node_1.expected_adjustment = .0015
    down_node_2 = patched_BPNeurode(up_node)
    down_node_2._delta = .2
    down_node_2.expected_adjustment = .003
    up_node.reset_neighbors([down_node_1, down_node_2],
                            BPNeurode.BPNeurode.Side.DOWNSTREAM)
    up_node._update_weights()
    assert down_node_1.adjust_weights_called and \
        down_node_2.adjust_weights_called, \
        "Did _update_weights call adjust_weights on each downstream node?"
    assert down_node_1.correct_arg_passed and \
        down_node_2.correct_arg_passed, \
        "Did _update_weights pass the correct arguments to downstream nodes?"

def test_fire_upstream():
    class patched_BPNeurode(BPNeurode.BPNeurode):
        def __init__(self, node):
            self.down_node = node
            self.expected_adjustment = .5
            self.data_ready_called = False
            self.correct_arg_passed = False
            super().__init__()

        def data_ready_downstream(self, node):
            if node is self.down_node:
                self.correct_arg_passed = True
            self.data_ready_called = True

    down_node = BPNeurode.BPNeurode()
    up_node_1 = patched_BPNeurode(down_node)

    up_node_2 = patched_BPNeurode(down_node)

    down_node.reset_neighbors([up_node_1, up_node_2],
                            BPNeurode.BPNeurode.Side.UPSTREAM)
    down_node._fire_upstream()
    assert up_node_1.data_ready_called and \
           up_node_2.data_ready_called, \
        "Did _fire_upstream call data_ready_downstream on each upstream node?"
    assert up_node_1.correct_arg_passed and \
           up_node_2.correct_arg_passed, \
        "Did _fire_upstream pass the correct arguments to upstream nodes?"


