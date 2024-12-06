import pytest

#
# DO NOT MODIFY THIS CODE
#

try:
    import LayerList
except ImportError:
    pytest.fail("Cannot import LayerList. Is LayerList.py present?")

try:
    from FFBPNeurode import FFBPNeurode
except ImportError:
    pytest.fail("Cannot import FFBPNeurode. Is FFBPNeurode.py present?")


def test_LayerList_constructor():
    try:
        my_list = LayerList.LayerList(3, 4, FFBPNeurode)
    except:
        pytest.fail("LayerList construction was not successful")
    try:
        inputs = my_list.input_nodes
        print(inputs)
    except:
        pytest.fail("Could not retrieve input nodes, is there a "
                    "@property for inputs?")
    try:
        outputs = my_list.output_nodes
        print(outputs)
    except:
        pytest.fail("Could not retrieve output nodes, is there a "
                    "@property for outputs?")
    assert len(inputs) == 3, \
        "Wrong number of input nodes returned."
    assert len(outputs) == 4, \
        "Wrong number of output nodes returned."
    assert type(inputs[0]) == FFBPNeurode, \
        "Input layer nodes should be of type FFBPNeurode"
    assert type(outputs[0]) == FFBPNeurode, \
        "Output layer nodes should be of type FFBPNeurode"
    assert inputs[0] is not inputs[1], \
        "Each input layer node should be a distinct object"
    assert outputs[0] is not outputs[1], \
        "Each output layer node should be a distinct object"
    set_of_inputs = set(inputs)
    set_of_outputs = set(outputs)
    i_neighbors = \
        set(inputs[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM])
    o_neighbors = \
        set(outputs[1]._neighbors[FFBPNeurode.Side.UPSTREAM])
    assert set_of_outputs == i_neighbors, \
        "Is each input layer node connected to all output layer nodes?"
    assert set_of_inputs == o_neighbors, \
        "Is each input layer node connected to all output layer nodes?"


def test_LayerList_constructor_alternate_neurode_type():
    class AltNeurode(FFBPNeurode):
        pass

    try:
        my_list = LayerList.LayerList(3, 4, AltNeurode)
    except:
        pytest.fail("LayerList construction was not successful with "
                    "alternative Neurode type")
    try:
        inputs = my_list.input_nodes
    except:
        pytest.fail("Could not retrieve input nodes, is there a "
                    "@property for inputs?")
    assert type(inputs[0]) == AltNeurode, \
        "LayerList with alternative Neurode type not set up correctly."


def test_LayerList_add_and_remove_layer():
    try:
        my_list = LayerList.LayerList(3, 4, FFBPNeurode)
    except:
        pytest.fail("LayerList construction was not successful")
    try:
        inputs = my_list.input_nodes
        print(inputs)
    except:
        pytest.fail("Could not retrieve input nodes, is there a "
                    "@property for inputs?")
    try:
        outputs = my_list.output_nodes
        print(outputs)
    except:
        pytest.fail("Could not retrieve output nodes, is there a "
                    "@property for outputs?")
    my_list.add_layer(2)
    my_list.add_layer(5)
    my_list.move_forward()
    hidden_one = my_list.curr_data
    my_list.move_forward()
    hidden_two = my_list.curr_data
    assert len(hidden_one) == 5, \
        "First hidden layer has wrong number of neurodes"
    assert len(hidden_two) == 2, \
        "Second hidden layer has wrong number of neurodes"
    assert outputs[1] not in inputs[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM], \
        ("Input and output layer neurodes should not be connected after hidden "
         "layer is added")
    assert inputs[1] not in outputs[1]._neighbors[FFBPNeurode.Side.UPSTREAM], \
        ("Input and output layer neurodes should not be connected after hidden "
         "layer is added")
    set_of_inputs = set(inputs)
    set_of_hidden_one = set(hidden_one)
    set_of_hidden_two = set(hidden_two)
    set_of_outputs = set(outputs)
    i_neighbors = \
        set(inputs[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM])
    h1_up_neighbors = \
        set(hidden_one[1]._neighbors[FFBPNeurode.Side.UPSTREAM])
    h1_down_neighbors = \
        set(hidden_one[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM])
    h2_up_neighbors = \
        set(hidden_two[1]._neighbors[FFBPNeurode.Side.UPSTREAM])
    h2_down_neighbors = \
        set(hidden_two[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM])
    o_neighbors = \
        set(outputs[1]._neighbors[FFBPNeurode.Side.UPSTREAM])
    assert set_of_inputs == h1_up_neighbors, \
        ("After adding two layers, is each input layer node connected to all "
         "first hidden layer nodes?")
    assert i_neighbors == set_of_hidden_one, \
        ("After adding two layers, is each input layer node connected to all "
         "first hidden layer nodes?")
    assert set_of_hidden_one == h2_up_neighbors, \
        ("After adding two layers, are first hidden layer nodes fully "
         "connected to second hidden layer nodes?")
    assert set_of_hidden_two == h1_down_neighbors, \
        ("After adding two layers, are first hidden layer nodes fully "
         "connected to second hidden layer nodes?")
    assert set_of_hidden_two == o_neighbors, \
        ("After adding two layers, is each node in the second hidden layer "
         "connected to all output layer nodes?")
    assert set_of_outputs == h2_down_neighbors, \
        ("After adding two layers, is each node in the second hidden layer "
         "connected to all output layer nodes?")
    with pytest.raises(IndexError):
        my_list.remove_layer()
    my_list.move_backward()
    try:
        my_list.remove_layer()
    except:
        pytest.fail("Error when trying to remove second hidden layer.")
    my_list.reset_to_head()
    my_list.move_forward()
    assert my_list.curr_data == hidden_one, \
        ("After removing hidden layer, neurodes in hidden layer one are not "
         "correct")
    assert (hidden_two[1] not in
            hidden_one[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM]), \
        ("After removing second hidden layer, neurodes in hidden layer one "
         "still list hidden layer two neurodes as neighbors.")
    assert (hidden_two[1] not in
            outputs[1]._neighbors[FFBPNeurode.Side.UPSTREAM]), \
        ("After removing second hidden layer, neurodes in output layer "
         "still list hidden layer two neurodes as neighbors.")
    h1_down_neighbors = \
        set(hidden_one[1]._neighbors[FFBPNeurode.Side.DOWNSTREAM])
    o_neighbors = \
        set(outputs[1]._neighbors[FFBPNeurode.Side.UPSTREAM])
    assert set_of_hidden_one == o_neighbors, \
        ("After adding two layers, is each node in the second hidden layer "
         "connected to all output layer nodes?")
    assert set_of_outputs == h1_down_neighbors, \
        ("After adding two layers, is each node in the second hidden layer "
         "connected to all output layer nodes?")
