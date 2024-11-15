"""Implement the LayerList class for input, hidden, and output layer management."""
from DoublyLinkedList import DoublyLinkedList
from Neurode import Neurode

class LayerList(DoublyLinkedList):
    """implement LayerList class"""
    def __init__(self, inputs: int, outputs: int, neurode_type: type(Neurode)):
        super().__init__()
        self._neurode_type = neurode_type
        self.input_layer = [
            neurode_type() for _ in range(inputs)
            ]
        self.output_layer = [
            neurode_type() for _ in range(outputs)
            ]
        self.add_to_head(self.input_layer)
        self.add_after_current(self.output_layer)
        self.linking_helper(self.input_layer, self.output_layer)

    def linking_helper(self, layer_upstream, layer_downstream):
        """Link neighboring layers."""
        for neurode in layer_upstream:
            neurode.reset_neighbors(layer_downstream, Neurode.Side.DOWNSTREAM)
        for neurode in layer_downstream:
            neurode.reset_neighbors(layer_upstream, Neurode.Side.UPSTREAM)

    def add_layer(self, num_nodes: int):
        """
        Ensure current layer is not output layer (tail).
        Insert hidden layer between input and output layers, link layers.
        """
        if self._curr == self._tail:
            raise IndexError
        new_hidden_layer = [
            self._neurode_type() for _ in range(num_nodes)
        ]
        self.add_after_current(new_hidden_layer)
        layer_upstream = self._curr.data
        self.move_forward()
        self.linking_helper(layer_upstream, new_hidden_layer)
        layer_downstream = self._curr.next.data
        self.linking_helper(new_hidden_layer, layer_downstream)

    def remove_layer(self):
        """Remove specific layer, ensure it is not removing output layer (tail)"""
        if self._curr.next == self._tail:
            raise IndexError
        layer_upstream = self._curr.data
        self.remove_after_current()
        if self._curr.next:
            layer_downstream = self._curr.next.data
            self.linking_helper(layer_upstream, layer_downstream)

    @property
    def input_nodes(self):
        """Access Input layer neurodes."""
        return self._head.data

    @property
    def output_nodes(self):
        """Access output layer neurodes."""
        return self._tail.data

"""
Layerlist created.
Input neurodes: 2
Output neurodes: 1
"""
