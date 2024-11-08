"""Implement Back Propagation Neurode Class."""
from Neurode import Neurode
from Neurode import MultiLinkNode


class BPNeurode(Neurode):
    """Implement backpropagation Neurode class."""

    def __init__(self):
        """Call superclass, initialize delta to zero."""
        super().__init__()
        self._delta = 0

    @staticmethod
    def _sigmoid_derivative(value: float):
        """Calculate the sigmoid function derivative."""
        return value * (1 - value)

    def _calculate_delta(self, expected_value: float = None):
        """
        Calculate based off whether an output or hidden/input layer.

        Calculate error margin, or weighted sum, update delta.
        """
        if expected_value is not None:
            delta_error = expected_value - self.value
            self._delta = delta_error * self._sigmoid_derivative(self.value)
        else:
            weighted_sum_downstream = sum(
                node.get_weight(self) * node._delta
                for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]
            )
            self._delta = (
                weighted_sum_downstream * self._sigmoid_derivative(self._value)
            )

    def data_ready_downstream(self, node: Neurode):
        """Indicate when data is ready to upstream neighbors."""
        if self._check_in(node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._update_weights()
            self._fire_upstream()

    def set_expected(self, expected_value: float):
        """Set expected value of output layer."""
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node: Neurode, adjustment: float):
        """Adjust weight for each respective neurode by given amount."""
        self._weights[node] += adjustment

    def _update_weights(self):
        """Update downstream node weights."""
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = self.learning_rate * node.delta * self._value
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        """Notify upstream neighbors."""
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    @property
    def delta(self):
        """Return current delta value."""
        return self._delta


"""
Output value: 0.6
Expected Output: 1.0
Delta of Output Neurode: 0.096
"""
