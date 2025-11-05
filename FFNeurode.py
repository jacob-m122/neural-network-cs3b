"""Implement FFNeurode Class."""
from __future__ import annotations
from Neurode import Neurode
from Neurode import MultiLinkNode
from math import exp
import random


class FFNeurode(Neurode):
    """Inherit from parent, Neurode, implement feed-forward process.
    
    Minimal upgrades:
      - Per-node learnable bias (self._bias), included in forward sum.
      - Safe default learning rate on the node (does not override if already set).
    """

    def __init__(self, *args, **kwargs):
        # Preserve whatever Neurode expects
        super().__init__(*args, **kwargs)
        # Small random bias init (centered near 0 to avoid saturation)
        self._bias = (random.random() - 0.5) * 0.02
        # Only set a default LR if one isn't already present on the instance
        if not hasattr(self, "learning_rate"):
            self.learning_rate = 0.1

    @staticmethod
    def _sigmoid(value: float):
        """Return sigmoid function result."""
        return 1 / (1 + exp(-value))

    # --- Bias helpers (optional, handy for inspection/checkpointing) ---
    def get_bias(self) -> float:
        return self._bias

    def set_bias(self, value: float) -> None:
        self._bias = float(value)

    def _calculate_value(self):
        """Calculate sum of weighted upstream node values (+ learnable bias)."""
        weighted_sum = self._bias  # <-- include bias
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node_value = node.value
            weight = self.get_weight(node)
            weighted_sum += node_value * weight
        self._value = self._sigmoid(weighted_sum)

    def _fire_downstream(self):
        """Indicate to downstream neighbors that node has data ready."""
        for neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            neighbor.data_ready_upstream(self)

    def data_ready_upstream(self, node: Neurode):
        """Handle data from upstream node."""
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value: float):
        """Set input node value, indicate to downstream nodes."""
        self._value = input_value
        for neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            neighbor.data_ready_upstream(self)
