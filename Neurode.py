"""Implement the MultiLinkNode and Neurode classes."""
from __future__ import annotations
from enum import Enum
import copy
import random
from abc import ABC, abstractmethod


class MultiLinkNode:
    """
    Base class which implements enum for upstream and downstream nodes.

    Initializes dictionaries for reporting, reference, and neighboring nodes.
    """

    class Side(Enum):
        """Enumerate UPSTREAM and DOWNSTREAM sides."""

        UPSTREAM = 1
        DOWNSTREAM = 2

    def __init__(self):
        """Initialize reference value, reporting, and neighboring nodes."""
        self._reporting_nodes = {
            MultiLinkNode.Side.UPSTREAM: 0,
            MultiLinkNode.Side.DOWNSTREAM: 0
        }
        self._reference_value = {
            MultiLinkNode.Side.UPSTREAM: 0,
            MultiLinkNode.Side.DOWNSTREAM: 0
            }
        self._neighbors = {
            MultiLinkNode.Side.UPSTREAM: [],
            MultiLinkNode.Side.DOWNSTREAM: []
        }

    def __str__(self):
        """Gather Node ID's and create string representations.

        of each node's ID.
        """
        upstream_id = [
            str(id(node))
            for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]
            ]
        downstream_id = [
            str(id(node))
            for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]
            ]

        return (
            f"Node ID: {id(self)}\n"
            f"Upstream Neighbors: {', '.join(upstream_id)}\n"
            f"Downstream Neighbors: {', '.join(downstream_id)}\n"
        )

    @abstractmethod
    def _process_new_neighbor(self, node: MultiLinkNode, side: Side):
        """
        Handle new neighboring node when added.

        Require implementation for Abstract methods.
        """
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        """Reset list of neighbors on either side and update."""
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)

        self._reference_value[side] = (1 << len(nodes)) - 1


class Neurode(MultiLinkNode):
    """Inherit from MultiLinkNode implement abstract methods."""

    _learning_rate = 0.05

    @property
    def learning_rate(self):
        """Return current learning rate."""
        return Neurode._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        """Update learning rate."""
        Neurode._learning_rate = value

    def __init__(self):
        """Initialize MultiLinkNode att's, current node value and weight."""
        super().__init__()
        self._value = 0
        self._weights = {}

    def _process_new_neighbor(self, node: Neurode, side: Side):
        """Process initialization of UPSTREAM weight."""
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node: Neurode, side: Side):
        """Track and report neighboring nodes have reported."""
        node_index = self._neighbors[side].index(node)
        self._reporting_nodes[side] |= 1 << node_index
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    def get_weight(self, node: Neurode):
        """Return weight of UPSTREAM node."""
        return self._weights[node]

    @property
    def value(self):
        """Provide current node value."""
        return self._value


"""
Test __str__:
Node ID: 4312508576
Upstream Neighbors: 4312101904
Downstream Neighbors:

Following resetting neighbors to empty list:
Node ID: 4312508576
Upstream Neighbors:
Downstream Neighbors:

Following resetting neighbors to node two:
Node ID: 4312508576
Upstream Neighbors: 4312101904
Downstream Neighbors:

Following node two reporting in, all_reported = True
Reporting nodes (UPSTREAM): 0
"""
