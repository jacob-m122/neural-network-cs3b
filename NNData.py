"""Implement the NNData class for management of training and testing data."""

import numpy as np
import random
from collections import deque
from enum import Enum


class Order(Enum):
    """Define order of presentation to neural network."""

    SHUFFLE = 1
    STATIC = 2


class Set(Enum):
    """Identify whether training or testing set data is requested."""

    TRAIN = 1
    TEST = 2


class NNData:
    """Manage training and testing data."""

    @staticmethod
    def percentage_limiter(percentage):
        """Ensure percentage value is in valid range."""
        if percentage < 0:
            return 0
        elif percentage > 1:
            return 1
        elif 0 <= percentage <= 1:
            return percentage

    def __init__(self, features=None, labels=None, train_factor=.9):
        """Ensure features and labels are lists of lists.

        Initialize various instance attributes.
        """
        if features is None:
            features = []
        if labels is None:
            labels = []

        self._features = None
        self._labels = None
        self._train_factor = self.percentage_limiter(train_factor)

        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()

        self.load_data(features, labels)

    def load_data(self, features=None, labels=None):
        """Load data and compare lengths of features and labels.

        Create arrays of features and labels.
        """
        if features is None or labels is None:
            self._features = None
            self._labels = None
            self.split_set()
            return

        if len(features) != len(labels):
            self._features = None
            self._labels = None
            self.split_set()
            raise ValueError

        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            self.split_set()
            raise ValueError

        self.split_set()

    def split_set(self, new_train_factor=None):
        """Divide data into training and testing sets.

        Find total number of examples and compute training examples,
        create indices and randomize indices for the training set,
        place the remaining in the testing set.
        """
        if new_train_factor is not None:
            self._train_factor = self.percentage_limiter(new_train_factor)

        if self._features is None:
            self._train_indices = []
            self._test_indices = []
            return

        examples_amount = len(self._features)
        training_examples = int(examples_amount * self._train_factor)
        indices_set = list(range(examples_amount))

        self._train_indices = random.sample(indices_set, training_examples)
        self._test_indices = []
        for i in indices_set:
            if i not in self._train_indices:
                self._test_indices.append(i)

    def prime_data(self, target_set=None, order=None):
        """Load deques for utilization as indirect indices."""
        if target_set == Set.TRAIN or target_set is None:
            self._train_pool.clear()
            if self._train_indices:
                training_indices = self._train_indices.copy()
                if order == Order.SHUFFLE:
                    random.shuffle(training_indices)
                self._train_pool.extend(training_indices)

        if target_set == Set.TEST or target_set is None:
            self._test_pool.clear()
            if self._test_indices:
                testing_indices = self._test_indices.copy()
                if order == Order.SHUFFLE:
                    random.shuffle(testing_indices)
                self._test_pool.extend(testing_indices)

    def get_one_item(self, target_set=None):
        """Return feature and label pair as a tuple."""
        if self._features is None:
            raise ValueError

        if target_set == Set.TRAIN or target_set is None:
            index_pool = self._train_pool
        elif target_set == Set.TEST:
            index_pool = self._test_pool
        else:
            return None

        if not index_pool:
            return None

        index_reference = index_pool.popleft()
        feature = self._features[index_reference]
        label = self._labels[index_reference]
        return (feature, label)

    def number_of_samples(self, target_set=None):
        """Return total number of testing or training examples.

        If target_set is None, return both as a tuple.
        """
        if target_set == Set.TEST:
            return len(self._test_indices)
        elif target_set == Set.TRAIN:
            return len(self._train_indices)
        else:
            if target_set is None:
                return len(self._test_indices), len(self._train_indices)

    def pool_is_empty(self, target_set=None):
        """Return True if the target set deque is empty.

        If target_set is None, utilize the training pool.
        """
        if target_set == Set.TRAIN or target_set is None:
            return not self._train_pool
        elif target_set == Set.TEST:
            return not self._test_pool
        else:
            return None


"""
In [1]: import NNData

In [2]: XOR_features = [[0, 0], [0, 1], [1, 0], [1, 1]]

In [3]: XOR_labels = [[0], [1], [1], [0]]

In [4]: my_data = NNData.NNData(features=XOR_features, labels=XOR_labels)

In [5]: my_data.get_one_item(NNData.Set.TRAIN)

In [6]: my_data.prime_data()

In [7]: my_data.get_one_item(NNData.Set.TRAIN)
Out[7]: (array([0., 1.]), array([1.]))

In [8]: my_data.get_one_item(NNData.Set.TRAIN)
Out[8]: (array([0., 0.]), array([0.]))

In [9]: my_data.get_one_item(NNData.Set.TRAIN)
Out[9]: (array([1., 0.]), array([1.]))

In [10]: my_data.get_one_item(NNData.Set.TRAIN)

In [11]: my_data.split_set(.5)

In [12]: my_data.prime_data()

In [13]: my_data.get_one_item(NNData.Set.TRAIN)
Out[13]: (array([1., 0.]), array([1.]))

In [14]: my_data.pool_is_empty()
Out[14]: False

In [15]: my_data.get_one_item(NNData.Set.TRAIN)
Out[15]: (array([1., 1.]), array([0.]))

In [16]: my_data.pool_is_empty()
Out[16]: True

In [17]: my_data.prime_data(order=NNData.Order.STATIC)

In [18]: my_data.get_one_item(NNData.Set.TRAIN)
Out[18]: (array([1., 0.]), array([1.]))

In [19]: my_data.get_one_item(NNData.Set.TRAIN)
Out[19]: (array([1., 1.]), array([0.]))

In [20]: my_data.prime_data(order=NNData.Order.STATIC)

In [21]: my_data.get_one_item(NNData.Set.TRAIN)
Out[21]: (array([1., 0.]), array([1.]))

In [22]: my_data.get_one_item(NNData.Set.TRAIN)
Out[22]: (array([1., 1.]), array([0.]))

In [23]: my_data.prime_data(order=NNData.Order.SHUFFLE)

In [24]: my_data.get_one_item(NNData.Set.TRAIN)
Out[24]: (array([1., 0.]), array([1.]))

In [25]: my_data.get_one_item(NNData.Set.TRAIN)
Out[25]: (array([1., 1.]), array([0.]))

In [26]:
"""
