"""Implementing the NNData Class."""
from enum import Enum
from collections import deque
import numpy as np
import random


class Order(Enum):
    """Indicate if training data shown in different or same order."""
    SHUFFLE = 1
    STATIC = 2


class Set(Enum):
    """Indicate whether requesting training or testing data."""
    TRAIN = 1
    TEST = 2


class NNData:
    """NNData manages training and testing data."""

    @staticmethod
    def percentage_limiter(percentage: float):
        """Return percentage clamped between 0 and 1."""
        if percentage < 0:
            return 0
        if percentage > 1:
            return 1
        return percentage

    def load_data(self, features=None, labels=None):
        """Add features and labels to the class."""
        match (features, labels):
            case (None, _) | (_, None):
                self._features = None
                self._labels = None
                self.split_set()
                return
            case (f, l) if len(f) is not len(l):
                self._features = None
                self._labels = None
                self.split_set()
                raise ValueError('Mismatched features and labels length')
            case _:
                try:
                    self._features = np.array(features, dtype=float)
                    self._labels = np.array(labels, dtype=float)
                except (ValueError, TypeError):
                    # ValueError eg, ['a', 'b'] - Not Float
                    # TypeError eg, 6 - Not Iterable
                    self._features = None
                    self._labels = None
                    self.split_set()
                    raise ValueError('Numpy array construction failed.')
        self.split_set()

    def __init__(self, features=None, labels=None, train_factor=0.9,):
        """Initialize NNData with data."""
        self._features = None
        self._labels = None
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        self._train_factor = NNData.percentage_limiter(train_factor)
        self.load_data(features, labels)

    def split_set(self, new_train_factor=None):
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        if self._features is None:
            self._train_indices = []
            self._test_indices = []
            return
        num_examples = self._features.size
        num_examples_for_testing = (
                num_examples - int(num_examples * self._train_factor))
        indices = list(range(num_examples))
        test_indices = random.sample(indices, num_examples_for_testing)
        train_indices = [i for i in indices if i not in test_indices]
        random.shuffle(train_indices)
        self._train_indices = train_indices
        self._test_indices = test_indices

    def prime_data(self, target_set=None, order=None):
        """Load one or both deques as indirect indices."""
        match (target_set, order):
            case (Set.TRAIN, None | Order.STATIC):
                self._train_pool = deque(self._train_indices)
            case (Set.TRAIN, Order.SHUFFLE):
                random.shuffle(self._train_indices)
            case (Set.TEST, None | Order.STATIC):
                self._test_pool = deque(self._test_indices)
            case (Set.TEST, Order.SHUFFLE):
                random.shuffle(self._test_indices)
            case (None, Order.SHUFFLE):
                random.shuffle(self._test_pool)
                random.shuffle(self._test_pool)
            case (None, None | Order.STATIC):
                self._train_pool = deque(self._train_indices)
                self._test_pool = deque(self._test_indices)
            case (_, _):
                raise ValueError('Unhandled arguments for prime_data')
