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

    def __init__(self, features=None, labels=None, train_factor=0.9):
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
        self._val_indices = []           # <-- optional validation indices
        self._train_pool = deque()
        self._test_pool = deque()

        # Standardization state (created by fit_standardizer)
        self._mu = None
        self._sigma = None

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
            self._val_indices = []
            return

        examples_amount = len(self._features)
        training_examples = int(examples_amount * self._train_factor)
        indices_set = list(range(examples_amount))

        self._train_indices = random.sample(indices_set, training_examples)
        self._test_indices = []
        for i in indices_set:
            if i not in self._train_indices:
                self._test_indices.append(i)
        # leave _val_indices unused here; use stratified_split() if you want train/val/test

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

    # ---------- NEW HELPERS BELOW ----------

    def _class_index(self, i: int) -> int:
        """Resolve class index from one-hot label at sample i."""
        lbl = self._labels[i]
        # Handle numpy scalars, 1-D arrays, lists
        if np.isscalar(lbl):
            return int(lbl)
        arr = np.asarray(lbl).ravel()
        if arr.size == 1:
            return int(arr.item())
        return int(np.argmax(arr))

    def stratified_split(self, train=0.7, val=0.15, test=0.15, random_state: int = 42):
        """
        Create stratified TRAIN/VAL/TEST index splits based on one-hot (or class index) labels.
        Does not alter pools; call prime_data() as usual for TRAIN/TEST.
        VAL indices are stored in self._val_indices for your own use.
        """
        if self._labels is None or self._features is None:
            self._train_indices, self._val_indices, self._test_indices = [], [], []
            return

        if not np.isclose(train + val + test, 1.0):
            # simple normalization to sum to 1.0
            s = train + val + test
            train, val, test = train / s, val / s, test / s

        rng = random.Random(random_state)
        by_class = {}
        n_samples = len(self._labels)
        for i in range(n_samples):
            c = self._class_index(i)
            by_class.setdefault(c, []).append(i)

        train_idx, val_idx, test_idx = [], [], []
        for _, idxs in by_class.items():
            rng.shuffle(idxs)
            n = len(idxs)
            n_tr = int(round(n * train))
            n_val = int(round(n * val))
            tr = idxs[:n_tr]
            vl = idxs[n_tr:n_tr + n_val]
            te = idxs[n_tr + n_val:]
            train_idx.extend(tr)
            val_idx.extend(vl)
            test_idx.extend(te)

        self._train_indices = train_idx
        self._val_indices = val_idx
        self._test_indices = test_idx

    def fit_standardizer(self, indices):
        """
        Compute per-feature mean/std from the provided indices (e.g., train indices)
        and store them on the instance for later transform.
        """
        if self._features is None or not indices:
            self._mu, self._sigma = None, None
            return
        X = np.array([self._features[i] for i in indices], dtype=float)
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        # Avoid divide-by-zero
        self._sigma[self._sigma == 0.0] = 1.0

    def transform_features(self, indices):
        """
        Apply z-score normalization in-place to the given indices using
        statistics computed by fit_standardizer.
        """
        if self._features is None or self._mu is None or self._sigma is None:
            return
        for i in indices:
            self._features[i] = ((np.asarray(self._features[i], dtype=float) - self._mu) / self._sigma)
