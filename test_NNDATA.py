"""Testing for Asst Five"""
import pytest
import copy
import collections
import numpy

#
# DO NOT MODIFY THIS CODE
#
try:
    import NNData
except:
    pytest.fail("NNData.py file is not present")


@staticmethod
def bare_percentage_limiter(data_in):
    return data_in


def bare_split_set():
    pass


def bare_load_data(self, features=None, labels=None):
    pass


@pytest.fixture
def bare_constructor():
    class StrippedClass(NNData.NNData):
        pass
    StrippedClass.split_set = bare_split_set
    StrippedClass.load_data = bare_load_data
    StrippedClass.percentage_limiter = bare_percentage_limiter
    StrippedClass._train_factor = .5
    return StrippedClass


@pytest.fixture
def my_NNData_full():
    features = [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8]]
    labels = [[1], [2], [3], [4], [5], [6], [7], [8]]
    test_object = NNData.NNData(features, labels)
    return test_object


@pytest.fixture
def fake_constructor():
    class FakeConstructor:
        def __init__(self):
            self._split_set_called = False
            self._train_factor = .5

        def split_set(self, new_train_factor=None):
            self._split_set_called = True

    FakeConstructor.percentage_limiter = bare_percentage_limiter
    return FakeConstructor

def test_random_assignment():
    features = [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8]]
    labels = [[1], [2], [3], [4], [5], [6], [7], [8]]
    test_object_one = NNData.NNData(features, labels,.75)
    test_object_two = NNData.NNData(features, labels,.75)
    test_object_three = NNData.NNData(features, labels, .75)
    comp_one = set(test_object_one._train_indices) == set(test_object_two._train_indices)
    comp_two = set(test_object_one._train_indices) == set(test_object_three._train_indices)
    assert not (comp_one and comp_two), \
        ("Examples must be randomly assigned to TEST or TRAIN")



def test_enums():
    """Test all functionality of DLLNode."""
    try:
        NNData.Order.SHUFFLE
        NNData.Order.STATIC
    except:
        pytest.fail("Is there an Enum class called Order with "
                    "elements SHUFFLE and STATIC?")
    try:
        NNData.Set.TEST
        NNData.Set.TRAIN
    except:
        pytest.fail("Is there an Enum class called Set with "
                    "elements TEST and STATIC?")


def test_empty_constructor(bare_constructor: type(NNData.NNData)):
    my_data = bare_constructor()
    assert my_data._features is None, \
        ("self._features and self._labels should be set to None in "
         "the constructor to avoid warnings")
    assert my_data._labels is None, \
        ("self._features and self._labels should be set to None in "
         "the constructor to avoid warnings")
    assert my_data._train_indices == [], \
        ("self._train_indices and self._test_indices should be set "
         "to empty lists in the constructor.")
    assert my_data._test_indices == [], \
        ("self._train_indices and self._test_indices should be set "
         "to empty lists in the constructor.")
    assert isinstance(my_data._train_pool, collections.deque), \
        ("self._train_pool and self._test_pool should be "
         "initialized as empty deques in the constructor.")
    assert isinstance(my_data._test_pool, collections.deque), \
        ("self._train_pool and self._test_pool should be "
         "initialized as empty deques in the constructor.")


def test_load_data_normal_operation(fake_constructor: type(NNData.NNData)):
    try:
        fake_constructor.load_data = NNData.NNData.load_data
    except:
        pytest.fail("Is load_data() implemented?")
    features = [[.1], [.2], [.3], [.4], [.5], [.6],
                [.7], [.8]]
    labels = [[1], [2], [3], [4], [5], [6], [7], [8]]
    my_data = fake_constructor()
    my_data.load_data(features=features, labels=labels)
    assert my_data._split_set_called, \
        "Was split_set() called in load_data()?"
    assert isinstance(my_data._features, numpy.ndarray), \
        "Is self._features set to a numpy array in load_data?"
    assert isinstance(my_data._labels, numpy.ndarray), \
        "Is self._labels set to a numpy array in load_data?"
    test_features = numpy.array(features)
    test_labels = numpy.array(labels)
    assert numpy.array_equal(my_data._features, test_features), \
        "load_data is not correctly loading self._features"
    assert numpy.array_equal(my_data._labels, test_labels), \
        "load_data is not correctly loading self._labels"
    fake_constructor._split_set_called = False
    try:
        my_data.load_data()
    except:
        pytest.fail("Do features and labels have default values "
                    "in load_data?")
    assert my_data._split_set_called, \
        "Was split_set() called in load_data()?"
    assert my_data._features is None, \
        "Is self._features set to None if no features are passed?"
    assert my_data._labels is None, \
        "Is self._labels set to None if no features are passed?"
    test_features = numpy.array(features)
    test_labels = numpy.array(labels)


def test_load_data_error_condition(fake_constructor: type(NNData.NNData)):
    try:
        fake_constructor.load_data = NNData.NNData.load_data
    except:
        pytest.fail("Is NNData.load_data() implemented?")
    my_data = fake_constructor()
    features = [[.1], [.2], [.3], [.4], [.5], [.6],
                [.7]]
    labels = [[1], [2], [3], [4], [5], [6], [7], [8]]
    try:
        my_data.load_data(features=features, labels=labels)
        pytest.fail("load_data() should raise a ValueError if the"
                    "length of features does not match the length"
                    "of labels")
    except ValueError:
        if my_data._features is not None or my_data._labels is not None:
            pytest.fail("load_data() must set self._features and "
                        "self._labels to None in error situations.")
        assert my_data._split_set_called, \
            "Is split_set() called in load_data() in error conditions?"
    my_data = fake_constructor()
    features = [["a"], ["b"], ["c"], ["d"], ["e"], ["f"],
                ["g"], ["h"]]
    labels = [[1], [2], [3], [4], [5], [6], [7], [8]]
    try:
        my_data.load_data(features=features, labels=labels)
        pytest.fail("load_data() should raise a ValueError if features "
                    "or labels cannot be made into arrays of floats.")
    except ValueError:
        if my_data._features is not None or my_data._labels is not None:
            pytest.fail("load_data() must set self._features and "
                        "self._labels to None in error situations.")
        assert my_data._split_set_called, \
            "Is split_set() called in load_data() in error conditions?"


def test_percentage_limiter():
    try:
        NNData.NNData.percentage_limiter(.5)
    except:
        pytest.fail("Is precentage_limiter defined as a @staticmethod?")
    assert NNData.NNData.percentage_limiter(.5) == .5, \
        "Percentage limiter is returning incorrect value"
    assert NNData.NNData.percentage_limiter(1.5) == 1, \
        "Percentage limiter is returning incorrect value"
    assert NNData.NNData.percentage_limiter(-.5) == 0, \
        "Percentage limiter is returning incorrect value"


def test_split_set(bare_constructor: type(NNData.NNData)):
    try:
        bare_constructor.split_set = NNData.NNData.split_set
    except:
        pytest.fail("Is split_set() implemented?")
    bare_constructor.percentage_limiter = bare_percentage_limiter
    my_data = bare_constructor()
    my_data._features = numpy.array([[.1], [.2], [.3], [.4], [.5], [.6],
                [.7], [.8]], dtype=float)
    my_data._labels = numpy.array([[1], [2], [3], [4], [5], [6], [7],
                                   [8]])
    my_data.split_set(.5)
    assert len(my_data._train_indices) == 4, \
        "split_set creates wrong number of train indices"
    assert len(my_data._test_indices) == 4, \
        "split_set creates wrong number of test indices"
    assert set(my_data._train_indices) | set(my_data._test_indices) == \
           set([0, 1, 2, 3, 4, 5, 6, 7]), \
        ("train_indices and test_indices should together contain a "
         "number of integers equal to the number of total samples")
    assert set(my_data._train_indices) & set(my_data._test_indices) == \
           set(), \
        ("train_indices and test_indices should not share any "
         "integers")
    my_data.split_set(0)
    assert len(my_data._train_indices) == 0, \
        "split_set creates wrong number of train indices"
    assert len(my_data._test_indices) == 8, \
        "split_set creates wrong number of test indices"
    assert set(my_data._test_indices)== {0, 1, 2, 3, 4, 5, 6, 7}, \
        ("train_indices and test_indices should together contain a "
         "number of integers equal to the number of total samples")
    assert set(my_data._train_indices) == set(), \
        "train_indices should be an empty list if train_factor == 0"
    my_data.split_set(1)
    assert len(my_data._train_indices) == 8, \
        "split_set creates wrong number of train indices"
    assert len(my_data._test_indices) == 0, \
        "split_set creates wrong number of test indices"
    assert set(my_data._train_indices) == {0, 1, 2, 3, 4, 5, 6, 7}, \
        ("train_indices and test_indices should together contain a "
         "number of integers equal to the number of total samples")
    assert set(my_data._test_indices) == set(), \
        "test_indices should be an empty list if train_factor == 0"
    my_data._features = None
    my_data._labels = None
    my_data.split_set(.5)
    assert my_data._train_indices == [], \
        ("train_indices should be an empty list if no examples are "
         "loaded.")
    assert my_data._test_indices == [], \
        ("test_indices should be an empty list if no examples are "
         "loaded.")
    try:
        my_data.split_set()
    except:
        pytest.fail("split_set must have a default value for train_factor")


def test_prime_data(bare_constructor: type(NNData.NNData)):
    try:
        bare_constructor.prime_data = NNData.NNData.prime_data
    except:
        pytest.fail("Is prime_data() implemented?")
    my_data = bare_constructor()
    my_data._train_indices = [3, 0, 5]
    my_data._test_indices = [4, 2, 1, 6, 7]
    my_data.prime_data()
    assert isinstance(my_data._train_pool, collections.deque), \
        "Is self._train_pool a deque?"
    assert isinstance(my_data._test_pool, collections.deque), \
        "Is self._test_pool a deque?"
    train_list = list(my_data._train_pool)
    assert set(train_list) == {3, 0, 5}, \
        ("self._train_pool does not contain the correct items when "
         "prime_data is called with no arguments.")
    test_list = list(my_data._test_pool)
    assert set(test_list) == {4, 2, 1, 6, 7}, \
        ("self._test_pool does not contain the correct items when "
         "prime_data is called with no arguments.")
    assert train_list == [3, 0, 5], \
        ("self._train_pool should not be shuffled when "
         "prime_data is called with no arguments.")
    test_list = list(my_data._test_pool)
    assert test_list == [4, 2, 1, 6, 7], \
        ("self._test_pool should not be shuffled when "
         "prime_data is called with no arguments.")
    try:
        my_data.prime_data(target_set=NNData.Set.TRAIN)
    except TypeError:
        my_data.prime_data(my_set=NNData.Set.TRAIN)
    while my_data._test_pool:
        my_data._test_pool.popleft()
    train_list = list(my_data._train_pool)
    assert set(train_list) == {3, 0, 5}, \
        ("self._train_pool does not contain the correct items when "
         "prime_data is called with target_set=NNData.Set.TRAIN.")
    assert len(my_data._test_pool) == 0, \
        ("self._test_pool should not be reset when "
         "prime_data is called with target_set=NNData.Set.TRAIN.")
    while my_data._train_pool:
        my_data._train_pool.popleft()
    try:
        my_data.prime_data(target_set=NNData.Set.TEST)
    except TypeError:
        my_data.prime_data(my_set=NNData.Set.TEST)
    test_list = list(my_data._test_pool)
    assert set(test_list) == {4, 2, 1, 6, 7}, \
        ("self._test_pool does not contain the correct items when "
         "prime_data is called with target_set=NNData.Set.TEST.")
    assert len(my_data._train_pool) == 0, \
        ("self._train_pool should not be reset when "
         "prime_data is called with target_set=NNData.Set.TEST.")
    static_list = [i for i in range(100)]
    my_data._train_indices = copy.copy(static_list)
    my_data._test_indices = copy.copy(static_list)
    my_data.prime_data(order=NNData.Order.SHUFFLE)
    assert list(my_data._train_pool) != static_list, \
        ("self._train_pool is not getting shuffled when "
         "prime_data is called with order=NNData.Order.SHUFFLE.")
    assert list(my_data._test_pool) != static_list, \
        ("self._test_pool is not getting shuffled when "
         "prime_data is called with order=NNData.Order.SHUFFLE.")


def test_pool_is_empty(bare_constructor: type(NNData.NNData)):
    try:
        bare_constructor.pool_is_empty = NNData.NNData.pool_is_empty
    except:
        pytest.fail("Is prime_data() implemented?")
    my_data = bare_constructor()
    my_data._test_pool = collections.deque([0, 1])
    my_data._train_pool = collections.deque([0, 1])
    assert not my_data.pool_is_empty(), \
        ("pool_is_empty should return False when no argument is "
         "provided and self._train_pool is not empty")
    assert not my_data.pool_is_empty(NNData.Set.TRAIN), \
        ("pool_is_empty should return False when NNData.Set.TRAIN is passed "
         "as an argument and self._train_pool is not empty")
    assert not my_data.pool_is_empty(NNData.Set.TEST), \
        ("pool_is_empty should return False when NNData.Set.TEST is passed "
         "as an argument and self._test_pool is not empty")
    my_data._train_pool = collections.deque()
    assert my_data.pool_is_empty(), \
        ("pool_is_empty should return True when no argument is "
         "provided and self._train_pool is empty")
    assert my_data.pool_is_empty(NNData.Set.TRAIN), \
        ("pool_is_empty should return True when NNData.Set.TRAIN is passed "
         "as an argument and self._train_pool is empty")
    assert not my_data.pool_is_empty(NNData.Set.TEST), \
        ("pool_is_empty should return False when NNData.Set.TEST is passed "
         "as an argument and self._test_pool is not empty")
    my_data._test_pool = collections.deque()
    assert my_data.pool_is_empty(NNData.Set.TEST), \
        ("pool_is_empty should return True when NNData.Set.TEST is passed "
         "as an argument and self._test_pool is empty")


def test_get_one_item(bare_constructor: type(NNData.NNData)):
    try:
        bare_constructor.get_one_item = NNData.NNData.get_one_item
    except:
        pytest.fail("Is get_one_item() implemented?")
    my_data = bare_constructor()
    my_data._test_pool = collections.deque([4, 0, 1])
    my_data._train_pool = collections.deque([2, 5, 3])
    my_data._features = numpy.array([[.1], [.2], [.3], [.4], [.5], [.6]],
                                    dtype=float)
    my_data._labels = numpy.array([[1], [2], [3], [4], [5], [6]],
                                  dtype=float)
    assert isinstance(my_data.get_one_item(NNData.Set.TEST), tuple), \
        "get_one_item() should return a tuple"
    assert isinstance(my_data.get_one_item(NNData.Set.TEST)[0],
                      numpy.ndarray), \
        "get_one_item() should return a tuple of two lists"
    assert my_data.get_one_item(NNData.Set.TEST) == ([.2], [2]), \
        ("get_one_item() is returning the wrong items when called with"
         "NNData.Set.TEST")
    assert my_data.get_one_item() == ([.3], [3]), \
        ("get_one_item() is returning the wrong items when called with"
         "no argument")
    assert my_data.get_one_item(NNData.Set.TRAIN) == ([.6], [6]), \
        ("get_one_item() is returning the wrong items when called with"
         "no argument")
    assert my_data.get_one_item(NNData.Set.TEST) is None, \
        ("get_one_item() should return None when there are not more"
         "items in the pool.")
    my_data.get_one_item(NNData.Set.TRAIN)
    assert my_data.get_one_item(NNData.Set.TRAIN) is None, \
        ("get_one_item() should return None when there are not more"
         "items in the pool.")


