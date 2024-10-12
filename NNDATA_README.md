# Objective

Demonstrate an understanding of:

* NumPy Arrays
* Deque
* Random
* Specification

Review these topics:

* Lists of Lists
* Enum Classes
* Custom Errors
* Static Methods

# Specification

We will build a class called NNData with methods that will help us efficiently manage our training and testing data. We will need a couple of Enum classes as helpers. 

## Enum Classes

Our class will need a couple of Enums. These should be coded outside of the NNData class.

* The Order enum will have members SHUFFLE and STATIC.  This will define whether the training data is presented in the same order to the neural network each time, or in random order.
* The Set enum will have members TRAIN and TEST.  At different times as we go along, this enum will help us identify whether we are requesting training set or testing set data.

## Methods inside NNData:

### Static Method percentage_limiter(percentage)
Our class will have one static method, percentage_limiter(percentage), which accepts percentage as a float and returns 0 if percentage < 0, 1 if percentage is > 1, or percentage if 0 <= percentage <= 1.

### __init__()
Our constructor for NNData will have three parameters:

* features should default to None.  It should be passed as a list of lists, with each row representing the features of one example from our data.  If features is None, then features should be set to [] right away in __init__().  Think about why we do this two-step method, rather than set the default to an empty list in the parameter list.
* labels should default to None.  It should be passed as a list of lists, with each row representing one label from our data.  If labels is None, then labels should be set to [] right away in __init__().  Think about why we do this two-step method, rather than set the default to an empty list in the parameter list.
* train_factor is a float that should default to .9, and represents the percentage of the data we want used as our training set.

The constructor will also initialize some internal data:

* self._labels and self._features can be set to None to avoid warnings.
* train_factor should be passed to NNData.percentage_limiter(), and the result assigned to self._train_factor.
* self._train_indices should be initialized as an empty list. We will eventually use this to point to the items in our dataset that make up the training set
* self._test_indices should be initialized as an empty list. We will eventually use this to point to the items in our dataset that make up the testing set
* self._train_pool should be initialized as an empty deque. We will eventually use this to keep track of which training items have not yet been seen in a particular training epoch
* self._test_pool should be initialized as an empty deque. We will eventually use this to keep track of which training items have not yet been seen in a testing run
Finally, the constructor will call load_data, passing features and labels as arguments.

### load_data()
This public method will take two parameters, features and labels.  Both should default to None.  Both should be lists of lists as described above.

If features or labels is None, set self._features and self._labels to None, call split_set() to make sure the indices are adjusted, and return.  We will assume that the client wants to clear the data.
Compare the length of features and labels.  If they don't have the same length, set self._features and self._labels to None. Then call split_set to make sure the indices are adjusted and finally raise a ValueError.
Create numpy arrays from features and labels and assign them to self._features and self._labels.  Specify a datatype of float for both.  If either construction fails, set both self._features and self._labels to None. Then call split_set to make sure the indices are adjusted and finally raise a ValueError.
Call split_set() at the end of load_data() to ensure that the indices line up with newly loaded data.
This method is really our main "setter" and this is a vulnerable point in our code. To make this project really robust, there are other things we should test...for instance, are features and labels both lists of lists? For simplicity, I am only requiring the tests listed above, but feel free to code more and raise a ValueError anytime something goes wrong.

### split_set(self, new_train_factor=None)

This public method will take one parameter, new_train_factor, that should default to None.

If new_train_factor is not None, use it to set self._train_factor. Remember to use percentage_limiter to make sure the value stays within range.
Find the number of examples that are loaded (the size of self._features or self._labels).
Calculate the number of examples that should be used for training
Set up self._train_indices and self._test_indices. These lists will be used as indirect indices for our example data. You may wish to review the video about indirect indices. The indices in self._train_indices should be randomly generated, and the number of indices should be equal to the number of examples used for training, calculated above. self._test_indices should contain the indices that did not appear in self._train_indices.

Suppose you have ten examples loaded, and a train factor of .7, then split_set() might result in the following:

    self._train_indices == [4, 1, 0, 9, 5, 8, 2]
    self._test_indices == [3, 7, 6]

The results should be different each time you run. Note that the numbers 0 to 9 appear in the two sets, representing the ten examples that are loaded. The two lists are mutually exclusive...there is no integer that appears in both lists.

A bit of advice, check early to see if self._features is None. This is not an error, it just means that the number of examples is zero. But it is a special case that might cause problems.

### prime_data(self, target_set=None, order=None)

This method will load one or both deques to be used as indirect indices.

The target_set parameter will dictate whether we are loading self._train_pool (if target_set is Set.TRAIN) or self._test_pool (if target_set is Set.TEST). If target_set is None, load both pools.
Load the pools by copying all data from self._train_indices and/or self._test_indices (if you are not making a direct assignment, be sure to clear the pool first so that you don't end up appending data).
If order is Order.SHUFFLE, shuffle the pool(s) that you just created. If order is None or Order.STATIC, leave the pool(s) in order.

### get_one_item(self, target_set=None)

This method will return exactly one feature/label pair as a tuple.

If target_set is Set.TRAIN or target_set is None, then use self._train_pool to find the pair. If target_set is Set.TEST, use self._test_pool.
Remember that these deques are used as indirect indices into the actual example data. We don't want to return the value from the deque, we want to use the value from the deque to return the correct value from the two numpy arrays.
Use "popleft" so that the index is not reused
Return None if there are no indices left in the chose target_set.

### number_of_samples(self, target_set=None)

This method returns the total number of testing examples (if target set is Set.TEST), or the total number of training examples (if the target set is Set.TRAIN), or both combined (if the target set is None).

### pool_is_empty(self, target_set=None)

target_set is an Enum of type Set

This method returns True if the target_set deque is empty, or False otherwise. If target_set is None, use the train pool.

# Other Requirements
Your assignment will be submitted on GitHub, using the repository you set up in module zero. Save your code in a file called "NNData.py". You should check to make sure that all autograder tests have passed. Do not modify any of the testing code.

There should be a module level docstring, a docstring for each class, and a docstring for each method.

Follow the PEP-8 style guide.

Provide a sample run in a file called "samplerun". This can be code that you ran in the console.
# Testing

Here's some data you can play with.

    XOR_features = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR_labels = [[0], [1], [1], [0]]

Do you recognize it?  That's the XOR function that we studied in the lectures. The first example has a feature [0, 0] and a label of [0]. This is the case where both inputs are False, so the result of XOR should be false. In the second example, [0, 1], the inputs are False and True. So the result is True, or [1]. 

Here I am playing around with this:

    Python 3.11.3 (v3.11.3:f3909b8bc8, Apr  4 2023, 20:12:10) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
    >>> import NNData
    >>> import NNData
    >>> XOR_features = [[0, 0], [0, 1], [1, 0], [1, 1]]
    >>> XOR_labels = [[0], [1], [1], [0]]
    >>> my_data = NNData.NNData(features=XOR_features, labels=XOR_labels)
    # Since we haven't called prime_data, there should be no item to get
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    >>> my_data.prime_data()
    # Now there should be items available
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([0., 0.]), array([0.]))
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([0., 1.]), array([1.]))
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([1., 1.]), array([0.]))
    # We retreived three items. Since train_factor is set to .9 by
    # default, there should only be three times in the training set.
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    # Let's change train_factor to .5
    >>> my_data.split_set(.5)
    >>> my_data.prime_data()
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([1., 1.]), array([0.]))
    # There's still another item available, let's make sure 
    # pool_is_empty() returns False
    >>> my_data.pool_is_empty()
    False
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([0., 0.]), array([0.]))
    # Now we have exhausted the two training items, pool_is_empty()
    # should return True
    >>> my_data.pool_is_empty()
    True
    >>> my_data.prime_data(order=Order.STATIC)
    >>> my_data.prime_data(order=NNData.Order.STATIC)
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([1., 1.]), array([0.]))
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([0., 0.]), array([0.]))
    >>> my_data.prime_data(order=NNData.Order.STATIC)
    # priming data with STATIC, we should see our items show up in the
    # same order as last time.
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([1., 1.]), array([0.]))
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([0., 0.]), array([0.]))
    >>> my_data.prime_data(order=NNData.Order.SHUFFLE)
    # priming data with SHUFFLE, they may show up in a different order.
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([0., 0.]), array([0.]))
    >>> my_data.get_one_item(NNData.Set.TRAIN)
    (array([1., 1.]), array([0.]))
