# Objective

Demonstrate an understanding of:

* Inheritance
* The self object
* Sigmoid Function and its Derivative
* Delta
* Feedforward

# Specification

This class is fairly simple, but there is a tricky bit that will require you to really understand what the self object is.

Import the Neurode class that you have already created. So that we don't have to do crazy things like Neurode.Neurode.Side.UPSTREAM, use:

    from Neurode import Neurode

Note: In order to use a class that you are working on as a typehint within the same class, you need:

    from \__future\__ import annotations

at the top of your file.

The work for this assignment should be saved in a file called FFNeurode.py

## class FFNeurode(Neurode)

This class is inherited from Neurode. We will add some methods specifically to deal with the feed-forward process that we use in testing and training. Information passes from upstream nodes to downstream nodes ("left to right") with ths goal of making a label prediction at the output layer based on the features provided at the input layer. 

### def \__init\__(self)

There's really nothing we need to do in the constructor. Do we need a call to super()? What happens if we completely leave the constructor out?

There are four other methods to code:

### def _sigmoid(value: float)

This should be a static method, and should return the result of the sigmoid function at value. You will need to use an exponential function, which will require an import. You can use the exp function from either math or numpy.

### def _calculate_value(self)

Calculate the weighted sum of the upstream nodes' values. Pass the result through self._sigmoid() and store the returned value into self._value.

### def _fire_downstream(self)

Call data_ready_upstream on each of this node's downstream neighbors, using self as an argument. Note that this is safe for output nodes, as there are no downstream neighbors and the loop will never be entered.

> A note on self: When calling data_ready_upstream (and it's backpropogation twin later), we will explicitly pass the self object. This is unusual, and you should take time to really understand what is going on. Our neurode is passing a reference to itself as an argument to another neurode. It's like an introduction, "Hi, I'm Sarah." It will let the other neurode know which of its upstream neurodes has information available. Note also that this is just a reference to the self object (like a memory address), and not the whole object...so it doesn't take up a bunch of memory.

### def data_ready_upstream(self, node: Neurode)

Upstream neurodes call this method when they have data ready. It should:

* Call self._check_in() to register that the node has data.
* If self._check_in() indicates that all upstream nodes have data, it is time to collect that data and make it available to the next layer. In this case, call self._calculate_value() and then self._fire_downstream().

### def set_input(self, input_value: float)

This method is used by the client to directly set the value of an input layer neurode. The neurode does not need to do any processing, so it should simply set self._value, and then call data_ready_upstream() on all of the downstream neighbors, passing self as an argument.

