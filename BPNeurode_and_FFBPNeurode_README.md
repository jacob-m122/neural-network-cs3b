# Objective

Demonstrate an understanding of:

* Multiple Inheritance
* Method Resolution Order
* Sigmoid Derivative
* Backpropogation

# Specification

We will now work on our final two classes. The last one is so easy!

For this assignment we will create two files, one for each class:
* BPNeurode.py
* FFBPNeurode.py

## Class BPNeurode(Neurode)

This class is inherited from Neurode. We will add some methods specifically to deal with the backpropogation process that we use only in training. Information passes from downstream nodes to upstream nodes ("right to left") with ths goal of adjusting the weights to make a better prediction with the next feedforward pass.

### def \__init\__(self)

Initialize self._delta to zero. self._delta represents the prediction error attributable to this Neurode.

Do we need a call to super()?

### def _sigmoid_derivative(value: float)

This should be a static method, and calculates the derivative using the simplified formula f(x)* (1 - f(x)). Note, we have already caluculated f(x), that's the value of the neurode. It would be costly to do so again. Therefore, the input/argument for this function should be f(x), the value we have already calculated. Definitely don't overthink this function.

### def _calculate_delta(self, expected_value:float = None)

The expected_value parameter is only used for output layer nodes.

Calculate the delta of this neurode as described in the lectures. Note that there are different cases for hidden/input layer vs. output layer neurodes. The neurode doesn't know its type, but think about how we can tell if this is an output layer neurode. In either case, save the result to self._delta. This method does not return any value.

### def data_ready_downstream(self, node: Neurode)

This method is very similar to the data_ready_upstream method that we coded for FFNeurode.

Downstream neurodes call this method when they have data ready. It should:

* Call self._check_in() to register that the node has data.
* If self._check_in() indicates that all downstream nodes have data, it is time to collect that data and make it available to the next layer up. Call self._calculate_delta() and then self._fire_upstream(). Finally, call _update_weights(). 

The order is important here, and you should think carefully about how control is jumping around the network. We want all the nodes to calculate delta before any of them updates weight...otherwise the new weights throw off the intended delta calculation.

### def set_expected(self, expected_value: float)

This method is used by the client to directly set the expected value of an output layer neurode. The neurode should call self._calculate_delta to calculate and save its own delta...be sure to pass expected_value as an argument. Call data_ready_downstream() on all of the upstream neighbors, passing self as an argument.

### def adjust_weights(self, node: Neurode, adjustment: float)

This method is called by an upstream node that is requesting to be more or less "important." Use the **node** reference to add **adjustment** to the appropriate entry of self._weights.

### def _update_weights(self)

This method is the partner of adjust weights, and is called after delta is calculated. This node will iterate through its downstream neighbors, and use adjust_weights to request an adjustment to the weight (importance) given to this node's data. This is another (unusual) case where we will be passing self explicitly as an argument, to let the downstream node know who is calling. Remember that the weight adjustment uses the downstream node's delta and learning rate, together with this node's value.

### def _fire_upstream(self)

This method is very similar to the _fire_downstream method that we coded for FFNeurode.

Call data_ready_downstream on each of this node's upstream neighbors, using self as an argument. Note that this is safe for input nodes, as there are no upstream neighbors and the loop will never be entered. Once more, this is an unusual example where we use "self" as an explicit argument.

### Property:

Create a @property for self._delta, just a getter.

## class FFBPNeurode()

After all that work, this will truly be easy. This class will inherit from both FFNeurode and BPNeurode. Does the order matter? Do we need a constructor? What about calls to super()?

Remember that this class should be in its own file.

