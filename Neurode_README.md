# Objective

Demonstrate an understanding of:

* Inheritance
* Abstract Base Class
* Binary Encoding
* Binary Arithmetic


Review these topics:

* Class attributes
* Enum classes
* Inner classes

# Specification

We are working toward a final class called FFBPNeurode.  FFBP stands for Feed-forward backpropogation and describes the way that our neural network will make and improve predictions.


Over the next three assignments, we will write a total of five classes. For this assignment we will write the first two. The work for this assignment (both classes) should be saved in a file called Neurode.py

Note: In order to use a class that you are working on as a typehint within the same class, you need:

    from \__future\__ import annotations

at the top of your file.

## class MultiLinkNode

This is an abstract base class that will be the starting point for our eventual FFBPNeurode class.

MultiLinkNode will contain one inner class and several methods:

### class Side(Enum)

This is an Enum inner class of MultiLinkNode with elements UPSTREAM and DOWNSTREAM. We will use these terms to identify relationships between neurodes. We are coding this as an inner class because Side will only be used by instances of the MultiLinkNode class and its subclasses.

### def \__init\__(self)

In the constructor we will initialize three instance attributes:

* **self._reporting_nodes** is a dictionary with two entries. The keys will be the two elements of the Side enum. The values will initially be set to zero. We will use this as a binary encoding to keep track of which neighboring nodes have indicated that they have information available
* **self._reference_value** is also a dictionary with two entries. The keys will again be the two elements of the Side enum. The values will initially be set to zero, but will represent what the reporting nodes value should be as a binary encoding when all of the nodes have reported. 
* **self._neighbors** is also a dictionary with two entries. The keys will again be the two elements of the Side enum. The values will initially be set to empty lists, but eventually will contain references to the neighboring nodes upstream and downstream.

### def \__str\__(self)

Overload this function to return a string representation of the node in context. In an attractive and easy to comprehend format, print the ID of the node and the ID's of the neighboring nodes upstream and downstream. The specific implementation is up to you.

### def _process_new_neighbor(self, node: MultiLinkNode, side: Side)

This is an abstract method that takes a node, and a Side Enum, as parameters. We will implement it in the Neurode class below.

### def reset_neighbors(self, nodes: list, side: Side)

This method accepts nodes as a list, and side as a Side Enum. It will store a reference to the nodes that link into this node either upstream or downstream. 

Using the nodes parameter, populate the appropriate entry of self._neighbors (UPSTREAM or DOWNSTREAM), making sure that any previous value is cleared in the process. Think carefully about whether this should be an assignmnet, a copy, or a deepcopy. Two considerations as you decide:
* The client could modify and reuse the nodes list, which has the potential to corrupt self._neighbors.
* We don't want to create new MultiLinkNodes, we want references to nodes that already exist.

Choose wisely.

![Indiana Jones GIF "You have chosen wisely"](img.png)

* Call _process_new_neighbor() for each node.
* Calculate and store the appropriate value in the correct element of self._reference_value.

>*Notes about self._reporting_nodes and self._reference_value:*
> 
>Most of the time, a neurode will be waiting for its upstream or downstream neighbors to be ready with information. The neighbors will report in one at a time, but the neurode doesn't do its work until all of its neighbors are ready on one side.
> 
>We need to establish an efficient way to determine whether all the neighboring neurodes are ready with information. We will do this by maintaining a single integer that represents a binary encoding of the reporting nodes. For example, if there are two nodes on the upstream side:
> 
>A reference value of 11 (binary) will be calculated in reset_neighbors. This is a bit of an expensive calculation, so we do it only once and store the result.
As we test or train, self._reporting_nodes will be updated:
> 
>Each neighboring node's position (integer index) in self._neighbors will determine its bit position in the encoding. The zeroth node in self._neighbors is represented by the 2^0 place in the encoding, and so on.
>>01 (or 1) represents that the zeroth node has reported
>>
>>10 (or 2) represents that the first node has reported
>>
>>11 (or 3) represents that both nodes have reported. This equals the reference value, so we know that all nodes have reported.


## Class Neurode

This class is inherited from and implements MultiLinkNode. It has four methods, plus some properties.

Include a class attribute **Neurode._learning_rate** that is set to .05.

### def learning_rate(...)
This is actually a pair of methods, a setter and a getter coded with the @property decorator. They should get or set the class attribute that you defined above. In your testing, make sure that this truly behaves as a class attribute. In other words, changing the learning rate for one object of the class should change the learning rate for all objects of the class.

### def \__init\__(self)

In the constructor we will initialize two attributes:

* self._value represents the current value of the neurode. Initialize to zero.
* self._weights is a dictionary representing the weights given to the upstream connections. The keys will be references to the upstream neurodes, and the values will be floats representing the weights. Initialize this to an empty dictionary.

Be sure to call the parent class constructor.

### def _process_new_neighbor(self, node: Neurode, side: Side)

This method will be called when a new neigboring node is added on either side, but we really are only concerned with upstream neighbors. When an upstream neighbor is added, the node reference will be added as a key to the self._weights dictionary. The value related to this key should be a randomly generated weight between 0 and 1. This weight represents the importance of the neighboring nodes to this node, and will be adjusted up or down during training.

There is bit of sloppiness here, as we have no process to remove a key from the dictionary if the upstream nodes are later changed. This is rare enough that it will not cause any performance issues, and addressing this issue will make the code needlessly complicated. One possible solution would be to override reset_neighbors in the Neurode class to clear out the dictionary entry.

### def _check_in(self, node: Neurode, side: Side)

This method will be called whenever the node learns that a neighboring node has information available. For credit, this method and the 

* Find the node's index in self._neighbors.
* Use this index to update self._reporting_nodes to reflect that this node has reported.
* Compare self._reporting_nodes to self._reference_value to determine whether all neighboring nodes have reported. If so, reset self._reporting_nodes to zero and return True.
* Otherwise, return False

### def get_weight(self, node: Neurode)

During backpropogation, each upstream node will need to know how important it is to our current node...represented by the weight of the incoming connection. The upstream node will pass in a reference to itself, and our current node will look up the upstream node's associated weight in the self._weights dictionary, and return that value. While this method is a getter, we can't use a @property because of the node parameter.

### Other Properties:

Create a property (getter only) for self._value (called value)

