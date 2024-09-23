[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=16098688)
# Objective

Demonstrate an understanding of:

* Containers and Abstract Data Types
* Linked List Concepts

# Specification

You will develop a Doubly-Linked List class based on the concepts that you have learned about linked lists. Do not inherit from the Linked List class, but feel free to duplicate any code from that class.

Start with a DLLNode class, similar to the Node class. The DLLNode class should have a self.prev attribute in addition to the attributes that a linked list node has.

Now let's work on the DoublyLinkedList class. First off, you will need to add a protected _tail attribute. Your code should implement the following public methods in a class called DoublyLinkedList. Some of these will identical to the LinkedList code. Others are trivially different. Still others require quite a bit of thought to implement the self.prev attribute. 

The new methods should behave in the same way that the LinkedList methods behaved. For example, remove_after_current() should return data.

* add_to_head(data)
  * self._curr should reset to head after this action
* add_after_current(data)
* remove_from_head(data)
  * self._curr should reset to head after this action
* remove_after_current()
* reset_to_head()
* reset_to_tail()
* move_forward()
* move_backward()
* find(data)
* remove(data)
  * self._curr should reset to head after this action
* curr_data()
  * This should be coded as a @property


# Other Requirements

Your assignment will be submitted on GitHub. Save your code in a file called "DoublyLinkedList.py". You should check to make sure
that all autograder tests have passed. Do not modify any of the testing code.

There should be a module level docstring, a docstring for each class, and a
docstring for each method.

Follow the PEP-8 style guide.

Provide a sample run in a file called "samplerun". This can be code that you ran in the console.