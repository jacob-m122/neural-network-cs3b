""" Implementing the Linked List ADT. """


class Node:
    """ Linked List Node """
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """ Implementing Linked List ADT """
    def __init__(self):
        self._head = None
        self._curr = None

    def reset_to_head(self):
        """ Reset the current pointer to head. """
        self._curr = self._head

    def add_to_head(self, data):
        """ Add a new node to the head of the list. """
        new_node = Node(data)
        new_node.next = self._head
        self._head = new_node
        self.reset_to_head()

    def remove_from_head(self):
        """ Remove a node from the head of the list and return data. """
        if not self._head:
            raise IndexError
        return_value = self._head.data
        self._head = self._head.next
        self.reset_to_head()
        return return_value

    def move_forward(self):
        """ Move forward through the list. """
        if not self._curr or not self._curr.next:
            raise IndexError
        self._curr = self._curr.next

    @property
    def curr_data(self):
        """ Return the data at the current position. """
        if not self._curr:
            raise IndexError
        return self._curr.data

    def add_after_current(self, data):
        """ Add a node after the current position. """
        if not self._curr:
            raise IndexError
        new_node = Node(data)
        new_node.next = self._curr.next
        self._curr.next = new_node

    def remove_after_current(self):
        """ Remove the node after the current node, returning data. """
        if not self._curr or not self._curr.next:
            raise IndexError
        return_value = self._curr.next.data
        self._curr.next = self._curr.next.next
        return return_value

    def find(self, data):
        """ Find and return an item in the list. """
        temp_curr = self._head
        while temp_curr:
            if temp_curr.data == data:
                return temp_curr.data
            temp_curr = temp_curr.next
        raise IndexError

    def remove(self, data):
        """ Find and remove a node. """
        if not self._head:
            raise IndexError
        if self._head.data == data:
            return self.remove_from_head()
        temp_curr = self._head
        while temp_curr.next:
            if temp_curr.next.data == data:
                return_value = temp_curr.next.data
                temp_curr.next = temp_curr.next.next
                self._curr = temp_curr
                return return_value
            temp_curr = temp_curr.next
        raise IndexError
