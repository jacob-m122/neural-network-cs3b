import pytest
from DoublyLinkedList import DoublyLinkedList

def test_add_to_head():
    dl_list = DoublyLinkedList()
    dl_list.add_to_head(5)
    dl_list.add_to_head(10)

    assert dl_list._head.data == 10, "Head updated to 10"
    assert dl_list._head.data.next.data == 5, "Next node following head is 5"
    assert dl_list.head.next.prev == dl_list._head, "Prev of next node points to the head"

def test_remove_from_head():
    dl_list = DoublyLinkedList()
    dl_list.add_to_head(5)
    dl_list.add_to_head(10)

    removed = dl_list.remove_from_head()
    assert removed == 10

def test_move_forward():
    dl_list = DoublyLinkedList()
    dl_list.move_forward(5)
    dl_list.move_forward(10)

def test_move_backward():
    dl_list = DoublyLinkedList()
    
def test_is_empty():
    dl_list = DoublyLinkedList()

def test_add_after_current():
    dl_list = DoublyLinkedList()

def test_remove_after_current():
    dl_list = DoublyLinkedList()

def test_find():
    dl_list = DoublyLinkedList()

def test_remove():
    dl_list = DoublyLinkedList()

