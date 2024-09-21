"""Testing for Asst Five"""
import pytest

#
# DO NOT MODIFY THIS CODE
#
try:
    import DoublyLinkedList
except:
    pytest.fail("DoublyLinkedList.py file is not present")


@pytest.fixture
def my_dll_empty():
    test_object = DoublyLinkedList.DoublyLinkedList()
    return test_object


@pytest.fixture
def my_dll_one_item():
    test_object = DoublyLinkedList.DoublyLinkedList()
    test_object.add_to_head("Cat")
    return test_object


@pytest.fixture
def my_dll_three_items():
    test_object = DoublyLinkedList.DoublyLinkedList()
    test_object.add_to_head(10)
    test_object.add_to_head(20)
    test_object.add_to_head(30)
    return test_object


def test_DLLNode():
    """Test all functionality of DLLNode."""
    with pytest.raises(TypeError):
        DoublyLinkedList.DLLNode()
        pytest.fail("DLLNode should have a required parameter")
    try:
        DoublyLinkedList.DLLNode("testdata")
    except:
        pytest.fail("Could not create DLLNode. Check that the class"
                    "is names correctly.")


def test_property(my_dll_one_item):
    assert my_dll_one_item.curr_data == "Cat", \
        ("Wrong value returned for head after adding one item. Most "
         "likely causes are missing or incorrect curr_data property, "
         "or malfunctioning add_to_head.")


def test_move_forward(my_dll_three_items):
    assert my_dll_three_items.curr_data == 30, \
        ("Wrong value returned for head after adding three items. Most "
         "likely causes are missing or incorrect curr_data property, "
         "or malfunctioning add_to_head.")
    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 20, \
        ("Wrong value returned for second node after adding three items. "
         "Most likely causes are malfunctioning move_forward, or "
         "malfunctioning add_to_head.")
    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 10, \
        ("Wrong value returned for third node after adding three items. "
         "Most likely cause is malfunctioning add_to_head.")


def test_reset_to_head(my_dll_three_items):
    my_dll_three_items.move_forward()
    my_dll_three_items.reset_to_head()
    assert my_dll_three_items.curr_data == 30, \
        "Wrong value returned after resetting to head."
    assert my_dll_three_items.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_reset_to_tail(my_dll_three_items):
    my_dll_three_items.reset_to_tail()
    assert my_dll_three_items.curr_data == 10, \
        "Wrong value returned after resetting to tail."
    assert my_dll_three_items.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_move_back(my_dll_three_items):
    my_dll_three_items.reset_to_tail()
    try:
        my_dll_three_items.move_back()
    except AttributeError:
        my_dll_three_items.move_backward()
    assert my_dll_three_items.curr_data == 20, \
        ("Wrong value returned for second node after adding three items. "
         "Most likely causes are malfunctioning move_back, or "
         "malfunctioning add_to_head.")
    try:
        my_dll_three_items.move_back()
    except AttributeError:
        my_dll_three_items.move_backward()
    assert my_dll_three_items.curr_data == 30, \
        ("Wrong value returned for second node after adding three items. "
         "Most likely causes are malfunctioning move_back, or "
         "malfunctioning add_to_head.")


def test_move_forward_past_tail(my_dll_one_item):
    with pytest.raises(IndexError):
        my_dll_one_item.move_forward()
    assert my_dll_one_item.curr_data == "Cat", \
        "curr_data is not correct after move_forward at tail."


def test_move_back_past_head(my_dll_one_item):
    with pytest.raises(IndexError):
        try:
            my_dll_one_item.move_back()
        except AttributeError:
            my_dll_one_item.move_backward()


def test_add_after_current_no_head(my_dll_empty):
    with pytest.raises(IndexError):
        my_dll_empty.add_after_current("Cat")
    assert my_dll_empty.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_add_after_current_one_item_in_list(my_dll_one_item):
    my_dll_one_item.add_after_current("Dog")
    assert my_dll_one_item.curr_data == "Cat", \
        "Wrong original item returned after add_after_current."
    my_dll_one_item.move_forward()
    assert my_dll_one_item.curr_data == "Dog", \
        "Wrong new item returned after add_after_current, move forward."
    with pytest.raises(IndexError):
        my_dll_one_item.move_forward()
    try:
        my_dll_one_item.move_back()
    except AttributeError:
        my_dll_one_item.move_backward()
    assert my_dll_one_item.curr_data == "Cat", \
        ("Wrong original item returned after add_after_current, move "
         "forward and back.")
    assert my_dll_one_item._tail.data == "Dog", \
        "Wrong item at tail. Was self._tail updated?"
    with pytest.raises(IndexError):
        try:
            my_dll_one_item.move_back()
        except AttributeError:
            my_dll_one_item.move_backward()
    assert my_dll_one_item.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_add_after_current_three_items_in_list(my_dll_three_items):
    my_dll_three_items.add_after_current(100)
    assert my_dll_three_items.curr_data == 30, \
        "Wrong original item returned after add_after_current."
    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 100, \
        "Wrong new item returned after add_after_current, move forward."
    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 20, \
        ("Wrong original item returned after add_after_current, move "
         "forward twice.")
    try:
        my_dll_three_items.move_back()
    except AttributeError:
        my_dll_three_items.move_backward()

    assert my_dll_three_items.curr_data == 100, \
        ("Wrong new item returned after add_after_current, move "
         "forward twice and back once.")
    try:
        my_dll_three_items.move_back()
    except AttributeError:
        my_dll_three_items.move_backward()
    assert my_dll_three_items.curr_data == 30, \
        ("Wrong original item returned after add_after_current, move "
         "forward twice and back twice.")
    assert my_dll_three_items._tail.data == 10, \
        "Wrong item at tail. Was self._tail updated?"
    assert my_dll_three_items.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_remove_from_head_no_items(my_dll_empty):
    with pytest.raises(IndexError):
        my_dll_empty.remove_from_head()
    assert my_dll_empty.is_empty() is True, \
        "is_empty should return True if list is empty."


def test_remove_from_head_one_item(my_dll_one_item):
    my_dll_one_item.remove_from_head()
    with pytest.raises(IndexError):
        my_dll_one_item.curr_data
    my_dll_one_item.reset_to_tail()
    with pytest.raises(IndexError):
        my_dll_one_item.curr_data
    my_dll_one_item.add_to_head("Dog")
    assert my_dll_one_item.curr_data == "Dog", \
        ("Wrong item returned after removing only item from head and "
         "adding an item back in.")
    assert my_dll_one_item._tail is None, \
        "Tail should point to None when list is empty."
    assert my_dll_one_item.is_empty() is True, \
        "is_empty should return True if list is empty."


def test_remove_from_head_three_items(my_dll_three_items):
    my_dll_three_items.move_forward()
    my_dll_three_items.move_forward()
    my_dll_three_items.remove_from_head()
    assert my_dll_three_items.curr_data == 20, \
        "Was self._curr reset to head?"
    with pytest.raises(IndexError):
        try:
            my_dll_three_items.move_back()
        except AttributeError:
            my_dll_three_items.move_backward()

    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 10, \
        "Possible issue with next/prev pointers after remove_from_head."
    assert my_dll_three_items._tail.data == 10, \
        "Wrong item at tail. Was self._tail updated?"
    assert my_dll_three_items.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_find_empty_list(my_dll_empty):
    with pytest.raises(IndexError):
        my_dll_empty.find(10)


def test_is_empty_on_empty_list(my_dll_empty):
    assert my_dll_empty.is_empty() is True, \
        "is_empty should return True if list is empty."


def test_is_empty_on_non_empty_list(my_dll_one_item):
    assert my_dll_one_item.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_find_one_item(my_dll_one_item):
    with pytest.raises(IndexError):
        my_dll_one_item.find(10)
    assert my_dll_one_item.find("Cat") == "Cat", \
        "Did not find item at head of one-item-list"


def test_find_three_items(my_dll_three_items):
    with pytest.raises(IndexError):
        my_dll_three_items.find("Cat")
    assert my_dll_three_items.find(30) == 30, \
        "Did not find item at head of three-item-list"
    assert my_dll_three_items.find(20) == 20, \
        "Did not find item inside three-item-list"
    assert my_dll_three_items.find(10) == 10, \
        "Did not find item at tail of three-item-list"


def test_remove_empty_list(my_dll_empty):
    with pytest.raises(IndexError):
        my_dll_empty.remove(10)
    assert my_dll_empty.is_empty() is True, \
        "is_empty should return True if list is empty."


def test_remove_one_item(my_dll_one_item):
    assert my_dll_one_item.remove("Cat") == "Cat", \
        "Did not find item at head of one-item-list"
    my_dll_one_item.reset_to_head()
    with pytest.raises(IndexError):
        my_dll_one_item.curr_data
    my_dll_one_item.reset_to_tail()
    with pytest.raises(IndexError):
        my_dll_one_item.curr_data
    with pytest.raises(IndexError):
        my_dll_one_item.find("Cat")
    assert my_dll_one_item._tail is None, \
        "Tail should point to None when list is empty."
    assert my_dll_one_item.is_empty() is True, \
        "is_empty should return True if list is empty."


def test_remove_head_three_items(my_dll_three_items):
    with pytest.raises(IndexError):
        my_dll_three_items.remove("Cat")
    assert my_dll_three_items.remove(30) == 30, \
        "Did not find item to remove at head of three-item-list"
    assert my_dll_three_items.curr_data == 20, \
        "List was not reset to head or head item is incorrect."
    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 10, \
        "Item after head is not correct."
    my_dll_three_items.reset_to_head()
    assert my_dll_three_items.curr_data == 20, \
        "Head is not correct after remove()."
    my_dll_three_items.reset_to_tail()
    assert my_dll_three_items.curr_data == 10, \
        "Tail is not correct after remove()."
    assert my_dll_three_items.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_remove_middle_three_items(my_dll_three_items):
    assert my_dll_three_items.remove(20) == 20, \
        "Did not find item to remove at head of three-item-list"
    assert my_dll_three_items.curr_data == 30, \
        "List was not reset to head or head item is incorrect."
    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 10, \
        "Item after head is not correct after remove()."
    my_dll_three_items.reset_to_head()
    assert my_dll_three_items.curr_data == 30, \
        "Head is not correct after remove()."
    my_dll_three_items.reset_to_tail()
    assert my_dll_three_items.curr_data == 10, \
        "Tail is not correct after remove()."
    assert my_dll_three_items.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_remove_tail_three_items(my_dll_three_items):
    assert my_dll_three_items.remove(10) == 10, \
        "Did not find item to remove at tail of three-item-list"
    assert my_dll_three_items.curr_data == 30, \
        "List was not reset to head or head item is incorrect."
    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 20, \
        "Item after head is not correct after remove()."
    my_dll_three_items.reset_to_head()
    assert my_dll_three_items.curr_data == 30, \
        "Head is not correct after remove()."
    my_dll_three_items.reset_to_tail()
    assert my_dll_three_items.curr_data == 20, \
        "Tail is not correct after remove()."
    try:
        my_dll_three_items.move_forward()
        pytest.fail("Target node can still be reached beyond the tail.")
    except IndexError:
        pass
    assert my_dll_three_items.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_remove_after_current_empty_list(my_dll_empty):
    with pytest.raises(IndexError):
        my_dll_empty.remove_after_current()
    assert my_dll_empty.is_empty() is True, \
        "is_empty should return True if list is empty."


def test_remove_after_current_one_item(my_dll_one_item):
    with pytest.raises(IndexError):
        my_dll_one_item.remove_after_current()
    assert my_dll_one_item.find("Cat") == "Cat", \
        "Item that should not have been removed cannot be found."
    assert my_dll_one_item.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_remove_after_current_middle_item_three_items(my_dll_three_items):
    assert my_dll_three_items.remove_after_current() == 20, \
        "expected value not returned from remove_after_current()"
    assert my_dll_three_items.curr_data == 30, \
        "Head item is incorrect after remove_after_current()."
    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 10, \
        "Item after head is not correct after remove_after_current()."
    my_dll_three_items.reset_to_head()
    assert my_dll_three_items.curr_data == 30, \
        "Head is not correct after remove_after_current()."
    my_dll_three_items.reset_to_tail()
    assert my_dll_three_items.curr_data == 10, \
        "Tail is not correct after after remove_after_current()."
    assert my_dll_three_items.is_empty() is False, \
        "is_empty should return False if list is not empty."


def test_remove_after_current_last_item_three_items(my_dll_three_items):
    my_dll_three_items.move_forward()
    assert my_dll_three_items.remove_after_current() == 10, \
        "expected value not returned from remove_after_current()"
    assert my_dll_three_items.curr_data == 20, \
        ("Wrong item return from curr_data after move_forward() and"
         "remove_after_current().")
    with pytest.raises(IndexError):
        my_dll_three_items.move_forward()
    my_dll_three_items.reset_to_head()
    assert my_dll_three_items.curr_data == 30, \
        "Head is not correct after remove_after_current()."
    my_dll_three_items.move_forward()
    assert my_dll_three_items.curr_data == 20, \
        "Item after head is not correct after remove_after_current()."
    my_dll_three_items.reset_to_tail()
    assert my_dll_three_items.curr_data == 20, \
        "Tail is not correct after remove_after_current()."
    assert my_dll_three_items.is_empty() is False, \
        "is_empty should return False if list is not empty."
