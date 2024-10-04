from DoublyLinkedList import DoublyLinkedList

dl_list = DoublyLinkedList()
try:
    dl_list.remove_from_head()
except IndexError:
    print("remove_from_head() raised an IndexError on empty list.")

try:
    dl_list.move_forward()
except IndexError:
    print("move_forward() raised an IndexError on empty list.")


dl_list.add_to_head(5)
assert dl_list.curr_data == 5
assert dl_list.remove_from_head() == 5
assert dl_list.is_empty()

try:
    dl_list.move_forward()
except IndexError:
    print("move_forward() raised an IndexError on a single node list.")

dl_list.add_to_head(5)
dl_list.add_to_head(10)
dl_list.add_to_head(15)

dl_list.reset_to_tail()
try:
    dl_list.move_forward()
except IndexError:
    print("move_forward() at tail raised an IndexError.")

dl_list.reset_to_head()
try:
    dl_list.move_backward()
except IndexError:
    print("move_backward() at head raised an IndexError.")
