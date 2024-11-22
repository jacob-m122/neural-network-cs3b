import RMSE
from RMSE import Euclidean
from RMSE import Taxicab

my_error = Taxicab()
my_error += ((.1, .7, .3, .9), (0, 1, 0, 0))
my_error += ((.2, .1, .8, .1), (0, 0, 1, 0))
my_error += ((.5, .5, .1, .7), (0, 0, 0, 1))
my_error += ((.2, .6, .1, .9), (0, 0, 0, 1))
print(my_error.error)
my_error.reset()
print(my_error.error)

my_error = Euclidean()
my_error += ((.1, .7, .3, .9), (0, 1, 0, 0))
my_error += ((.2, .1, .8, .1), (0, 0, 1, 0))
my_error += ((.5, .5, .1, .7), (0, 0, 0, 1))
my_error += ((.2, .6, .1, .9), (0, 0, 0, 1))
print(my_error.error)
my_error.reset()
print(my_error.error)
