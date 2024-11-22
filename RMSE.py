"""Implement RMSE, as well as subsequent Euclidean and Taxicab child classes."""
from abc import ABC, abstractmethod
import math

class RMSE(ABC):
    """Implement RMSE class."""

    def __init__(self):
        """Initialize squared distance sum and count."""
        self._squared_distances_total = 0
        self._count = 0

    def __add__(self, total_dataset):
        """Overload + operator for adding values to RMSE object."""
        predicted_value, expected_value = total_dataset
        distance = self.distance(predicted_value, expected_value)
        new_object = self.__class__()
        new_object._squared_distances_total = self._squared_distances_total + distance ** 2
        new_object._count = self._count + 1
        return new_object

    def __iadd__(self, total_dataset):
        """Overload += operator for adding values to RMSE object."""
        predicted_value, expected_value = total_dataset
        distance = self.distance(predicted_value, expected_value)
        self._squared_distances_total += distance ** 2
        self._count += 1
        return self

    def reset(self):
        """Reset RMSE object."""
        self._squared_distances_total = 0
        self._count = 0

    @property
    def error(self):
       """Calculate and return RMSE value."""
       if self._count > 0:
          mean_error_squared = self._squared_distances_total / self._count
          rmse = math.sqrt(mean_error_squared)
          return rmse
       else:
            return 0

    @staticmethod
    @abstractmethod
    def distance(predicted_value, expected_value):
        """Declare abstract method."""
        pass


class Euclidean(RMSE):
    """Implement Euclidiean child class, inherit from RMSE."""

    @staticmethod
    def distance(predicted_value, expected_value):
        """Calculate Euclidian distance of predicted and expected values."""
        squares_sum = 0
        for predicted, expected in zip(predicted_value, expected_value):
            diff = predicted - expected
            diff_squared = diff ** 2
            squares_sum += diff_squared
        distance = math.sqrt(squares_sum)
        return distance


class Taxicab(RMSE):
    """Implement Taxicab child class, inherit from RMSE."""

    @staticmethod
    def distance(predicted_value, expected_value):
        """Calculate taxicab distance between predicted and expected values."""
        distance_total = 0
        for predicted, expected in zip(predicted_value, expected_value):
            abs_diff = abs(predicted - expected)
            distance_total += abs_diff
        return distance_total

"""
1.2124355652982142
0
0.7280109889280518
0
"""
