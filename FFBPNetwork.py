from RMSE import RMSE
from NNData import NNData, Order, Set
from LayerList import LayerList
from Neurode import Neurode
from FFNeurode import FFNeurode
from BPNeurode import BPNeurode
from FFBPNeurode import FFBPNeurode

class EmptySetException(Exception):
    pass

class FFBPNetwork():
        """Implement feed-forward back-propagation network class"""
        def __init__(self, num_inputs: int, num_outputs: int, error_model: type(RMSE)):
            """Initialize LayerList instance, error model, inputs, and outputs."""
            self._list = LayerList(num_inputs, num_outputs, neurode_type=FFBPNeurode)
            self._error_model = error_model
            self._num_inputs = num_inputs
            self.num_outputs = num_outputs

        def add_hidden_layer(self, num_nodes: int, position=0):
            """Add hidden layer if position is greater than zero, move forward through layers."""
            self._list.reset_to_head()
            for _ in range(position):
                if self._list.curr.next is not None:
                    self._list.move_forward()
                else:
                    print("Unable to move forward.")

            self._list.add_layer(num_nodes)
    
        def train(self, data_set: NNData, epochs=1000, verbosity=2, order = Order.SHUFFLE):
            """
            Train data set: for each epoch iteration, create training errors, prime training data.
                Randomize training set as necessary. Retrieve feature-label pair, assign feature values
                to input layer, note progress for every 100 and 1000 epochs.
            """

            if data_set.number_of_samples(Set.TRAIN) == 0:
                raise self.EmptySetException

            for epoch in range(epochs):
                rmse_object = self._error_model()
                data_set.prime_data(Set.TRAIN, order)

                while not data_set.pool_is_empty(Set.TRAIN):
                    features, labels = data_set.get_one_item(Set.TRAIN)

                    for neurode, feature in zip(self._list.input_nodes, features):
                        neurode.set_input(input_value=feature)

                    predicted_values = [neurode.value for neurode in self._list.output_nodes]
                    expected_values = labels

                    rmse_object += (predicted_values, expected_values)

                    for neurode, expected in zip(self._list.output_nodes, expected_values):
                        neurode.set_expected(expected)

                    if verbosity > 0 and epoch % 1000 == 0:
                            print(f"Epoch: {epoch}")
                            print(f"Input: {features} Output: {labels}, Predicted: {predicted_values}")
                        
                    if verbosity > 1  and epoch % 100 == 0:
                            print(f"Epoch: {epoch}")
                            print(f"Input: {features} Output: {labels}, Predicted: {predicted_values}")
                            print(f"RMSE: {rmse_object.error}")
        
            print(f"Final RMSE value report: {rmse_object.error}")

        def test(self, data_set: NNData, order=Order.STATIC):
            """Utilize testing set to track testing progress.
            record RMSE, print input, expected output, predicted output.
            """
            if data_set.number_of_samples(Set.TEST) == 0:
                raise self.EmptySetException

            rmse_object = self._error_model()

            data_set.prime_data(Set.TEST, order)

            while not data_set.pool_is_empty(Set.TEST):
                features, labels = data_set.get_one_item(Set.TEST)
                for neurode, feature in zip(self._list.input_nodes, features):
                    neurode.set_input(feature)

                predicted_values = [neurode.value for neurode in self._list.output_nodes]
                expected_values = labels
                rmse_object += (predicted_values, expected_values)

            print(f"(test) Input: {features} ")
            print(f"(test) Output: {labels}")
            print(f"(test) Predicted: {predicted_values}")
            print(f"(test) Final RMSE: {rmse_object.error}")

"""
"""