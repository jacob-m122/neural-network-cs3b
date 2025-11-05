from RMSE import RMSE
from NNData import NNData, Order, Set
from LayerList import LayerList
from Neurode import Neurode
from FFNeurode import FFNeurode
from BPNeurode import BPNeurode
from FFBPNeurode import FFBPNeurode
from typing import Type, Callable, Dict, List, Optional, Tuple
import logging, math, random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EmptySetException(Exception):
    """Implement custom exception class."""
    pass

class FFBPNetwork():
    """Implement feed-forward back-propagation network class"""

    def __init__(self, num_inputs: int, num_outputs: int, error_model: Type[RMSE], seed: Optional[int] = None):

        """
        Initialize LayerList instance, error model, inputs, and outputs.
        seed: optional RNG seed for reproducibility of weight/bias init (if upstream uses random).
        """

        if seed is not None:
            random.seed(seed)
        self._list = LayerList(num_inputs, num_outputs, neurode_type=FFBPNeurode)
        self._error_model = error_model
        self._num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._output_activation = "softmax"

    def add_hidden_layer(self, num_nodes: int, position=0):
        """Add hidden layer if position is greater than zero, move forward through layers."""
        self._list.reset_to_head()
        for _ in range(position):
            if self._list.curr.next is not None:
                self._list.move_forward()
            else:
                print("Unable to move forward.")
        self._list.add_layer(num_nodes)

    # ---- small helper to optionally compute accuracy for one-hot labels ----

    @staticmethod
    def _maybe_accuracy(predicted: List[float], expected: List[float]) -> Optional[bool]:
        """
        Returns True/False if both look like one-hot vectors (same length >=2),
        otherwise returns None to indicate 'no accuracy for this sample'.
        """
        if not isinstance(predicted, list) or not isinstance(expected, list):
            return None
        if len(predicted) != len(expected) or len(expected) < 2:
            return None
        # "Looks" like one-hot: expected has a clear argmax at a 1.0 (or near it)
        e_max_i = max(range(len(expected)), key=lambda i: expected[i])
        # tolerate slight float noise (expected often exactly 0/1 in your pipeline)
        is_one_hotish = abs(expected[e_max_i] - 1.0) < 1e-6 and sum(1 for v in expected if abs(v) > 1e-6) == 1
        if not is_one_hotish:
            return None
        p_max_i = max(range(len(predicted)), key=lambda i: predicted[i])
        return p_max_i == e_max_i

    def train(self, data_set: NNData, epochs=1000, verbosity=1, order=Order.SHUFFLE):
        """
        Train data set: for each epoch iteration, create training errors, prime training data.
        Randomize training set as necessary. Retrieve feature-label pair, assign feature values
        to input layer. Aggregate metrics per epoch (RMSE, optional accuracy).
        verbosity: 0 = silent, 1 = epoch summaries, 2 = epoch summaries + few sample previews.
        Returns:
            history: dict with keys 'rmse' and (when applicable) 'accuracy'.
        """

        if data_set.number_of_samples(Set.TRAIN) == 0:
            raise EmptySetException

        history: Dict[str, List[float]] = {'rmse': []}
        track_accuracy = None  # set to True/False after first batch if we detect one-hot labels

        for epoch in range(epochs):
            rmse_object = self._error_model()
            correct = 0
            total = 0

            data_set.prime_data(Set.TRAIN, order)

            sample_preview: List[Tuple[List[float], List[float], List[float]]] = []
            # ------------------- iterate over training samples -------------------
            while not data_set.pool_is_empty(Set.TRAIN):
                features, labels = data_set.get_one_item(Set.TRAIN)

                # push inputs
                for neurode, feature in zip(self._list.input_nodes, features):
                    neurode.set_input(input_value=feature)

                predicted_values = [neurode.value for neurode in self._list.output_nodes]
                expected_values = labels

                # accumulate loss
                rmse_object += (predicted_values, expected_values)

                # set expected for backprop and trigger update downstream
                for neurode, expected in zip(self._list.output_nodes, expected_values):
                    neurode.set_expected(expected)

                # detect/track accuracy if one-hot classification
                if track_accuracy is None:
                    acc_flag = self._maybe_accuracy(predicted_values, expected_values)
                    if acc_flag is not None:
                        track_accuracy = True
                        history.setdefault('accuracy', [])
                        correct += 1 if acc_flag else 0
                        total += 1
                    else:
                        track_accuracy = False
                elif track_accuracy:
                    acc_flag = self._maybe_accuracy(predicted_values, expected_values)
                    if acc_flag is not None:
                        correct += 1 if acc_flag else 0
                        total += 1

                # keep 2 examples for preview when verbosity > 1
                if verbosity > 1 and len(sample_preview) < 2:
                    sample_preview.append((features, expected_values, predicted_values))
            # ------------------- end epoch loop -------------------

            epoch_rmse = rmse_object.error
            history['rmse'].append(epoch_rmse)

            if track_accuracy:
                epoch_acc = (correct / max(1, total))
                history['accuracy'].append(epoch_acc)

            # compact, once-per-epoch logging
            if verbosity > 0:
                if track_accuracy:
                    print(f"[epoch {epoch+1}/{epochs}] RMSE={epoch_rmse:.6f}  ACC={epoch_acc:.3f}")
                else:
                    print(f"[epoch {epoch+1}/{epochs}] RMSE={epoch_rmse:.6f}")

                if verbosity > 1 and sample_preview:
                    for i, (f, y, yhat) in enumerate(sample_preview, start=1):
                        print(f"  ex{i}  X:{f}  y:{y}  ŷ:{[round(v,6) for v in yhat]}")

        # final line mirrors your style, but now reflects the last epoch summary
        print(f"Final RMSE value report: {history['rmse'][-1]:.12f}")
        return history

    def test(self, data_set: NNData, order=Order.STATIC, show_examples: int = 3):
        """
        Utilize testing set to track testing progress.
        Records RMSE and (if one-hot) accuracy. Prints a brief summary and a few examples.
        Returns:
            metrics: dict with 'rmse' and optional 'accuracy'.
        """
        if data_set.number_of_samples(Set.TEST) == 0:
            raise EmptySetException

        rmse_object = self._error_model()
        correct = 0
        total = 0
        examples: List[Tuple[List[float], List[float], List[float]]] = []

        data_set.prime_data(Set.TEST, order)

        while not data_set.pool_is_empty(Set.TEST):
            features, labels = data_set.get_one_item(Set.TEST)

            for neurode, feature in zip(self._list.input_nodes, features):
                neurode.set_input(feature)

            predicted_values = [neurode.value for neurode in self._list.output_nodes]
            expected_values = labels

            rmse_object += (predicted_values, expected_values)

            # accuracy if one-hot
            acc_flag = self._maybe_accuracy(predicted_values, expected_values)
            if acc_flag is not None:
                total += 1
                correct += 1 if acc_flag else 0

            if len(examples) < show_examples:
                examples.append((features, expected_values, predicted_values))

        # summary
        metrics = {'rmse': rmse_object.error}
        if total > 0:
            metrics['accuracy'] = correct / total

        # print compact testing summary
        print(f"(test) Final RMSE: {metrics['rmse']}")
        if 'accuracy' in metrics:
            print(f"(test) Accuracy: {metrics['accuracy']:.3f}")

        for i, (f, y, yhat) in enumerate(examples, start=1):
            print(f"(test ex{i}) X:{f}")
            print(f"(test ex{i}) y:{y}")
            print(f"(test ex{i}) ŷ:{[round(v,6) for v in yhat]}")

        return metrics
