import pytest
import NNData
import RMSE


try:
    import FFBPNetwork
except ImportError:
    pytest.fail("Cannot import RMSE. Is RMSE.py present?")

@pytest.fixture()
def NNDATA_obj():
    features = [[.1, .1], [.2, .2], [.3, .3], [.4, .4], [.5, .5], [.6, .6],
                [.7, .7], [.8, .8]]
    labels = [[1], [2], [3], [4], [5], [6], [7], [8]]
    my_data = NNData.NNData()
    my_data.load_data(features=features, labels=labels)
    return my_data

def test_RMSE():
    try:
        FFBPNetwork.FFBPNetwork
    except TypeError:
        pytest.fail("Is the RMSE class defined?")


def test_custom_exception():
    try:
        try:
            raise FFBPNetwork.FFBPNetwork.EmptySetException
        except FFBPNetwork.FFBPNetwork.EmptySetException:
            pass
    except:
        pytest.fail("Could not raise EmptySetException")

def test_test_method(NNDATA_obj):
    try:
        network = FFBPNetwork.FFBPNetwork(2, 1, RMSE.Euclidean)
    except:
        pytest.fail("Could not create FFBPNetwork object.")
    try:
        network.test
    except:
        pytest.fail("Cannot find test() method.")
    try:
        network.test(NNDATA_obj)
    except:
        pytest.fail("Cannot run test() method using default parameters")
    try:
        network.test(NNDATA_obj, NNData.Order.STATIC)
    except:
        pytest.fail("Cannot run test() method passing optional parameters")

def test_train_method(NNDATA_obj):
    try:
        network = FFBPNetwork.FFBPNetwork(2, 1, RMSE.Euclidean)
    except:
        pytest.fail("Could not create FFBPNetwork object.")
    try:
        network.train
    except:
        pytest.fail("Cannot find train() method.")
    try:
        network.train(NNDATA_obj)
    except:
        pytest.fail("Cannot run train() method using default parameters")
    try:
        network.train(NNDATA_obj, 2)
    except:
        pytest.fail("Cannot run train() method passing one optional parameter")
    try:
        network.train(NNDATA_obj, 2, 2)
    except:
        pytest.fail("Cannot run train() method passing two optional "
                    "parameters")
    try:
        network.train(NNDATA_obj, 2, 2, NNData.Order.SHUFFLE)
    except:
        pytest.fail("Cannot run train() method passing two optional "
                    "parameters")


def test_add_hidden_layer():
    try:
        network = FFBPNetwork.FFBPNetwork(2, 1, RMSE.Euclidean)
    except:
        pytest.fail("Could not create FFBPNetwork object.")
    try:
        network.add_hidden_layer
    except:
        pytest.fail("Cannot find add_hidden_layer() method.")
    try:
        network.add_hidden_layer(3, 1)
    except IndexError:
        pass
    except:
        pytest.fail("Trying to add a hidden layer after the tail raised some"
                    "error other than IndexError.")
    try:
        network.add_hidden_layer(3, 2)
    except IndexError:
        pass
    except:
        pytest.fail("Trying to add a hidden layer after the tail raised some"
                    "error other than IndexError.")
    try:
        network.add_hidden_layer(3, 1)
    except IndexError:
        pass
    except:
        pytest.fail("Is add_hidden_layer allowing a layer to be added after"
                    "the tail?")
    try:
        network.add_hidden_layer(3)
    except:
        pytest.fail("Could not add first hidden layer using default"
                    "parameter.")
    try:
        network.add_hidden_layer(3, 0)
    except:
        pytest.fail("Could not add second hidden layer before first.")
    try:
        network.add_hidden_layer(3, 2)
    except:
        pytest.fail("Could not add third hidden layer after first two.")

