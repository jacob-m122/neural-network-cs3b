#To test: are features and labels both lists of lists?
XOR_features = [[0, 0], [0, 1], [1,0], [1, 1]]
XOR_labels = [[0], [1], [1], [0]]
my_data = NNData.NNData(features=XOR_features, labels=XOR_labels)
my_data.get_one_item(NNData.Set.TRAIN)
my_data.prime_data()