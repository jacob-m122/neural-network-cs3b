from LayerList import LayerList
from Neurode import Neurode

def layer_list_test():

    layer_list = LayerList(inputs=2, outputs=1, neurode_type=Neurode)

    print("Layerlist created.")

    print(f"Input neurodes: {len(layer_list.input_nodes)}")

    print(f"Output neurodes: {len(layer_list.output_nodes)}")

if __name__ == "__main__":
    layer_list_test()
