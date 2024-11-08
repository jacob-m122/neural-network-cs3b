"""Test Output Neurode"""
from BPNeurode import BPNeurode

output_node = BPNeurode()

output_node._value = 0.6

expected_output = 1.0

output_node._calculate_delta(expected_value=expected_output)

print(f"Output Neurode value: {output_node.value}")
print(f"Expected Output: {expected_output}")
print(f"Calculated Delta of Output Neurode: {output_node.delta}")