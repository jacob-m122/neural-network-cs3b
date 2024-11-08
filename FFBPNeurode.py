from FFNeurode import FFNeurode
from BPNeurode import BPNeurode

class FFBPNeurode(FFNeurode, BPNeurode):
    """Combine feedforward and backpropagation functions into one class."""

    def __init__(self):
        """Initialize neurode with superclass."""
        super().__init__()
