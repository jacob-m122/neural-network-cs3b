# CrossEntropy.py
import math

class CrossEntropy:
    """
    Cross-Entropy error model: supports += (y_hat, y) and .error
    Assumes y_hat are probabilities (apply softmax externally or ensures outputs are normalized (0,1))
    """
    def __init__(self, eps: float = 1e-12):
        self._sum = 0.0
        self._n = 0
        self._eps = eps

    def __iadd__(self, pair):
        y_hat, y = pair

        for ph, t in zip(y_hat, y):
            if t > 0.5:
                self._sum += -math.log(max(ph, self._eps))
                self._n += 1
        return self
    
    @property
    def error(self) -> float:
        if self._n == 0: return float("nan")
        return self._sum / self._n