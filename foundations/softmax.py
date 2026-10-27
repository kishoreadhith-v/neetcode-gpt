import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_max = max(z)
        z = z - z_max
        z_sum = np.sum(np.exp(z))
        softmax = np.exp(z) / z_sum
        return np.round(softmax, 4)
