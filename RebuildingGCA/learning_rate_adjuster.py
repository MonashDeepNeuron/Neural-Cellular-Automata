from typing import List
import torch.nn.functional as f
import torch as t
from math import exp
import numpy as np


class lradj:
    """
    Learning Rate Adjuster

    Methods:
      - get_adjusted_learning_rate static
      calculates the new learning rate for a given batch of epoch losses

    Attributes:
    - SCALING_FACTOR
    The current scaling_factor highly depends on the source image and the range of loss encountered when training. Training the cat image started off with high losses around 0.5 and converged to losses of 0.02 (very good results visually) so this scaling factor is derived from that.
    """

    # SCALING_FACTOR: float = 1e-3
    LOSS_UB: float = 1e-3
    LOSS_LB: float = 1e-5

    TURBULENCE_LIMIT: float = 1e-4

    @staticmethod
    def aggegate_sigmoid(aggregate_loss: float) -> float:
        exponent_factor: float = -50 * (aggregate_loss - 0.2)
        denominator: float = 1 + exp(exponent_factor)
        unscaled_lr: float = 1 / denominator
        normalise_lr: float = lradj.map_to_log(unscaled_lr)
        return normalise_lr

    @staticmethod
    def map_to_log(input: float) -> float:
        log_value = np.log10(lradj.LOSS_LB) + input * (
            np.log10(lradj.LOSS_UB) - np.log10(lradj.LOSS_LB)
        )
        return 10**log_value

    @staticmethod
    def turbulence_sigmoid(median_delta_loss: float) -> float:
        return 0.0

    @staticmethod
    def get_adjusted_learning_rate(losses: List[float]) -> float:
        n: int = len(losses)
        ## Calculating aggregate base learning rate

        # Filtering out outliers
        loss_sigma: float = t.std(losses)
        low_range: float = t.median(losses) - loss_sigma
        high_range: float = t.median(losses) + loss_sigma
        filtered_losses: List[float] = filter(
            lambda loss: (low_range <= loss) and (loss <= high_range), losses
        )
        aggregate_loss: float = sum(filtered_losses)
        base_learning_rate: float = lradj.aggegate_sigmoid(aggregate_loss)

        ## Calculating turbulence adjustment bias
        delta: List[float] = [None for i in range(n - 1)]
        for i in range(0, len(losses) - 1):
            delta[i] = abs(losses[i + 1] - losses[i])
        delta_median: float = t.median(delta)
        adjustment_learning_rate: float = lradj.turbulence_sigmoid(delta_median)

        new_learning_rate: float = base_learning_rate + adjustment_learning_rate

        return new_learning_rate


if __name__ == "__main__":
    test_losses: List[float] = [0.5, 0.2, 0.05, 0.02]
    adjustment_losses: List[float] = map(
        lambda loss: lradj.aggegate_sigmoid(loss), test_losses
    )
    print(list(adjustment_losses))
