from typing import List
import torch.nn.functional as f
import torch as t
from math import exp
import numpy as np
import statistics


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

    TURBULENCE_SCALING_FACTOR: float = 1e2 # for mapping to sigmoid
    TURBULENCE_UB: float = 1e-4
    TURBULENCE_LB: float = -1e-4

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
        exponent_factor: float = 25 * (median_delta_loss * lradj.TURBULENCE_SCALING_FACTOR - 1)
        denominator: float = 1 + exp(exponent_factor if exponent_factor < 200 else 200)
        unscaled_lr: float = (1 / denominator)
        normalise_lr: float = lradj.map_value_to_range(unscaled_lr)
        return normalise_lr
    
    @staticmethod
    def map_value_to_range(input: float) -> float:
        mapped_value = input * (lradj.TURBULENCE_UB - lradj.TURBULENCE_LB) + lradj.TURBULENCE_LB # to map between -1e-4 and 1e-4
        return mapped_value

    @staticmethod
    def get_adjusted_learning_rate(losses: List[float]) -> float:
        n: int = len(losses)
        ## Calculating aggregate base learning rate

        # Filtering out outliers
        loss_sigma: float = statistics.stdev(losses)
        low_range: float = statistics.median(losses) - loss_sigma
        high_range: float = statistics.median(losses) + loss_sigma
        filtered_losses: List[float] = filter(
            lambda loss: (low_range <= loss) and (loss <= high_range), losses
        )
        aggregate_loss: float = sum(filtered_losses)
        base_learning_rate: float = lradj.aggegate_sigmoid(aggregate_loss)

        ## Calculating turbulence adjustment bias
        delta: List[float] = [None for i in range(n - 1)]
        for i in range(0, len(losses) - 1):
            delta[i] = abs(losses[i + 1] - losses[i])
        delta_median: float = statistics.median(delta)
        adjustment_learning_rate: float = lradj.turbulence_sigmoid(delta_median)

        new_learning_rate: float = base_learning_rate + adjustment_learning_rate

        return new_learning_rate


# if __name__ == "__main__":
    # test_losses: List[float] = [0.5, 0.2, 0.05, 0.02]
    # adjustment_losses: List[float] = lradj.get_adjusted_learning_rate(t.Tensor(test_losses))
    # print((adjustment_losses))
#     # list:List[float] = [0.5125302, 0.5117947, 0.44350803, 0.43563068, 0.43197787, 0.3666371, 0.15286264, 0.0984481, 0.12255937, 0.14001535, 0.1265211, 0.1348856, 0.13952297, 0.11741823, 0.11464076, 0.12191339, 0.09644841, 0.12031035, 0.11494549, 0.10283274, 0.1231882, 0.10839787, 0.11618645, 0.11455134, 0.10809456, 0.10307693, 0.12078851, 0.10172762, 0.10440645,0.09962489]
#     # deltas:List[float] = [None for i in range(len(list) - 1)]
#     # for i in range(0, len(list) - 1):
#     #     deltas[i] = abs(list[i + 1] - list[i])
#     # delta_median: float = statistics.median(deltas)
#     # print(delta_median)
#     # print(deltas)