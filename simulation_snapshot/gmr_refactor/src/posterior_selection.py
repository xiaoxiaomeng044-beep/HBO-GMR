import numpy as np


def posterior_argmin(iterations, values):
    best_position = None
    best_value = float("inf")
    for position, (iteration, value) in enumerate(zip(iterations, values)):
        try:
            iter_val = float(iteration)
            metric_val = float(value)
        except (TypeError, ValueError):
            continue
        if iter_val > 0 and np.isfinite(metric_val) and metric_val < best_value:
            best_value = metric_val
            best_position = position
    return best_position
