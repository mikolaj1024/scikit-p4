import numpy as np


def generate_random_labels(*, nsize=1000, total_labels=100, min_num_labels=0, max_num_labels=3, seed=31415) -> list[list[np.number]]:
    rng = np.random.default_rng(seed=seed)

    return [rng.choice(range(1, total_labels+1),
                           size=rng.integers(min_num_labels, max_num_labels+1),
                           replace=False).tolist() 
            for _ in range(nsize)]