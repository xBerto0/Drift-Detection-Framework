"""Generatore di dati sintetici per testare i drift detector."""

import numpy as np


def bernoulli_with_abrupt_drift(n_samples, p_before, p_after, drift_point, seed=None):
    """Genera un flusso Bernoulli con cambio brusco di p al passo drift_point.

    Replica lo scenario della Figura 2 del paper Bifet & Gavalda 2006:
    un cambiamento improvviso del parametro mu della distribuzione.
    """
    rng = np.random.default_rng(seed)
    before = rng.binomial(n=1, p=p_before, size=drift_point)
    after = rng.binomial(n=1, p=p_after, size=n_samples - drift_point)
    return np.concatenate([before, after]).astype(float)
