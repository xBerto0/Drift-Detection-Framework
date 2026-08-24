"""Strategia di drift detection basata sul test di Kolmogorov-Smirnov."""

from collections import deque

from scipy.stats import ks_2samp

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


class KSStrategy(BaseDriftDetector):
    """Confronta due finestre con il test KS a due campioni, ovvero la massima distanza tra le due funzioni di distribuzione."""

    def __init__(self, window_size: int = 100, alpha: float = 0.05):
        super().__init__(detector_name="KS", drift_type="feature")
        self.window_size = window_size
        self.alpha = alpha
        self.reference = deque(maxlen=window_size)
        self.current = deque(maxlen=window_size)

    def update(self, value: float) -> None:
        # Prima si riempie la reference, poi si popola la current scorrevole.
        if len(self.reference) < self.window_size:
            self.reference.append(value)
        else:
            self.current.append(value)

    def detect(self) -> DriftResult:
        # Servono entrambe le finestre piene per eseguire il test.
        if len(self.current) < self.window_size:
            return DriftResult(
                detector_name=self.detector_name,
                drift_detected=False,
                drift_type=self.drift_type,
                metadata={"status": "warming_up"},
            )

        statistic, p_value = ks_2samp(list(self.reference), list(self.current))
        # bool() esplicito: scipy restituisce un numpy.bool_, che poi
        # comparirebbe come 'np.True_' nei metadata e nei file JSON.
        drift = bool(p_value < self.alpha)

        return DriftResult(
            detector_name=self.detector_name,
            drift_detected=drift,
            drift_type=self.drift_type,
            score=float(p_value),
            metadata={"statistic": float(statistic), "alpha": self.alpha},
        )

    def reset(self) -> None:
        self.reference.clear()
        self.current.clear()

# confronta le funzioni di distribuzioni cumulativa empiriche dei due campioni, restituendo la massima distanza tra le due funzioni. 
# il p-value viene calcolato dal confronto delle due windows e deve scendere sotto alpha affinchè scatti il drift. 