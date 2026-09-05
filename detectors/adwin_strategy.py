"""Strategia di drift detection basata su ADWIN (ADaptive WINdowing).

Wrapping di river.drift.ADWIN, implementazione canonica del paper
Bifet & Gavalda 2006.
"""

from river.drift import ADWIN

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


class ADWINStrategy(BaseDriftDetector):
    """Monitora uno stream di numeri usando finestre adattive."""

    def __init__(self, delta: float = 0.002):
        super().__init__(detector_name="ADWIN", drift_type="feature")
        self.delta = delta
        self.adwin = ADWIN(delta=delta)

    def update(self, value: float) -> None:
        # Passiamo il valore direttamente all'algoritmo di river.
        self.adwin.update(value)

    def detect(self) -> DriftResult:
        # river espone drift_detected come bool: True solo nell'istante
        # in cui ADWIN ha appena tagliato la finestra.
        return DriftResult(
            detector_name=self.detector_name,
            drift_detected=bool(self.adwin.drift_detected),
            drift_type=self.drift_type,
            score=float(self.adwin.estimation),
            metadata={"delta": self.delta},
        )

    def reset(self) -> None:
        # Si "resetta" creando una nuova istanza di ADWIN.
        self.adwin = ADWIN(delta=self.delta)
