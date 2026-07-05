"""Detector di prediction drift: monitora lo stream delle predizioni del modello."""

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


class PredictionDriftDetector(BaseDriftDetector):
    """Monitora un singolo stream (le predizioni del modello) con una strategia."""

    def __init__(self, strategy_cls, **strategy_kwargs):
        super().__init__(detector_name="PredictionDriftDetector",
                         drift_type="prediction")
        self.strategy = strategy_cls(**strategy_kwargs)

    def update(self, y_pred) -> None:
        # Delega alla strategia sottostante.
        self.strategy.update(y_pred)

    def detect(self) -> DriftResult:
        # Legge il verdetto della strategia e riclassifica il tipo di drift.
        result = self.strategy.detect()
        return DriftResult(
            detector_name=self.detector_name,
            drift_detected=result.drift_detected,
            drift_type=self.drift_type,
            score=result.score,
            metadata=result.metadata,
        )

    def reset(self) -> None:
        self.strategy.reset()
