"""Detector di concept drift: monitora lo stream degli errori del modello."""

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


class ConceptDriftDetector(BaseDriftDetector):
    """Monitora lo stream degli errori del modello con una strategia.

    Riceve la coppia (y_pred, y_true) e calcola internamente l'errore
    binario (1 se il modello ha sbagliato, 0 altrimenti), che viene poi
    passato alla strategia sottostante.
    """

    def __init__(self, strategy_cls, **strategy_kwargs):
        super().__init__(detector_name="ConceptDriftDetector",
                         drift_type="concept")
        self.strategy = strategy_cls(**strategy_kwargs)

    def update(self, y_pred, y_true) -> None:
        # Calcolo dell'errore binario.
        error = 0.0 if y_pred == y_true else 1.0
        self.strategy.update(error)

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
