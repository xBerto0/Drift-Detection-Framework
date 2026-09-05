"""Detector di concept drift: monitora lo stream degli errori del modello."""

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


def errore_binario(y_pred, y_true) -> float:
    """Errore 0/1 per la classificazione: 1 se il modello ha sbagliato."""
    return 0.0 if y_pred == y_true else 1.0


def errore_assoluto(y_pred, y_true) -> float:
    """Errore assoluto per la regressione: |y_true - y_pred|."""
    return abs(float(y_true) - float(y_pred))


class ConceptDriftDetector(BaseDriftDetector):
    """Monitora lo stream degli errori del modello con una strategia.

    Riceve la coppia (y_pred, y_true), calcola internamente l'errore e lo
    passa alla strategia sottostante.

    Il modo in cui l'errore viene calcolato dipende dal tipo di problema, ed e'
    per questo configurabile tramite il parametro `error_fn`:

    - CLASSIFICAZIONE: `errore_binario` (default) produce uno stream di 0/1.
      E' il formato richiesto da DDM, ed e' gestito nativamente anche da ADWIN.
    - REGRESSIONE: `errore_assoluto` produce uno stream continuo e non
      limitato. DDM non e' applicabile in questo caso; la strategia adatta e'
      PageHinkleyStrategy.

    Il default resta l'errore binario per non modificare il comportamento degli
    esperimenti di classificazione gia' esistenti.
    """

    def __init__(self, strategy_cls, error_fn=errore_binario, **strategy_kwargs):
        super().__init__(detector_name="ConceptDriftDetector",
                         drift_type="concept")
        self.strategy = strategy_cls(**strategy_kwargs)
        self.error_fn = error_fn

    def update(self, y_pred, y_true) -> None:
        # Il calcolo dell'errore e' delegato alla funzione configurata, cosi'
        # lo stesso detector serve sia la classificazione sia la regressione.
        errore = self.error_fn(y_pred, y_true)
        self.strategy.update(errore)

    def imposta_riferimento(self, y_pred_riferimento, y_true_riferimento) -> None:
        """Fissa la baseline sugli errori commessi sui dati di training.

        Riceve le coppie e non uno stream gia' pronto perche' il calcolo
        dell'errore e' responsabilita' di questo detector, non del chiamante.
        """
        errori = [
            self.error_fn(pred, vero)
            for pred, vero in zip(y_pred_riferimento, y_true_riferimento)
        ]
        self.strategy.imposta_riferimento(errori)

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
