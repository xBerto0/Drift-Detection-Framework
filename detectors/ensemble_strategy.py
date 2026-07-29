"""Strategia ensemble: aggrega piu' strategie di drift detection sullo stesso stream.

E' una strategia come le altre (estende BaseDriftDetector), quindi puo' essere
usata dentro qualunque detector (FeatureDriftDetector, PredictionDriftDetector,
ConceptDriftDetector) senza modifiche al codice di orchestrazione.

Ogni strategia interna riceve lo stesso flusso di valori. Al momento del
detect() vengono raccolti i verdetti individuali e aggregati secondo una
delle regole supportate:

- OR:       drift globale se almeno una strategia segnala drift
- AND:      drift globale solo se tutte le strategie segnalano drift
- MAJORITY: drift globale se piu' di meta' delle strategie segnala drift
"""

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


VALID_AGGREGATIONS = ("OR", "AND", "MAJORITY")


class EnsembleStrategy(BaseDriftDetector):
    """Aggrega piu' strategie di drift detection sullo stesso stream."""

    def __init__(self, strategy_specs, aggregation="OR"):
        # strategy_specs = [(cls, kwargs), (cls, kwargs), ...]
        super().__init__(detector_name="Ensemble", drift_type="feature")

        aggregation = aggregation.upper()
        if aggregation not in VALID_AGGREGATIONS:
            raise ValueError(
                f"Aggregazione '{aggregation}' non supportata. "
                f"Valori ammessi: {VALID_AGGREGATIONS}"
            )
        self.aggregation = aggregation

        # Istanzia le strategie interne con i rispettivi parametri.
        self.strategies = [cls(**kwargs) for cls, kwargs in strategy_specs]

    def update(self, value) -> None:
        # Ogni strategia interna riceve lo stesso valore.
        for strategy in self.strategies:
            strategy.update(value)

    def detect(self) -> DriftResult:
        # Raccoglie i verdetti di tutte le strategie interne.
        sub_results = [s.detect() for s in self.strategies]
        verdicts = [r.drift_detected for r in sub_results]

        # Applica la regola di aggregazione.
        drift = self._aggregate(verdicts)

        # Costruisce il metadata con i dettagli individuali.
        metadata = {
            "aggregation_rule": self.aggregation,
            "strategy_verdicts": {
                s.detector_name: r.drift_detected
                for s, r in zip(self.strategies, sub_results)
            },
            "strategy_scores": {
                s.detector_name: r.score
                for s, r in zip(self.strategies, sub_results)
            },
        }

        return DriftResult(
            detector_name=self.detector_name,
            drift_detected=drift,
            drift_type=self.drift_type,
            score=None,  # aggregato non ha uno score comparabile
            metadata=metadata,
        )

    def reset(self) -> None:
        for strategy in self.strategies:
            strategy.reset()

    def _aggregate(self, verdicts) -> bool:
        # Applica la regola di aggregazione ai verdetti booleani.
        if self.aggregation == "OR":
            return any(verdicts)
        if self.aggregation == "AND":
            return all(verdicts)
        if self.aggregation == "MAJORITY":
            return sum(verdicts) > len(verdicts) / 2
        # Non dovrebbe mai arrivare qui grazie alla validazione nel costruttore.
        raise ValueError(f"Aggregazione non gestita: {self.aggregation}")
