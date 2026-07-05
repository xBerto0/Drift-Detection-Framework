"""Detector di feature drift basato su una strategia indipendente per feature."""

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


class FeatureDriftDetector(BaseDriftDetector):
    """Monitora N feature applicando una strategia separata a ciascuna."""

    def __init__(
        self,
        strategy_cls,
        n_features: int,    # quante feature monitora il detector 
        k: int = 1,         # quante feature devono essere in drift per dichiarare il drift globale
        feature_names=None,
        **strategy_kwargs,
    ):
        super().__init__(detector_name="FeatureDriftDetector", drift_type="feature")
        self.n_features = n_features 
        self.k = k 
        self.feature_names = feature_names 
        # Un'istanza di strategia indipendente per ogni feature.
        self.strategies = [strategy_cls(**strategy_kwargs) for _ in range(n_features)]

    def update(self, x) -> None:
        # Smista ogni valore alla strategia della feature corrispondente.
        for i in range(self.n_features):
            self.strategies[i].update(x[i])

    def detect(self) -> DriftResult:
        # Chiede ad ogni strategia se vede drift sulla sua feature.
        drifted_indices = []
        for i, strategy in enumerate(self.strategies):
            result = strategy.detect()
            if result.drift_detected:
                drifted_indices.append(i)

        drift = len(drifted_indices) >= self.k

        # Se l'utente ha passato i nomi delle feature, usa quelli; altrimenti gli indici.
        if self.feature_names is not None:
            drifted = [self.feature_names[i] for i in drifted_indices]
        else:
            drifted = drifted_indices

        return DriftResult(
            detector_name=self.detector_name,
            drift_detected=drift,
            drift_type=self.drift_type,
            metadata={
                "drifted_features": drifted,
                "n_drifted": len(drifted_indices),
                "k_threshold": self.k,
            },
        )

    def reset(self) -> None:
        for strategy in self.strategies:
            strategy.reset()
