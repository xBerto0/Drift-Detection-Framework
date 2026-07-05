"""Interfaccia base per i drift detector."""

from abc import ABC, abstractmethod

from results.drift_result import DriftResult


class BaseDriftDetector(ABC):
    """Classe astratta che definisce il contratto comune a tutti i detector."""

    def __init__(self, detector_name: str, drift_type: str) -> None:
        self.detector_name = detector_name
        self.drift_type = drift_type

    @abstractmethod
    def update(self, value: float) -> None:
        """Aggiorna lo stato interno del detector con una nuova osservazione."""

    @abstractmethod
    def detect(self) -> DriftResult:
        """Valuta lo stato corrente e restituisce un DriftResult."""

    @abstractmethod
    def reset(self) -> None:
        """Riporta il detector allo stato iniziale."""
