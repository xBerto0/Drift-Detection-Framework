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

    def imposta_riferimento(self, valori) -> None:
        """Riceve i dati che descrivono il comportamento "normale" atteso.

        Serve nell'uso a lotti, dove si dispone di due insiemi distinti: i dati
        su cui il modello monitorato e' stato addestrato e i dati nuovi in
        arrivo. Il confronto ha senso solo se il riferimento e' il primo dei due.

        L'implementazione predefinita fa semplicemente scorrere i valori come
        normali update: e' il comportamento corretto per le strategie
        sequenziali (ADWIN, DDM, Page-Hinkley), che non hanno una nozione
        esplicita di riferimento ma si limitano ad accumulare statistica sul
        regime iniziale.

        Le strategie che invece confrontano due campioni, come il KS,
        sovrascrivono questo metodo per riempire la propria finestra di
        riferimento senza sporcare quella corrente.
        """
        for valore in valori:
            self.update(valore)
