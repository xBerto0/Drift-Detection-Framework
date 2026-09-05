"""Strategia di drift detection basata sul test di Page-Hinkley.

Wrapping di river.drift.PageHinkley. Il test e' una variante del CUSUM
(somma cumulativa) introdotto da Page (1954) e ripreso da Hinkley (1971);
nella letteratura sul concept drift viene descritto in Gama et al. (2014).

Perche' serve al framework
--------------------------
KS lavora su distribuzioni, ADWIN sulla media di uno stream limitato, DDM su
uno stream binario di errori. Manca un rilevatore che lavori su uno stream
CONTINUO e NON limitato, ed e' esattamente il caso dell'errore di un modello
di REGRESSIONE: l'errore assoluto |y_true - y_pred| e' un numero reale
positivo senza limite superiore, su cui DDM non e' applicabile.

Page-Hinkley copre quel caso, e in piu' rappresenta una terza famiglia
teorica (analisi sequenziale / CUSUM) accanto ai test statistici a due
campioni e al windowing adattivo, completando la tassonomia dei metodi
descritta nel capitolo di stato dell'arte.

L'idea dell'algoritmo
---------------------
Si mantiene la media corrente dello stream e si accumula, passo dopo passo,
la somma degli scostamenti dalla media, ridotti di una tolleranza delta:

    m_T = somma_{t=1..T} ( x_t - media_t - delta )

Si tiene traccia del minimo storico di questa somma, m_min. Il test segnala
un cambiamento quando la distanza fra il valore corrente e il minimo storico
supera una soglia lambda:

    PH_T = m_T - m_min > threshold

Il parametro delta e' la magnitudine di cambiamento che accettiamo come
rumore e non vogliamo segnalare; threshold governa il compromesso fra
prontezza e falsi allarmi (soglia bassa = rilevazioni rapide ma piu' falsi
allarmi, soglia alta = il contrario).

Il parametro mode
-----------------
river permette di monitorare aumenti ('up'), diminuzioni ('down') o entrambi
('both'). Il default della libreria e' 'both' e viene mantenuto qui per
coerenza con la scelta, adottata in tutto il framework, di non alterare i
default delle implementazioni di riferimento.

Nota operativa: quando si monitora l'errore di un modello interessa in genere
solo la sua CRESCITA, quindi negli esperimenti sul concept drift ha senso
passare esplicitamente mode='up'. Un errore che cala non e' un problema da
segnalare.
"""

from river.drift import PageHinkley

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


class PageHinkleyStrategy(BaseDriftDetector):
    """Monitora uno stream continuo con il test sequenziale di Page-Hinkley."""

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        mode: str = "both",
    ):
        super().__init__(detector_name="PageHinkley", drift_type="concept")
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mode = mode
        self.ph = self._nuova_istanza()

    def _nuova_istanza(self) -> PageHinkley:
        return PageHinkley(
            min_instances=self.min_instances,
            delta=self.delta,
            threshold=self.threshold,
            alpha=self.alpha,
            mode=self.mode,
        )

    def update(self, value: float) -> None:
        # Lo stream puo' essere qualunque sequenza di numeri reali: errore di
        # regressione, valore di una feature continua, punteggio di un modello.
        self.ph.update(value)

    def detect(self) -> DriftResult:
        return DriftResult(
            detector_name=self.detector_name,
            drift_detected=bool(self.ph.drift_detected),
            drift_type=self.drift_type,
            score=None,  # la statistica interna PH_T non e' esposta da river
            metadata={
                "min_instances": self.min_instances,
                "delta": self.delta,
                "threshold": self.threshold,
                "alpha": self.alpha,
                "mode": self.mode,
            },
        )

    def reset(self) -> None:
        self.ph = self._nuova_istanza()
