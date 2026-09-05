"""Strategia di drift detection basata sul test di Kolmogorov-Smirnov."""

from collections import deque

from scipy.stats import ks_2samp

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


class KSStrategy(BaseDriftDetector):
    """Confronta due finestre con il test KS a due campioni, ovvero la massima distanza tra le due funzioni di distribuzione."""

    def __init__(self, window_size: int = 100, alpha: float = 0.05,
                 reference_data=None):
        super().__init__(detector_name="KS", drift_type="feature")
        self.window_size = window_size
        self.alpha = alpha
        self.reference = deque(maxlen=window_size)
        self.current = deque(maxlen=window_size)
        # Se il riferimento e' fornito dall'esterno non va piu' riempito
        # dallo stream in arrivo: vedere imposta_riferimento().
        self.riferimento_esterno = False
        if reference_data is not None:
            self.imposta_riferimento(reference_data)

    def imposta_riferimento(self, valori) -> None:
        """Fissa la finestra di riferimento a partire dai dati di training.

        Sovrascrive il comportamento predefinito, che farebbe scorrere i valori
        come normali update: cosi' facendo il riferimento verrebbe riempito
        correttamente, ma i valori successivi finirebbero nella finestra
        corrente, che si troverebbe gia' popolata di dati di riferimento
        all'inizio del monitoraggio.

        Nota metodologica: il riferimento deve rappresentare la distribuzione su
        cui il modello monitorato e' stato addestrato. Riempirlo con i primi
        valori dello stream in arrivo, come avveniva prima, significa
        confrontare "l'inizio della produzione" con "il resto della produzione"
        anziche' "il training" con "la produzione".

        Dei valori forniti si prendono gli ULTIMI `window_size`, cioe' i piu'
        recenti fra quelli di riferimento: sono i piu' vicini nel tempo al
        periodo che si sta per monitorare.
        """
        self.reference.clear()
        self.current.clear()
        for valore in list(valori)[-self.window_size:]:
            self.reference.append(valore)
        self.riferimento_esterno = True

    def update(self, value: float) -> None:
        # Se il riferimento e' stato fissato dall'esterno, ogni nuovo valore
        # appartiene per definizione alla finestra corrente.
        if self.riferimento_esterno or len(self.reference) >= self.window_size:
            self.current.append(value)
        else:
            self.reference.append(value)

    def detect(self) -> DriftResult:
        # Servono entrambe le finestre piene per eseguire il test.
        if len(self.reference) < self.window_size or len(self.current) < self.window_size:
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
        # Il riferimento fissato dall'esterno sopravvive al reset: rappresenta
        # il training del modello, che un reset del detector non cambia.
        self.current.clear()
        if not self.riferimento_esterno:
            self.reference.clear()

# confronta le funzioni di distribuzioni cumulativa empiriche dei due campioni, restituendo la massima distanza tra le due funzioni.
# il p-value viene calcolato dal confronto delle due windows e deve scendere sotto alpha affinche' scatti il drift.
