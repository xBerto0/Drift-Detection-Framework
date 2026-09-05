"""Strategia di drift detection basata su DDM (Drift Detection Method).

Wrapping di river.drift.binary.DDM, implementazione dell'algoritmo proposto da
Gama, Medas, Castillo e Rodrigues (2004).

DDM e' un metodo ERROR-BASED e SUPERVISIONATO: non guarda la distribuzione dei
dati, ma il tasso di errore del modello. Va quindi alimentato con uno stream
BINARIO in cui 1 = il modello ha sbagliato, 0 = il modello ha indovinato. E'
esattamente cio' che produce il ConceptDriftDetector, motivo per cui DDM e'
pensato per essere usato dentro quel detector.

L'idea dell'algoritmo
---------------------
Se il concetto e' stazionario, il tasso di errore di un classificatore che
apprende dovrebbe diminuire o restare stabile al crescere del numero di
esempi. Se invece il tasso di errore ricomincia a salire in modo
significativo, la distribuzione sottostante e' cambiata.

Trattando gli errori come una variabile di Bernoulli, dopo i primi n esempi il
tasso di errore p_i ha deviazione standard s_i = sqrt(p_i * (1 - p_i) / i).
DDM memorizza il minimo storico della somma p_min + s_min e confronta il
valore corrente con due soglie:

    p_i + s_i >= p_min + 2 * s_min   ->  livello di WARNING
    p_i + s_i >= p_min + 3 * s_min   ->  livello di DRIFT

Perche' le due soglie contano per questa tesi
---------------------------------------------
La soglia di warning non e' un dettaglio: e' il meccanismo che rende possibile
una politica di retraining sensata. Quando si entra in zona di warning si
inizia ad accumulare i campioni in un nuovo buffer di training; se poi il
drift viene confermato, il modello viene riaddestrato su quei campioni, che
appartengono gia' al nuovo concetto. Se invece l'allarme rientra senza
conferma, il buffer viene semplicemente scartato.

Nessuna delle strategie implementate finora (KS, ADWIN) espone una nozione di
warning: DDM e' il primo detector del framework a fornirla, e per questo il
verdetto la riporta nei metadata.
"""

from river.drift.binary import DDM

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


class DDMStrategy(BaseDriftDetector):
    """Monitora lo stream binario degli errori di un classificatore."""

    def __init__(
        self,
        warm_start: int = 30,
        warning_threshold: float = 2.0,
        drift_threshold: float = 3.0,
    ):
        super().__init__(detector_name="DDM", drift_type="concept")
        self.warm_start = warm_start
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.ddm = DDM(
            warm_start=warm_start,
            warning_threshold=warning_threshold,
            drift_threshold=drift_threshold,
        )

    def update(self, value: float) -> None:
        # Il vincolo binario riguarda QUESTA implementazione, non il metodo.
        # river calcola s_i = sqrt(p_i(1-p_i)/i), che presuppone un esito a due
        # valori. Gama et al. (2004), nelle conclusioni, prospettano invece
        # l'uso dell'algoritmo con qualunque funzione di perdita, dati valori
        # appropriati di alpha, e riportano risultati preliminari in regressione
        # con l'errore quadratico medio: quella generalizzazione richiederebbe
        # pero' una diversa stima della varianza, che river non implementa.
        if value not in (0, 1, 0.0, 1.0, True, False):
            raise ValueError(
                f"DDMStrategy accetta solo valori binari (0 = corretto, "
                f"1 = errore), ricevuto {value!r}. L'implementazione di river "
                f"usa la derivazione bernoulliana della deviazione standard. "
                f"Per stream continui usare PageHinkleyStrategy o ADWINStrategy."
            )
        self.ddm.update(int(value))

    def detect(self) -> DriftResult:
        # river espone due flag distinti: warning_detected e drift_detected.
        # Il verdetto del framework resta binario (drift si' / drift no), ma il
        # warning viaggia nei metadata perche' e' l'informazione su cui si
        # basano le politiche di retraining.
        return DriftResult(
            detector_name=self.detector_name,
            drift_detected=bool(self.ddm.drift_detected),
            drift_type=self.drift_type,
            score=None,  # DDM non espone una statistica continua confrontabile
            metadata={
                "warning_detected": bool(self.ddm.warning_detected),
                "warm_start": self.warm_start,
                "warning_threshold": self.warning_threshold,
                "drift_threshold": self.drift_threshold,
            },
        )

    def reset(self) -> None:
        # Come per ADWIN, il reset si ottiene ricreando l'istanza.
        self.ddm = DDM(
            warm_start=self.warm_start,
            warning_threshold=self.warning_threshold,
            drift_threshold=self.drift_threshold,
        )
