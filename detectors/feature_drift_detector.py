"""Detector di feature drift basato su una strategia indipendente per feature.

Il problema del test multiplo
-----------------------------
Il detector esegue un test statistico INDIPENDENTE su ciascuna feature e poi
dichiara drift globale se almeno `k` feature risultano drittate. Con `k=1`,
questo significa che basta un solo test positivo su n.

Se ogni test ha un livello di significativita' alpha, la probabilita' che
almeno uno degli n test produca un falso positivo non e' alpha, ma:

    P(almeno un falso positivo) = 1 - (1 - alpha)^n

Con alpha = 0.05 e n = 8 feature si ottiene il 34%. Con n = 10 si arriva al
40%. Il detector, con i parametri considerati "standard", sbaglierebbe quindi
in circa un passo su tre anche su dati perfettamente stazionari — e in effetti
e' esattamente cio' che si osserva sul controllo negativo dei dataset
Friedman, dove le feature sono uniformi per costruzione e non contengono
alcun drift.

Non e' un difetto del test di Kolmogorov-Smirnov: e' un difetto della REGOLA
DI AGGREGAZIONE, che ignora quanti test sono stati eseguiti. La correzione
per test multipli e' la risposta classica al problema.

Due correzioni sono implementate:

- BONFERRONI: la piu' conservativa. Si rifiuta l'ipotesi nulla solo se
  p_i < alpha / n. Garantisce che la probabilita' di ALMENO un falso positivo
  resti sotto alpha (controllo del family-wise error rate), al prezzo di una
  perdita di potenza statistica.

- BENJAMINI-HOCHBERG: controlla invece la frazione ATTESA di falsi positivi
  fra i test dichiarati positivi (false discovery rate). Ordinati i p-value in
  modo crescente, si cerca il piu' grande k tale che p_(k) <= (k/n) * alpha e
  si rifiutano le prime k ipotesi. Meno conservativa di Bonferroni, e in
  genere preferibile quando ci si aspetta che piu' feature possano drittare
  insieme.

La correzione richiede che la strategia esponga un p-value in `score`. E' il
caso di `KSStrategy`; ADWIN, DDM e Page-Hinkley non sono test di ipotesi e non
producono p-value, quindi per loro il parametro viene ignorato e si usano i
verdetti nativi.
"""

from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


CORREZIONI_VALIDE = (None, "bonferroni", "benjamini-hochberg")


class FeatureDriftDetector(BaseDriftDetector):
    """Monitora N feature applicando una strategia separata a ciascuna."""

    def __init__(
        self,
        strategy_cls,
        n_features: int,    # quante feature monitora il detector
        k: int = 1,         # quante feature devono essere in drift per dichiarare il drift globale
        feature_names=None,
        correzione=None,        # None | "bonferroni" | "benjamini-hochberg"
        alpha_correzione=None,  # richiesto se correzione e' attiva
        **strategy_kwargs,
    ):
        super().__init__(detector_name="FeatureDriftDetector", drift_type="feature")

        if correzione not in CORREZIONI_VALIDE:
            raise ValueError(
                f"Correzione '{correzione}' non supportata. "
                f"Valori ammessi: {CORREZIONI_VALIDE}"
            )
        if correzione is not None and alpha_correzione is None:
            raise ValueError(
                "Con una correzione per test multipli va indicato anche "
                "alpha_correzione, perche' la soglia va ricalcolata a partire "
                "da esso. Il nome e' distinto da 'alpha' per non entrare in "
                "conflitto con l'omonimo parametro delle strategie, che viene "
                "inoltrato tramite **strategy_kwargs."
            )

        self.n_features = n_features
        self.k = k
        self.feature_names = feature_names
        self.correzione = correzione
        self.alpha_correzione = alpha_correzione
        # Un'istanza di strategia indipendente per ogni feature.
        self.strategies = [strategy_cls(**strategy_kwargs) for _ in range(n_features)]

    def update(self, x) -> None:
        # Smista ogni valore alla strategia della feature corrispondente.
        for i in range(self.n_features):
            self.strategies[i].update(x[i])

    def imposta_riferimento(self, X) -> None:
        """Passa a ogni strategia la colonna di riferimento che le compete.

        `X` e' una matrice campioni x feature: si scompone per colonna e ogni
        colonna va alla strategia che monitora quella feature.
        """
        for i in range(self.n_features):
            colonna = [riga[i] for riga in X]
            self.strategies[i].imposta_riferimento(colonna)

    def detect(self) -> DriftResult:
        # Chiede ad ogni strategia il proprio verdetto sulla sua feature.
        risultati = [strategia.detect() for strategia in self.strategies]

        if self.correzione is None:
            drifted_indices = [i for i, r in enumerate(risultati) if r.drift_detected]
        else:
            drifted_indices = self._verdetti_corretti(risultati)

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
                "correzione": self.correzione,
            },
        )

    def _verdetti_corretti(self, risultati):
        """Riapplica la soglia ai p-value tenendo conto del numero di test.

        Se anche una sola strategia non espone un p-value (perche' non e' un
        test di ipotesi, o perche' e' ancora in warming up), la correzione non
        e' applicabile e si ricade sui verdetti nativi delle strategie.
        """
        p_values = [r.score for r in risultati]
        if any(p is None for p in p_values):
            return [i for i, r in enumerate(risultati) if r.drift_detected]

        n = len(p_values)

        if self.correzione == "bonferroni":
            soglia = self.alpha_correzione / n
            return [i for i, p in enumerate(p_values) if p < soglia]

        # Benjamini-Hochberg: si ordinano i p-value in modo crescente e si
        # cerca il piu' grande rango k per cui p_(k) <= (k/n) * alpha.
        ordinati = sorted(enumerate(p_values), key=lambda coppia: coppia[1])
        k_max = 0
        for rango, (_, p) in enumerate(ordinati, start=1):
            if p <= (rango / n) * self.alpha_correzione:
                k_max = rango
        return [indice for indice, _ in ordinati[:k_max]]

    def reset(self) -> None:
        for strategy in self.strategies:
            strategy.reset()
