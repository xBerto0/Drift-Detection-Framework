"""Test della correzione per test multipli nel FeatureDriftDetector.

Con n test indipendenti a livello alpha, la probabilita' che almeno uno
produca un falso positivo e' 1 - (1 - alpha)^n, non alpha. Con 10 feature e
alpha = 0.05 si arriva al 40%: e' il difetto che questi test verificano
essere corretto.

Lancio dalla radice del progetto:
    pytest tests/ -v
"""

import numpy as np
import pytest

from detectors.adwin_strategy import ADWINStrategy
from detectors.feature_drift_detector import FeatureDriftDetector
from detectors.ks_strategy import KSStrategy


class StrategiaFinta:
    """Strategia di comodo che restituisce un p-value deciso dal test.

    Serve a verificare la logica di correzione in isolamento, senza dipendere
    dai valori che il test KS produce su dati casuali.
    """

    p_values = []
    contatore = 0

    def __init__(self, **kwargs):
        self.indice = StrategiaFinta.contatore
        StrategiaFinta.contatore += 1
        self.detector_name = "Finta"
        self.drift_type = "feature"

    def update(self, value):
        pass

    def detect(self):
        from results.drift_result import DriftResult
        p = StrategiaFinta.p_values[self.indice]
        return DriftResult(
            detector_name="Finta",
            drift_detected=p < 0.05,     # verdetto nativo, senza correzione
            drift_type="feature",
            score=p,
        )

    def reset(self):
        pass


def costruisci(p_values, correzione, alpha=0.05):
    StrategiaFinta.p_values = p_values
    StrategiaFinta.contatore = 0
    extra = {"correzione": correzione, "alpha_correzione": alpha} if correzione else {}
    return FeatureDriftDetector(
        StrategiaFinta, n_features=len(p_values), k=1, **extra,
    )


# ---------------------------------------------------------------------------
# Validazione dei parametri
# ---------------------------------------------------------------------------

def test_correzione_sconosciuta_rifiutata():
    with pytest.raises(ValueError):
        FeatureDriftDetector(KSStrategy, n_features=3, correzione="holm")


def test_correzione_senza_alpha_rifiutata():
    # Serve una soglia da cui ricalcolare quella corretta.
    with pytest.raises(ValueError):
        FeatureDriftDetector(KSStrategy, n_features=3, correzione="bonferroni")


def test_alpha_correzione_non_confligge_con_alpha_della_strategia():
    # Il parametro del detector ha un nome distinto proprio per poter
    # convivere con l'alpha che va inoltrato alla strategia.
    d = FeatureDriftDetector(
        KSStrategy, n_features=3,
        correzione="bonferroni", alpha_correzione=0.01,
        window_size=50, alpha=0.05,
    )
    assert d.alpha_correzione == 0.01
    assert d.strategies[0].alpha == 0.05


# ---------------------------------------------------------------------------
# Bonferroni
# ---------------------------------------------------------------------------

def test_bonferroni_scarta_un_positivo_marginale():
    # p = 0.04 supera alpha = 0.05 senza correzione, ma non la soglia
    # corretta 0.05/10 = 0.005.
    p = [0.04] + [0.9] * 9
    assert costruisci(p, None).detect().drift_detected is True
    assert costruisci(p, "bonferroni").detect().drift_detected is False


def test_bonferroni_conserva_un_positivo_forte():
    # p = 0.0001 sta sotto anche alla soglia corretta.
    p = [0.0001] + [0.9] * 9
    risultato = costruisci(p, "bonferroni").detect()
    assert risultato.drift_detected is True
    assert risultato.metadata["n_drifted"] == 1


def test_bonferroni_soglia_esatta():
    # Con 10 test e alpha 0.05 la soglia e' 0.005: il confronto e' stretto.
    assert costruisci([0.0049] + [0.9] * 9, "bonferroni").detect().drift_detected is True
    assert costruisci([0.0051] + [0.9] * 9, "bonferroni").detect().drift_detected is False


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_meno_conservativo_di_bonferroni():
    # Quattro p-value bassi insieme: BH ne accetta piu' di Bonferroni, che
    # richiede a ciascuno di superare da solo la soglia alpha/n.
    p = [0.001, 0.008, 0.012, 0.02] + [0.9] * 6
    n_bonf = costruisci(p, "bonferroni").detect().metadata["n_drifted"]
    n_bh = costruisci(p, "benjamini-hochberg").detect().metadata["n_drifted"]
    assert n_bh > n_bonf


def test_benjamini_hochberg_nessun_positivo():
    p = [0.5] * 10
    assert costruisci(p, "benjamini-hochberg").detect().drift_detected is False


# ---------------------------------------------------------------------------
# Ricadute e compatibilita'
# ---------------------------------------------------------------------------

def test_senza_correzione_il_comportamento_e_quello_storico():
    # Il default non deve cambiare nulla per gli esperimenti gia' esistenti.
    p = [0.04] + [0.9] * 9
    risultato = costruisci(p, None).detect()
    assert risultato.drift_detected is True
    assert risultato.metadata["correzione"] is None


def test_strategia_senza_p_value_ricade_sui_verdetti_nativi():
    # ADWIN non e' un test di ipotesi: espone una stima della media in `score`,
    # non un p-value. La correzione non e' applicabile e non deve rompere nulla.
    detector = FeatureDriftDetector(
        ADWINStrategy, n_features=3, k=1,
        correzione="bonferroni", alpha_correzione=0.05, delta=0.002,
    )
    rng = np.random.default_rng(4)
    for _ in range(500):
        detector.update(rng.random(3))
    assert detector.detect() is not None


def test_ks_reale_con_correzione_riduce_i_falsi_positivi():
    # Su 10 feature indipendenti e stazionarie ogni segnalazione e' un falso
    # positivo. La correzione deve ridurne il numero in modo netto.
    rng = np.random.default_rng(9)
    dati = rng.normal(size=(3000, 10))

    def conta(correzione):
        extra = ({"correzione": correzione, "alpha_correzione": 0.05}
                 if correzione else {})
        d = FeatureDriftDetector(KSStrategy, n_features=10, k=1,
                                 **extra, window_size=100, alpha=0.05)
        n = 0
        for riga in dati:
            d.update(riga)
            if d.detect().drift_detected:
                n += 1
        return n

    senza = conta(None)
    con_bonf = conta("bonferroni")
    assert con_bonf < senza
