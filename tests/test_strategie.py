"""Test delle strategie di drift detection.

Verificano il contratto comune definito da `BaseDriftDetector` e il
comportamento atteso di ciascuna strategia su stream costruiti a tavolino.

Sono la verifica empirica dell'affermazione architetturale della tesi: tutte
le strategie sono intercambiabili perche' rispettano lo stesso contratto.

Lancio dalla radice del progetto:
    pytest tests/ -v
"""

import numpy as np
import pytest

from data.synthetic_generator import bernoulli_improvviso, bernoulli_stazionario
from detectors.adwin_strategy import ADWINStrategy
from detectors.concept_drift_detector import (
    ConceptDriftDetector, errore_assoluto, errore_binario,
)
from detectors.ddm_strategy import DDMStrategy
from detectors.ensemble_strategy import EnsembleStrategy
from detectors.ks_strategy import KSStrategy
from detectors.page_hinkley_strategy import PageHinkleyStrategy


# Le strategie che accettano uno stream binario, con parametri di prova.
STRATEGIE_BINARIE = [
    (KSStrategy, {"window_size": 100, "alpha": 0.05}),
    (ADWINStrategy, {"delta": 0.002}),
    (DDMStrategy, {"warm_start": 200}),
    (PageHinkleyStrategy, {"mode": "up"}),
]


def scorri(strategia, valori):
    """Fa scorrere lo stream e restituisce i passi in cui e' segnalato drift."""
    passi = []
    for t, v in enumerate(valori):
        strategia.update(v)
        if strategia.detect().drift_detected:
            passi.append(t)
    return passi


# ---------------------------------------------------------------------------
# Contratto comune
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,parametri", STRATEGIE_BINARIE)
def test_detect_restituisce_sempre_un_risultato(cls, parametri):
    # Anche prima di aver visto dati, detect() non deve restituire None:
    # il chiamante non deve gestire casi speciali.
    strategia = cls(**parametri)
    risultato = strategia.detect()
    assert risultato is not None
    assert risultato.drift_detected is False
    assert risultato.detector_name


@pytest.mark.parametrize("cls,parametri", STRATEGIE_BINARIE)
def test_detect_e_idempotente(cls, parametri):
    # Chiamare detect() due volte senza update() in mezzo deve dare lo stesso
    # verdetto: la lettura dello stato non lo modifica.
    strategia = cls(**parametri)
    for v in bernoulli_stazionario(300, p=0.3, seed=7)[0]:
        strategia.update(v)
    assert strategia.detect().drift_detected == strategia.detect().drift_detected


@pytest.mark.parametrize("cls,parametri", STRATEGIE_BINARIE)
def test_reset_riporta_allo_stato_iniziale(cls, parametri):
    strategia = cls(**parametri)
    valori, _ = bernoulli_improvviso(2000, 0.2, 0.8, 1000, seed=7)
    scorri(strategia, valori)
    strategia.reset()
    assert strategia.detect().drift_detected is False


# ---------------------------------------------------------------------------
# Comportamento atteso
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,parametri", STRATEGIE_BINARIE)
def test_nessun_drift_su_stream_stazionario(cls, parametri):
    # Su uno stream senza drift le segnalazioni devono essere rare.
    # Non si pretende zero: un tasso di falsi positivi non nullo e' atteso e
    # documentato, ma deve restare marginale.
    valori, _ = bernoulli_stazionario(3000, p=0.3, seed=11)
    passi = scorri(cls(**parametri), valori)
    assert len(passi) < 0.05 * len(valori)


@pytest.mark.parametrize("cls,parametri", STRATEGIE_BINARIE)
def test_drift_improvviso_in_aumento_viene_rilevato(cls, parametri):
    # Tutte le strategie devono vedere un aumento netto di p, entro una
    # tolleranza generosa. Si usa un aumento perche' DDM e Page-Hinkley in
    # mode='up' sono unilaterali per costruzione.
    valori, punti = bernoulli_improvviso(3000, 0.2, 0.7, 1500, seed=11)
    passi = scorri(cls(**parametri), valori)
    dopo_drift = [t for t in passi if punti[0] <= t <= punti[0] + 500]
    assert dopo_drift, f"{cls.__name__} non ha rilevato il drift"


def test_ddm_rifiuta_valori_non_binari():
    # DDM e' definito solo su stream binari: un valore continuo deve produrre
    # un errore esplicito, non un risultato silenziosamente privo di senso.
    with pytest.raises(ValueError):
        DDMStrategy().update(3.7)


def test_ddm_espone_il_livello_di_warning():
    # Il warning e' l'informazione su cui si baseranno le politiche di
    # retraining: deve essere presente nei metadata.
    strategia = DDMStrategy(warm_start=200)
    valori, _ = bernoulli_improvviso(3000, 0.1, 0.6, 1500, seed=3)
    visto_warning = False
    for v in valori:
        strategia.update(v)
        if strategia.detect().metadata["warning_detected"]:
            visto_warning = True
            break
    assert visto_warning


def test_page_hinkley_accetta_stream_continui():
    # A differenza di DDM, Page-Hinkley deve funzionare su valori reali non
    # limitati: e' la ragione per cui e' stato aggiunto.
    rng = np.random.default_rng(5)
    valori = np.concatenate([rng.normal(1.0, 0.3, 800), rng.normal(4.0, 0.3, 800)])
    passi = scorri(PageHinkleyStrategy(mode="up"), valori)
    assert any(800 <= t <= 1300 for t in passi)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def test_ensemble_or_e_piu_permissivo_di_and():
    specs = [
        (KSStrategy, {"window_size": 100, "alpha": 0.05}),
        (ADWINStrategy, {"delta": 0.002}),
    ]
    valori, _ = bernoulli_improvviso(3000, 0.2, 0.7, 1500, seed=13)
    passi_or = scorri(EnsembleStrategy(specs, aggregation="OR"), valori)
    passi_and = scorri(EnsembleStrategy(specs, aggregation="AND"), valori)
    assert len(passi_or) >= len(passi_and)


def test_ensemble_majority_con_tre_strategie_non_degenera_in_and():
    # Con due sole strategie MAJORITY equivale ad AND; con tre diventa un
    # voto reale (bastano 2 su 3).
    specs = [
        (KSStrategy, {"window_size": 100, "alpha": 0.05}),
        (ADWINStrategy, {"delta": 0.002}),
        (PageHinkleyStrategy, {"mode": "up"}),
    ]
    valori, _ = bernoulli_improvviso(3000, 0.2, 0.7, 1500, seed=13)
    passi_maj = scorri(EnsembleStrategy(specs, aggregation="MAJORITY"), valori)
    passi_and = scorri(EnsembleStrategy(specs, aggregation="AND"), valori)
    assert len(passi_maj) >= len(passi_and)


def test_ensemble_rifiuta_regola_sconosciuta():
    with pytest.raises(ValueError):
        EnsembleStrategy([(ADWINStrategy, {})], aggregation="XOR")


# ---------------------------------------------------------------------------
# ConceptDriftDetector: classificazione e regressione
# ---------------------------------------------------------------------------

def test_errore_binario_e_assoluto():
    assert errore_binario(1, 1) == 0.0
    assert errore_binario(1, 0) == 1.0
    assert errore_assoluto(10.0, 12.5) == 2.5


def test_concept_drift_su_regressione():
    # La catena completa modello -> errore assoluto -> Page-Hinkley deve
    # rilevare un peggioramento dell'errore di un regressore.
    rng = np.random.default_rng(21)
    detector = ConceptDriftDetector(
        PageHinkleyStrategy, error_fn=errore_assoluto, mode="up",
    )
    passi = []
    for t in range(2000):
        y_true = 10.0
        scarto = rng.normal(0, 0.3) if t < 1000 else rng.normal(2.5, 0.3)
        detector.update(10.0 + scarto, y_true)
        if detector.detect().drift_detected:
            passi.append(t)
    assert any(1000 <= t <= 1300 for t in passi)


def test_concept_drift_su_classificazione_usa_errore_binario():
    # Il default non deve essere cambiato: gli esperimenti di classificazione
    # gia' esistenti devono continuare a comportarsi come prima.
    detector = ConceptDriftDetector(DDMStrategy, warm_start=200)
    for t in range(2000):
        y_true = 1
        y_pred = 1 if t < 1000 else 0   # il modello comincia a sbagliare sempre
        detector.update(y_pred, y_true)
    assert detector.detect().drift_type == "concept"
