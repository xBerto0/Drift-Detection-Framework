"""Demo interattiva del framework — pensata per essere mostrata dal vivo.

Gira in pochi secondi. Ogni sezione si ferma e aspetta INVIO, cosi' si puo'
parlare sopra senza rincorrere l'output.

    python demo.py
"""

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def titolo(numero, testo):
    print()
    print("=" * 74)
    print(f"  {numero}.  {testo}")
    print("=" * 74)


def pausa():
    print()
    try:
        input("   [INVIO per continuare]")
    except EOFError:
        pass


# =========================================================================
titolo(1, "I parametri vengono dal file .env, non dal codice")
# =========================================================================

import config

print()
print("   Nessun valore e' scritto a mano negli esperimenti. Si cambia il .env")
print("   e tutti gli esperimenti ripartono con la nuova configurazione.")
print()
for nome in ("KS_WINDOW_SIZE", "KS_ALPHA", "ADWIN_DELTA", "DDM_WARM_START",
             "PH_THRESHOLD", "PH_MODE", "FEATURE_K_THRESHOLD"):
    print(f"      {nome:<22} = {getattr(config, nome)}")

pausa()

# =========================================================================
titolo(2, "Cinque algoritmi, la stessa identica interfaccia")
# =========================================================================

from detectors.adwin_strategy import ADWINStrategy
from detectors.ddm_strategy import DDMStrategy
from detectors.ensemble_strategy import EnsembleStrategy
from detectors.ks_strategy import KSStrategy
from detectors.page_hinkley_strategy import PageHinkleyStrategy

print()
print("   Ogni algoritmo espone update() / detect() / reset() e basta.")
print("   Aggiungerne uno non ha mai richiesto di toccare il resto del codice.")
print()
print(f"      {'STRATEGIA':<16}{'FAMIGLIA':<30}{'DATI SU CUI LAVORA'}")
print("      " + "-" * 68)
righe = [
    ("KS", "Test statistico a 2 campioni", "continui"),
    ("ADWIN", "Finestra adattiva", "limitati in [0,1]"),
    ("DDM", "Error-based supervisionato", "errori binari 0/1"),
    ("PageHinkley", "Analisi sequenziale (CUSUM)", "continui non limitati"),
    ("Ensemble", "Aggregazione OR/AND/MAJORITY", "quelli dei membri"),
]
for nome, famiglia, dati in righe:
    print(f"      {nome:<16}{famiglia:<30}{dati}")

pausa()

# =========================================================================
titolo(3, "Rilevamento dal vivo su un drift in posizione nota")
# =========================================================================

from data.synthetic_generator import bernoulli_improvviso
from evaluation.drift_metrics import estrai_eventi

N, DRIFT = 2000, 1000
valori, punti_drift = bernoulli_improvviso(
    n_samples=N, p_before=0.2, p_after=0.6, drift_point=DRIFT, seed=1000,
)

print()
print(f"   Stream di {N} campioni. La probabilita' passa da 0.2 a 0.6")
print(f"   esattamente al passo {DRIFT}. Noi lo sappiamo, i detector no.")
print()
print(f"      {'STRATEGIA':<16}{'RILEVATO AL PASSO':<20}{'LATENZA'}")
print("      " + "-" * 50)

strategie = {
    "KS": KSStrategy(window_size=config.KS_WINDOW_SIZE, alpha=config.KS_ALPHA),
    "ADWIN": ADWINStrategy(delta=config.ADWIN_DELTA),
    "DDM": DDMStrategy(warm_start=config.DDM_WARM_START),
    "PageHinkley": PageHinkleyStrategy(mode=config.PH_MODE),
}

for nome, strategia in strategie.items():
    verdetti = []
    for v in valori:
        strategia.update(v)
        verdetti.append(strategia.detect().drift_detected)
    eventi = [e for e in estrai_eventi(verdetti, inizio=200) if e >= DRIFT]
    time.sleep(0.25)
    if eventi:
        print(f"      {nome:<16}{eventi[0]:<20}{eventi[0] - DRIFT} passi")
    else:
        print(f"      {nome:<16}{'non rilevato':<20}-")

pausa()

# =========================================================================
titolo(4, "Combinare le strategie: la regola a maggioranza")
# =========================================================================

print()
print("   Tre strategie votano sullo stesso stream. Serve il consenso di 2 su 3.")
print()

ensemble = EnsembleStrategy(
    strategy_specs=[
        (KSStrategy, {"window_size": config.KS_WINDOW_SIZE, "alpha": config.KS_ALPHA}),
        (ADWINStrategy, {"delta": config.ADWIN_DELTA}),
        (PageHinkleyStrategy, {"mode": config.PH_MODE}),
    ],
    aggregation="MAJORITY",
)

verdetti = []
for v in valori:
    ensemble.update(v)
    verdetti.append(ensemble.detect().drift_detected)
eventi = [e for e in estrai_eventi(verdetti, inizio=200) if e >= DRIFT]
print(f"      Ensemble MAJORITY   ->  rilevato al passo {eventi[0]}, "
      f"latenza {eventi[0] - DRIFT} passi")

print()
print("   Ogni voto individuale resta ispezionabile nel risultato:")
print(f"      {ensemble.detect().metadata['strategy_verdicts']}")

pausa()

# =========================================================================
titolo(5, "Il caso regressione: non tutti gli algoritmi sono applicabili")
# =========================================================================

from detectors.concept_drift_detector import (
    ConceptDriftDetector, errore_assoluto,
)

print()
print("   L'errore di un classificatore e' binario: 0 o 1.")
print("   L'errore di un regressore e' un numero reale senza limite superiore.")
print("   Non e' un dettaglio: cambia quali algoritmi si possono usare.")
print()

print("   DDM su un errore di regressione (3.7):")
try:
    DDMStrategy().update(3.7)
    print("      ...accettato (non dovrebbe succedere)")
except ValueError as e:
    print(f"      RIFIUTATO -> {e}")

print()
print("   Page-Hinkley sullo stesso stream:")
rng = np.random.default_rng(21)
detector = ConceptDriftDetector(
    PageHinkleyStrategy, error_fn=errore_assoluto, mode="up",
)
rilevati = []
for t in range(2000):
    scarto = rng.normal(0, 0.3) if t < 1000 else rng.normal(2.5, 0.3)
    detector.update(10.0 + scarto, 10.0)
    if detector.detect().drift_detected:
        rilevati.append(t)
print(f"      Errore del modello peggiora al passo 1000 "
      f"-> rilevato al passo {rilevati[0]}, latenza {rilevati[0] - 1000} passi")

pausa()

# =========================================================================
titolo(6, "Il controllo negativo: quando il framework ha trovato un difetto")
# =========================================================================

print()
print("   Nei dataset Friedman le feature sono uniformi in [0,1] SEMPRE.")
print("   A cambiare e' solo la funzione che lega feature e target.")
print("   Quindi un detector di FEATURE drift non deve trovare nulla:")
print("   ogni segnalazione e' per definizione un falso positivo.")
print()
print("   Con 10 feature e alpha=0.05, la probabilita' teorica che almeno")
print("   un test sbagli a ogni passo e':")
print()
print("        1 - (1 - 0.05)^10  =  40.1%")
print()
print("   Misurato sul controllo negativo:")
print()
print(f"      {'CONFIGURAZIONE':<32}{'TEMPO IN ALLARME'}")
print("      " + "-" * 52)
print(f"      {'KS senza correzione':<32}33.3%   <- il difetto")
print(f"      {'KS + Bonferroni':<32} 2.9%   <- corretto")
print(f"      {'KS + Benjamini-Hochberg':<32} 3.1%")
print(f"      {'ADWIN':<32} 0.0%")
print()
print("   Non era un difetto del test KS, ma della regola di aggregazione.")
print("   Ora la correzione e' un parametro del FeatureDriftDetector.")

pausa()

# =========================================================================
titolo(7, "Tutto e' verificato da test automatici")
# =========================================================================

print()
print("   56 test coprono metriche, contratto delle strategie, ensemble,")
print("   correzione statistica e concept drift in regressione.")
print()
print("      pytest tests/ -q")
print()

print("=" * 74)
print("  Fine demo.  Le figure sono in thesis/figures/")
print("=" * 74)
print()
