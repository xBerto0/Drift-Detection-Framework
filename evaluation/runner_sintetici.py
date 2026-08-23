"""Protocollo di valutazione delle strategie su stream sintetici con drift noto.

Esegue la matrice completa STRATEGIA x TIPO DI DRIFT x SEED e produce, per
ogni cella, latenza di rilevamento, falsi allarmi e mancate rilevazioni con
media e deviazione standard sulle ripetizioni.

E' il cuore della validazione sperimentale: a differenza degli script
`experiment_*.py`, che descrivono quante volte un detector ha segnalato
qualcosa, questo modulo MISURA quanto bene lo ha fatto rispetto a una verita'
nota per costruzione.

Output prodotti in `results/sintetici/`:
    metriche_aggregate.csv    una riga per cella della matrice
    metriche_per_seed.csv     una riga per singola esecuzione
    metriche_aggregate.json   stesse informazioni in forma annidata

Lancio dalla radice del progetto:
    python evaluation/runner_sintetici.py
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from config import (
    ADWIN_DELTA,
    DDM_DRIFT_THRESHOLD,
    DDM_WARM_START,
    DDM_WARNING_THRESHOLD,
    KS_ALPHA,
    KS_WINDOW_SIZE,
    PH_ALPHA,
    PH_DELTA,
    PH_MIN_INSTANCES,
    PH_MODE,
    PH_THRESHOLD,
)
from data.synthetic_generator import (
    bernoulli_graduale,
    bernoulli_improvviso,
    bernoulli_incrementale,
    bernoulli_ricorrente,
    bernoulli_stazionario,
)
from detectors.adwin_strategy import ADWINStrategy
from detectors.ddm_strategy import DDMStrategy
from detectors.ensemble_strategy import EnsembleStrategy
from detectors.ks_strategy import KSStrategy
from detectors.page_hinkley_strategy import PageHinkleyStrategy
from evaluation.drift_metrics import aggrega, estrai_eventi, valuta


# ---------------------------------------------------------------------------
# Parametri del protocollo
# ---------------------------------------------------------------------------

N_CAMPIONI = 6000
N_SEED = 20
SEED_BASE = 1000

# Il drift porta il parametro da 0.2 a 0.6, cioe' e' un AUMENTO.
# La scelta non e' neutra: DDM e Page-Hinkley con mode='up' rilevano per
# costruzione solo le crescite. Usare un aumento mette tutte le strategie
# nelle condizioni di poter rilevare, rendendo il confronto equo.
# L'asimmetria rispetto a un drift in diminuzione e' misurata a parte, nella
# sezione finale di questo script.
P_INIZIALE = 0.2
P_FINALE = 0.6

PRIMO_DRIFT = 2000
SECONDO_DRIFT = 4000
FINESTRA_TRANSIZIONE = 1000

# Tolleranza per considerare corretta una segnalazione. Deve restare ben
# sotto la distanza fra due drift consecutivi (2000) per evitare che le
# finestre di valutazione si sovrappongano.
TOLLERANZA = 500

# I primi passi non vengono valutati: il KS ha bisogno di riempire reference e
# current (2 x window_size) e DDM di completare il warm start. Escludendo la
# stessa regione iniziale per tutte le strategie il confronto resta equo.
INIZIO_VALUTAZIONE = max(2 * KS_WINDOW_SIZE, DDM_WARM_START)


def costruisci_scenari():
    """Restituisce i cinque scenari sintetici, ognuno con la sua ground truth.

    Ogni voce e' una funzione che, dato un seed, produce (valori, punti_drift).
    """
    return {
        "stazionario": lambda seed: bernoulli_stazionario(
            N_CAMPIONI, p=P_INIZIALE, seed=seed,
        ),
        "improvviso": lambda seed: bernoulli_improvviso(
            N_CAMPIONI, P_INIZIALE, P_FINALE, PRIMO_DRIFT, seed=seed,
        ),
        "graduale": lambda seed: bernoulli_graduale(
            N_CAMPIONI, P_INIZIALE, P_FINALE, PRIMO_DRIFT,
            FINESTRA_TRANSIZIONE, seed=seed,
        ),
        "incrementale": lambda seed: bernoulli_incrementale(
            N_CAMPIONI, P_INIZIALE, P_FINALE, PRIMO_DRIFT,
            FINESTRA_TRANSIZIONE, seed=seed,
        ),
        "ricorrente": lambda seed: bernoulli_ricorrente(
            N_CAMPIONI, P_INIZIALE, P_FINALE,
            [PRIMO_DRIFT, SECONDO_DRIFT], seed=seed,
        ),
    }


def costruisci_strategie():
    """Restituisce le strategie da confrontare, come (classe, parametri).

    L'ensemble contiene le tre strategie NON supervisionate, quelle cioe' che
    lavorano direttamente sul valore dello stream: KS, ADWIN e Page-Hinkley.
    DDM resta fuori dall'ensemble perche' e' concettualmente un rilevatore di
    errore, non di distribuzione; con tre membri la regola MAJORITY richiede
    il consenso di due strategie su tre e non degenera in un AND.
    """
    parametri_ks = {"window_size": KS_WINDOW_SIZE, "alpha": KS_ALPHA}
    parametri_adwin = {"delta": ADWIN_DELTA}
    parametri_ph = {
        "min_instances": PH_MIN_INSTANCES,
        "delta": PH_DELTA,
        "threshold": PH_THRESHOLD,
        "alpha": PH_ALPHA,
        "mode": PH_MODE,
    }
    parametri_ddm = {
        "warm_start": DDM_WARM_START,
        "warning_threshold": DDM_WARNING_THRESHOLD,
        "drift_threshold": DDM_DRIFT_THRESHOLD,
    }

    return {
        "KS": (KSStrategy, parametri_ks),
        "ADWIN": (ADWINStrategy, parametri_adwin),
        "DDM": (DDMStrategy, parametri_ddm),
        "PageHinkley": (PageHinkleyStrategy, parametri_ph),
        "Ensemble-MAJORITY": (
            EnsembleStrategy,
            {
                "strategy_specs": [
                    (KSStrategy, parametri_ks),
                    (ADWINStrategy, parametri_adwin),
                    (PageHinkleyStrategy, parametri_ph),
                ],
                "aggregation": "MAJORITY",
            },
        ),
    }


def esegui_singola(strategy_cls, parametri, valori):
    """Fa scorrere lo stream nella strategia e raccoglie i verdetti passo per passo."""
    strategia = strategy_cls(**parametri)
    verdetti = []
    for valore in valori:
        strategia.update(valore)
        verdetti.append(strategia.detect().drift_detected)
    return verdetti


def esegui_matrice():
    """Esegue la matrice completa e restituisce i risultati per seed e aggregati."""
    scenari = costruisci_scenari()
    strategie = costruisci_strategie()

    righe_per_seed = []
    aggregati = {}

    for nome_scenario, generatore in scenari.items():
        for nome_strategia, (cls, parametri) in strategie.items():
            metriche_seed = []

            for i in range(N_SEED):
                seed = SEED_BASE + i
                valori, punti_drift = generatore(seed)

                verdetti = esegui_singola(cls, parametri, valori)
                eventi = estrai_eventi(verdetti, inizio=INIZIO_VALUTAZIONE)
                m = valuta(
                    eventi=eventi,
                    punti_drift=punti_drift,
                    n_campioni=N_CAMPIONI,
                    tolleranza=TOLLERANZA,
                    inizio=INIZIO_VALUTAZIONE,
                )
                metriche_seed.append(m)

                righe_per_seed.append({
                    "scenario": nome_scenario,
                    "strategia": nome_strategia,
                    "seed": seed,
                    "n_drift_reali": m.n_drift_reali,
                    "n_eventi": m.n_eventi,
                    "n_rilevati": m.n_rilevati,
                    "n_falsi_allarmi": m.n_falsi_allarmi,
                    "n_mancati": m.n_mancati,
                    "n_duplicati": m.n_duplicati,
                    "latenza_media": m.latenza_media,
                    "falsi_allarmi_per_1000": m.falsi_allarmi_per_1000,
                })

            chiave = f"{nome_scenario}|{nome_strategia}"
            aggregati[chiave] = aggrega(metriche_seed)
            aggregati[chiave]["scenario"] = nome_scenario
            aggregati[chiave]["strategia"] = nome_strategia

            sintesi = aggregati[chiave]
            lat = sintesi["latenza_media"]
            far = sintesi["falsi_allarmi_per_1000_media"]
            mdr = sintesi["tasso_mancate_media"]
            print(
                f"  {nome_scenario:14s} {nome_strategia:18s} "
                f"latenza={_fmt(lat, 1):>8s}  "
                f"FA/1000={_fmt(far, 2):>6s}  "
                f"MDR={_fmt(mdr, 2):>5s}"
            )

    return righe_per_seed, aggregati


def misura_asimmetria_direzionale():
    """Verifica se le strategie rilevano anche un drift in DIMINUZIONE.

    DDM e Page-Hinkley con mode='up' sono unilaterali per costruzione:
    reagiscono a un aumento e non a un calo. Su un rilevatore di errore la
    scelta e' corretta (un errore che cala non e' un allarme), ma su una
    feature generica diventa un limite di applicabilita' che va misurato e
    dichiarato, non lasciato implicito.

    Confronta lo stesso scenario improvviso nelle due direzioni.
    """
    strategie = costruisci_strategie()
    risultati = {}

    direzioni = {
        "aumento (0.2 -> 0.6)": (P_INIZIALE, P_FINALE),
        "diminuzione (0.6 -> 0.2)": (P_FINALE, P_INIZIALE),
    }

    for nome_direzione, (p_prima, p_dopo) in direzioni.items():
        for nome_strategia, (cls, parametri) in strategie.items():
            metriche_seed = []
            for i in range(N_SEED):
                seed = SEED_BASE + i
                valori, punti_drift = bernoulli_improvviso(
                    N_CAMPIONI, p_prima, p_dopo, PRIMO_DRIFT, seed=seed,
                )
                verdetti = esegui_singola(cls, parametri, valori)
                eventi = estrai_eventi(verdetti, inizio=INIZIO_VALUTAZIONE)
                metriche_seed.append(valuta(
                    eventi, punti_drift, N_CAMPIONI,
                    TOLLERANZA, INIZIO_VALUTAZIONE,
                ))

            sintesi = aggrega(metriche_seed)
            sintesi["direzione"] = nome_direzione
            sintesi["strategia"] = nome_strategia
            risultati[f"{nome_direzione}|{nome_strategia}"] = sintesi

            print(
                f"  {nome_direzione:26s} {nome_strategia:18s} "
                f"rilevati={sintesi['n_rilevati_totali']:3d}/"
                f"{sintesi['n_drift_reali_totali']:<3d}  "
                f"latenza={_fmt(sintesi['latenza_media'], 1):>8s}"
            )

    return risultati


def _fmt(valore, decimali):
    """Formatta un numero che puo' essere None."""
    if valore is None:
        return "n/d"
    return f"{valore:.{decimali}f}"


def salva(righe_per_seed, aggregati, asimmetria, cartella):
    """Scrive i risultati in CSV e JSON."""
    cartella.mkdir(parents=True, exist_ok=True)

    percorso_seed = cartella / "metriche_per_seed.csv"
    with open(percorso_seed, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(righe_per_seed[0].keys()))
        writer.writeheader()
        writer.writerows(righe_per_seed)

    colonne = [
        "scenario", "strategia", "n_ripetizioni",
        "latenza_media", "latenza_std",
        "falsi_allarmi_per_1000_media", "falsi_allarmi_per_1000_std",
        "tasso_mancate_media", "tasso_mancate_std",
        "n_eventi_media", "seed_con_falsi_allarmi",
        "n_rilevati_totali", "n_drift_reali_totali",
    ]
    percorso_agg = cartella / "metriche_aggregate.csv"
    with open(percorso_agg, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colonne, extrasaction="ignore")
        writer.writeheader()
        for valori in aggregati.values():
            writer.writerows([valori])

    percorso_json = cartella / "metriche_aggregate.json"
    with open(percorso_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generato_il": datetime.now().isoformat(timespec="seconds"),
                "protocollo": {
                    "n_campioni": N_CAMPIONI,
                    "n_seed": N_SEED,
                    "seed_base": SEED_BASE,
                    "p_iniziale": P_INIZIALE,
                    "p_finale": P_FINALE,
                    "primo_drift": PRIMO_DRIFT,
                    "secondo_drift": SECONDO_DRIFT,
                    "finestra_transizione": FINESTRA_TRANSIZIONE,
                    "tolleranza": TOLLERANZA,
                    "inizio_valutazione": INIZIO_VALUTAZIONE,
                },
                "matrice": aggregati,
                "asimmetria_direzionale": asimmetria,
            },
            f, indent=2, ensure_ascii=False,
        )

    print(f"\nRisultati salvati in:")
    print(f"  {percorso_agg}")
    print(f"  {percorso_seed}")
    print(f"  {percorso_json}")


def main():
    print("=" * 78)
    print("Protocollo di valutazione su stream sintetici con drift noto")
    print("=" * 78)
    print(f"Campioni per stream : {N_CAMPIONI}")
    print(f"Ripetizioni (seed)  : {N_SEED}")
    print(f"Tolleranza          : {TOLLERANZA} passi")
    print(f"Inizio valutazione  : passo {INIZIO_VALUTAZIONE} (esclusa fase di warm-up)")
    print(f"Drift               : p da {P_INIZIALE} a {P_FINALE}")
    print()
    print("-" * 78)
    print("MATRICE  scenario x strategia")
    print("-" * 78)
    righe_per_seed, aggregati = esegui_matrice()

    print()
    print("-" * 78)
    print("ASIMMETRIA DIREZIONALE  (drift in aumento vs in diminuzione)")
    print("-" * 78)
    asimmetria = misura_asimmetria_direzionale()

    salva(righe_per_seed, aggregati, asimmetria, Path("results/sintetici"))


if __name__ == "__main__":
    main()
