"""Concept drift su modelli di REGRESSIONE — dataset Friedman con drift noto.

Perche' questo esperimento
--------------------------
Tutti gli esperimenti precedenti riguardano la classificazione. La proposta di
tesi parla pero' esplicitamente di "classification and regression", e la
regressione non e' una semplice ripetizione con un modello diverso: cambia la
natura dello stream monitorato e quindi cambiano le strategie applicabili.

L'errore di un classificatore e' binario (0/1), e su di esso DDM e' il metodo
elettivo. L'errore di un regressore e' |y_vero - y_predetto|, cioe' un numero
reale positivo NON LIMITATO. Su quello stream:

- DDM non e' applicabile NELL'IMPLEMENTAZIONE USATA: river deriva la
  deviazione standard dalla binomiale e presuppone quindi un errore binario.
  Il wrapper solleva un'eccezione invece di produrre numeri privi di
  significato. Il metodo in se' non ha questo limite: Gama et al. (2004) ne
  prospettano l'uso con qualunque funzione di perdita.
- ADWIN e' applicabile ma fuori specifica: le garanzie del Teorema 3.1 valgono
  per variabili limitate in [0,1]. Viene incluso proprio per misurare cosa
  succede quando lo si usa fuori dalle sue ipotesi.
- KS e' applicabile e nel suo dominio d'elezione, perche' l'errore e' continuo.
- Page-Hinkley e' la scelta naturale: nato per rilevare spostamenti della media
  di un segnale continuo.

Il controllo negativo
---------------------
I dataset Friedman hanno una proprieta' preziosa: le 10 feature sono sempre
uniformi in [0,1], mentre a cambiare e' la FUNZIONE che lega feature e target.
E' quindi concept drift PURO, con P(X) costante e P(y|X) variabile.

Ne segue una previsione verificabile: i detector di FEATURE drift non devono
rilevare nulla. Se rilevassero qualcosa sarebbero falsi positivi. E' il
controllo negativo che dimostra sperimentalmente la distinzione fra data drift
e concept drift, e il motivo per cui monitorare solo le feature in ingresso
non basta a proteggere un modello in produzione.

Limite dichiarato
-----------------
I tre dataset sono file fissi versionati in repository, generati una volta con
seed 42. La valutazione non e' quindi ripetuta su piu' seed come negli
esperimenti sintetici Bernoulli: le metriche qui riportate provengono da una
singola esecuzione per variante, su 7 punti di drift complessivi. Vanno lette
come indicative, non come stime con incertezza quantificata.

Lancio dalla radice del progetto:
    python -m evaluation.runner_friedman
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from config import (
    ADWIN_DELTA,
    FRIEDMAN_TRAIN_SIZE,
    KS_ALPHA,
    KS_WINDOW_SIZE,
    PH_ALPHA,
    PH_DELTA,
    PH_MIN_INSTANCES,
    PH_MODE,
    PH_THRESHOLD,
)
from detectors.adwin_strategy import ADWINStrategy
from detectors.concept_drift_detector import ConceptDriftDetector, errore_assoluto
from detectors.ensemble_strategy import EnsembleStrategy
from detectors.feature_drift_detector import FeatureDriftDetector
from detectors.ks_strategy import KSStrategy
from detectors.page_hinkley_strategy import PageHinkleyStrategy
from evaluation.drift_metrics import conta_episodi, estrai_eventi, valuta

CARTELLA_DATI = Path("data")
CARTELLA_RISULTATI = Path("results/friedman")
CARTELLA_FIGURE = Path("thesis/figures")

# La tolleranza e' piu' ampia di quella usata sui Bernoulli (500) perche' la
# variante 'gsg' ha una finestra di transizione di 1000 campioni: nessun
# detector puo' ragionevolmente reagire prima che la transizione sia in corso.
TOLLERANZA = 1000

INIZIO_VALUTAZIONE = 2 * KS_WINDOW_SIZE

FINESTRA_ERRORE = 500


def parametri():
    ks = {"window_size": KS_WINDOW_SIZE, "alpha": KS_ALPHA}
    adwin = {"delta": ADWIN_DELTA}
    ph = {
        "min_instances": PH_MIN_INSTANCES,
        "delta": PH_DELTA,
        "threshold": PH_THRESHOLD,
        "alpha": PH_ALPHA,
        "mode": PH_MODE,
    }
    return ks, adwin, ph


def carica_metadati():
    with open(CARTELLA_DATI / "friedman_metadata.json", encoding="utf-8") as f:
        return json.load(f)


def strategie_concept():
    """Strategie applicabili a uno stream di errori CONTINUO.

    DDM e' deliberatamente assente: non e' applicabile a errori reali, e la
    verifica che sollevi un'eccezione sta nella suite di test.
    """
    ks, adwin, ph = parametri()
    return {
        "KS": (KSStrategy, ks),
        "ADWIN": (ADWINStrategy, adwin),
        "PageHinkley": (PageHinkleyStrategy, ph),
        "Ensemble-MAJORITY": (
            EnsembleStrategy,
            {
                "strategy_specs": [
                    (KSStrategy, ks),
                    (ADWINStrategy, adwin),
                    (PageHinkleyStrategy, ph),
                ],
                "aggregation": "MAJORITY",
            },
        ),
    }


def esegui_variante(nome, meta):
    """Addestra il regressore, produce lo stream di errori e applica i detector."""
    df = pd.read_csv(CARTELLA_DATI / meta["file"])
    colonne_feature = [c for c in df.columns if c != "y"]
    X = df[colonne_feature].values
    y = df["y"].values

    # --- Modello statico, addestrato una volta sola ---
    # Stessa impostazione dell'esperimento ELEC2: il modello non viene mai
    # aggiornato, cosi' il degrado osservato e' interamente imputabile al
    # cambiamento del concetto e non all'apprendimento.
    modello = DecisionTreeRegressor(max_depth=8, random_state=42)
    modello.fit(X[:FRIEDMAN_TRAIN_SIZE], y[:FRIEDMAN_TRAIN_SIZE])

    # Il modello e' statico: le predizioni non dipendono dall'ordine di
    # arrivo, quindi si calcolano in un colpo solo.
    X_stream = X[FRIEDMAN_TRAIN_SIZE:]
    y_stream = y[FRIEDMAN_TRAIN_SIZE:]
    y_pred = modello.predict(X_stream)
    errori = np.abs(y_stream - y_pred)

    # Ground truth riportata all'origine dello stream.
    drift_assoluti = meta["drift_positions"]
    drift_relativi = [d - FRIEDMAN_TRAIN_SIZE for d in drift_assoluti
                      if d > FRIEDMAN_TRAIN_SIZE]

    n_stream = len(errori)

    # --- Concept drift: detector sullo stream degli errori ---
    risultati_concept = {}
    for nome_strategia, (cls, kwargs) in strategie_concept().items():
        detector = ConceptDriftDetector(cls, error_fn=errore_assoluto, **kwargs)
        verdetti = []
        for i in range(n_stream):
            detector.update(y_pred[i], y_stream[i])
            verdetti.append(detector.detect().drift_detected)

        eventi = estrai_eventi(verdetti, inizio=INIZIO_VALUTAZIONE)
        m = valuta(eventi, drift_relativi, n_stream, TOLLERANZA, INIZIO_VALUTAZIONE)
        ep = conta_episodi(verdetti, inizio=INIZIO_VALUTAZIONE)

        risultati_concept[nome_strategia] = {
            "eventi": [int(e) for e in eventi],
            "n_episodi": int(ep["n_episodi"]),
            "n_rilevati": m.n_rilevati,
            "n_drift_reali": m.n_drift_reali,
            "n_falsi_allarmi": m.n_falsi_allarmi,
            "n_mancati": m.n_mancati,
            "latenza_media": m.latenza_media,
            "latenze": m.latenze,
            "falsi_allarmi_per_1000": m.falsi_allarmi_per_1000,
            "tasso_mancate": m.tasso_mancate,
        }

    # --- Controllo negativo: feature drift, che NON dovrebbe esserci ---
    # Le feature Friedman sono uniformi in [0,1] per costruzione: ogni
    # segnalazione qui e' per definizione un falso positivo. Il controllo
    # viene ripetuto con e senza correzione per test multipli, per misurarne
    # l'effetto invece di darlo per scontato.
    ks, adwin, _ = parametri()
    n_feature = len(colonne_feature)
    configurazioni = {
        "KS (nessuna correzione)": (KSStrategy, ks, None),
        "KS + Bonferroni": (KSStrategy, ks, "bonferroni"),
        "KS + Benjamini-Hochberg": (KSStrategy, ks, "benjamini-hochberg"),
        "ADWIN": (ADWINStrategy, adwin, None),
    }

    risultati_feature = {}
    for etichetta, (cls, kwargs, correzione) in configurazioni.items():
        extra = ({"correzione": correzione, "alpha_correzione": KS_ALPHA}
                 if correzione else {})
        detector = FeatureDriftDetector(
            cls, n_features=n_feature, k=1,
            feature_names=colonne_feature, **extra, **kwargs,
        )
        verdetti = []
        for riga in X_stream:
            detector.update(riga)
            verdetti.append(detector.detect().drift_detected)

        eventi = estrai_eventi(verdetti, inizio=INIZIO_VALUTAZIONE)
        ep = conta_episodi(verdetti, inizio=INIZIO_VALUTAZIONE)
        risultati_feature[etichetta] = {
            "n_episodi": int(ep["n_episodi"]),
            "frazione_tempo_in_allarme": round(float(ep["frazione_tempo"]), 4),
            "gia_in_drift": bool(ep["gia_in_drift"]),
            "n_eventi": len(eventi),
        }

    # Tasso di falso positivo atteso senza correzione, per confronto.
    risultati_feature["_atteso_senza_correzione"] = {
        "n_episodi": None,
        "frazione_tempo_in_allarme": round(1 - (1 - KS_ALPHA) ** n_feature, 4),
        "gia_in_drift": None,
        "n_eventi": None,
    }

    # --- Degrado del modello: MAE a finestre ---
    curva_mae = []
    for inizio in range(0, n_stream - FINESTRA_ERRORE, FINESTRA_ERRORE):
        finestra = errori[inizio:inizio + FINESTRA_ERRORE]
        curva_mae.append((inizio + FINESTRA_ERRORE, float(finestra.mean())))

    return {
        "variante": nome,
        "descrizione": meta["descrizione"],
        "drift_relativi": drift_relativi,
        "drift_assoluti": drift_assoluti,
        "n_stream": n_stream,
        "errori": errori,
        "curva_mae": curva_mae,
        "mae_globale": float(errori.mean()),
        "concept": risultati_concept,
        "feature": risultati_feature,
    }


def figura_errore(risultati):
    """Errore del regressore nel tempo, con i drift reali e quelli rilevati."""
    fig, assi = plt.subplots(len(risultati), 1, figsize=(13, 3.4 * len(risultati)),
                             sharex=False)
    if len(risultati) == 1:
        assi = [assi]

    stili = {
        "KS": ("#1f77b4", 0.93),
        "ADWIN": ("#d62728", 0.86),
        "PageHinkley": ("#9467bd", 0.79),
        "Ensemble-MAJORITY": ("#ff7f0e", 0.72),
    }

    for ax, r in zip(assi, risultati):
        passi = [p for p, _ in r["curva_mae"]]
        valori = [v for _, v in r["curva_mae"]]
        ax.plot(passi, valori, color="#333333", linewidth=1.2,
                label=f"MAE su finestre di {FINESTRA_ERRORE} campioni")

        for d in r["drift_relativi"]:
            ax.axvline(d, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axvline(r["drift_relativi"][0] if r["drift_relativi"] else 0,
                   color="black", linestyle="--", linewidth=1.2, alpha=0.7,
                   label="Drift reale (ground truth)")

        alto = max(valori) * 1.05 if valori else 1
        for nome, (colore, quota) in stili.items():
            eventi = r["concept"][nome]["eventi"]
            if not eventi:
                continue
            ax.plot(eventi, [alto * quota] * len(eventi), "|", color=colore,
                    markersize=10, markeredgewidth=1.6,
                    label=f"{nome} ({len(eventi)} eventi)")

        ax.set_title(f"{r['variante']} — {r['descrizione']}", fontsize=10)
        ax.set_ylabel("Errore assoluto medio")
        ax.grid(alpha=0.3, linestyle=":")
        # Legenda fuori dagli assi: dentro coprirebbe i marker dei rilevamenti,
        # che stanno nella banda alta del grafico.
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  frameon=False)

    assi[-1].set_xlabel("Passo dello stream")
    fig.suptitle("Friedman: degrado di un regressore statico e concept drift rilevato",
                 fontsize=12, y=1.0)
    fig.tight_layout()

    CARTELLA_FIGURE.mkdir(parents=True, exist_ok=True)
    for estensione in ("pdf", "png"):
        fig.savefig(CARTELLA_FIGURE / f"friedman_concept_drift.{estensione}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  {CARTELLA_FIGURE / 'friedman_concept_drift'}.pdf / .png")


def salva(risultati):
    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    righe = []
    for r in risultati:
        for nome, m in r["concept"].items():
            righe.append({
                "variante": r["variante"],
                "strategia": nome,
                "n_drift_reali": m["n_drift_reali"],
                "n_rilevati": m["n_rilevati"],
                "n_mancati": m["n_mancati"],
                "n_falsi_allarmi": m["n_falsi_allarmi"],
                "latenza_media": (round(m["latenza_media"], 1)
                                  if m["latenza_media"] is not None else None),
                "falsi_allarmi_per_1000": (round(m["falsi_allarmi_per_1000"], 3)
                                           if m["falsi_allarmi_per_1000"] is not None
                                           else None),
                "n_episodi": m["n_episodi"],
            })

    with open(CARTELLA_RISULTATI / "concept_drift.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        writer.writeheader()
        writer.writerows(righe)

    righe_feature = []
    for r in risultati:
        for nome, m in r["feature"].items():
            righe_feature.append({
                "variante": r["variante"], "strategia": nome, **m,
            })
    with open(CARTELLA_RISULTATI / "controllo_negativo_feature.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(righe_feature[0].keys()))
        writer.writeheader()
        writer.writerows(righe_feature)

    with open(CARTELLA_RISULTATI / "riepilogo.json", "w", encoding="utf-8") as f:
        json.dump({
            "generato_il": datetime.now().isoformat(timespec="seconds"),
            "modello": "DecisionTreeRegressor(max_depth=8), statico",
            "train_size": FRIEDMAN_TRAIN_SIZE,
            "tolleranza": TOLLERANZA,
            "inizio_valutazione": INIZIO_VALUTAZIONE,
            "varianti": [
                {
                    "variante": r["variante"],
                    "descrizione": r["descrizione"],
                    "drift_assoluti": r["drift_assoluti"],
                    "mae_globale": round(r["mae_globale"], 4),
                    "concept": {
                        k: {kk: vv for kk, vv in v.items() if kk != "eventi"}
                        for k, v in r["concept"].items()
                    },
                    "controllo_negativo_feature": r["feature"],
                }
                for r in risultati
            ],
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRisultati in {CARTELLA_RISULTATI}/")


def main():
    print("=" * 78)
    print("Friedman — concept drift su modello di REGRESSIONE")
    print("=" * 78)
    print(f"Modello: DecisionTreeRegressor(max_depth=8), statico")
    print(f"Training: primi {FRIEDMAN_TRAIN_SIZE} campioni")
    print(f"Stream monitorato: errore assoluto |y_vero - y_predetto|")
    print(f"Tolleranza: {TOLLERANZA} passi")

    metadati = carica_metadati()
    risultati = []
    for nome, meta in metadati.items():
        print(f"\nVariante '{nome}' — {meta['descrizione']}")
        print(f"  drift reali ai campioni {meta['drift_positions']}")
        r = esegui_variante(nome, meta)
        risultati.append(r)

        print(f"  MAE globale del modello statico: {r['mae_globale']:.3f}")
        print(f"  {'strategia':<20}{'rilevati':>10}{'mancati':>9}"
              f"{'falsi':>7}{'latenza':>10}{'episodi':>9}")
        for nome_strategia, m in r["concept"].items():
            lat = "n/d" if m["latenza_media"] is None else f"{m['latenza_media']:.0f}"
            print(f"  {nome_strategia:<20}"
                  f"{m['n_rilevati']:>6}/{m['n_drift_reali']:<3}"
                  f"{m['n_mancati']:>9}{m['n_falsi_allarmi']:>7}"
                  f"{lat:>10}{m['n_episodi']:>9}")

        print(f"  CONTROLLO NEGATIVO (feature drift: P(X) e' costante, atteso nessuno)")
        atteso = r["feature"]["_atteso_senza_correzione"]["frazione_tempo_in_allarme"]
        print(f"    {'falso positivo atteso senza correzione':<28} "
              f"{100 * atteso:>10.1f}%")
        for etichetta, m in r["feature"].items():
            if etichetta.startswith("_"):
                continue
            print(f"    {etichetta:<28} episodi={m['n_episodi']:<6} "
                  f"tempo in allarme={100 * m['frazione_tempo_in_allarme']:>5.1f}%")

    print("\nFigure:")
    figura_errore(risultati)
    salva(risultati)


if __name__ == "__main__":
    main()
