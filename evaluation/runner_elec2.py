"""Rianalisi di ELEC2 con conteggio a episodi invece che a passi.

Perche' questo script esiste
----------------------------
Gli script `experiment_ks_elec2.py` e `experiment_adwin_elec2.py` contano,
per ogni detector, il NUMERO DI PASSI in cui il verdetto vale True. Su un
detector edge-triggered come ADWIN quel numero coincide con il numero di
eventi. Su un detector level-triggered come il KS no: il test viene rifatto
a ogni passo su finestre che si sovrappongono quasi completamente e resta
True finche' la condizione persiste, quindi un singolo episodio di drift
produce migliaia di conteggi.

Da qui il risultato riportato finora, "il KS segnala nel 99,1% dei passi e
ADWIN nell'1,2%, un rapporto di 80 a 1", che confronta grandezze non
omogenee. Questo script ricalcola tutto contando EPISODI (transizioni da
non-drift a drift) e riporta le due misure affiancate, cosi' che la
correzione sia esplicita e documentata invece che silenziosa.

Cosa NON si puo' fare su ELEC2
------------------------------
ELEC2 e' un dataset reale: non esiste una ground truth dei punti di drift.
Latenza, falsi allarmi e mancate rilevazioni non sono quindi calcolabili —
quelle metriche vivono negli esperimenti sintetici. Qui la valutazione resta
necessariamente qualitativa e si fonda sulla coerenza fra le segnalazioni dei
detector e il comportamento osservabile del classificatore.

Lancio dalla radice del progetto:
    python -m evaluation.runner_elec2
"""

import csv
import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from config import (
    ADWIN_DELTA,
    DATASET_ELEC2_PATH,
    DDM_DRIFT_THRESHOLD,
    DDM_WARM_START,
    DDM_WARNING_THRESHOLD,
    ELEC2_TRAIN_SIZE,
    FEATURE_K_THRESHOLD,
    KS_ALPHA,
    KS_WINDOW_SIZE,
    PH_ALPHA,
    PH_DELTA,
    PH_MIN_INSTANCES,
    PH_MODE,
    PH_THRESHOLD,
)
from detectors.adwin_strategy import ADWINStrategy
from detectors.concept_drift_detector import ConceptDriftDetector
from detectors.ddm_strategy import DDMStrategy
from detectors.ensemble_strategy import EnsembleStrategy
from detectors.feature_drift_detector import FeatureDriftDetector
from detectors.ks_strategy import KSStrategy
from detectors.page_hinkley_strategy import PageHinkleyStrategy
from detectors.prediction_drift_detector import PredictionDriftDetector
from evaluation.drift_metrics import conta_episodi, estrai_eventi

CARTELLA_RISULTATI = Path("results/elec2")
CARTELLA_FIGURE = Path("thesis/figures")

# Come negli esperimenti sintetici, i primi passi non si valutano: servono a
# riempire le finestre del KS e a completare il warm start di DDM.
INIZIO_VALUTAZIONE = max(2 * KS_WINDOW_SIZE, DDM_WARM_START)

FINESTRA_ACCURACY = 1000


def parametri():
    ks = {"window_size": KS_WINDOW_SIZE, "alpha": KS_ALPHA}
    adwin = {"delta": ADWIN_DELTA}
    ddm = {
        "warm_start": DDM_WARM_START,
        "warning_threshold": DDM_WARNING_THRESHOLD,
        "drift_threshold": DDM_DRIFT_THRESHOLD,
    }
    ph = {
        "min_instances": PH_MIN_INSTANCES,
        "delta": PH_DELTA,
        "threshold": PH_THRESHOLD,
        "alpha": PH_ALPHA,
        "mode": PH_MODE,
    }
    return ks, adwin, ddm, ph


def carica_dataset():
    df = pd.read_csv(DATASET_ELEC2_PATH)
    nomi_feature = [c for c in df.columns if c != "class"]
    X = df[nomi_feature].values.astype(float)
    y = np.array([1 if v == "UP" else 0 for v in df["class"].values])
    return X, y, nomi_feature


def controlla_qualita_dataset(X, nomi_feature):
    """Cerca regioni costanti nelle feature: sono artefatti, non drift.

    ELEC2 contiene un difetto noto: i dati del mercato di Victoria non erano
    raccolti nella prima parte del periodo osservato, e i valori mancanti sono
    stati riempiti con una costante. Le feature interessate restano quindi
    perfettamente costanti per decine di migliaia di campioni, poi diventano
    variabili di colpo.

    La conseguenza sul drift detection e' che qualunque test basato sulla
    DISTRIBUZIONE rileva un cambiamento massiccio nel punto in cui i dati veri
    cominciano ad arrivare: si passa da una delta di Dirac a una distribuzione
    reale. Non e' un drift del fenomeno osservato, e' un artefatto di
    imputazione dei mancanti.

    Il controllo va eseguito PRIMA di interpretare qualunque risultato, perche'
    distingue i drift reali da quelli strutturali.
    """
    righe = []
    for i, nome in enumerate(nomi_feature):
        colonna = X[:, i]
        diversi = np.where(colonna != colonna[0])[0]
        fine_costante = int(diversi[0]) if len(diversi) else len(colonna)
        if fine_costante > 1:
            prima = colonna[:fine_costante]
            dopo = colonna[fine_costante:]
            righe.append({
                "feature": nome,
                "fine_regione_costante": fine_costante,
                "quota_costante": round(100 * fine_costante / len(colonna), 1),
                "valore_costante": round(float(colonna[0]), 6),
                "media_dopo": round(float(dopo.mean()), 6) if len(dopo) else None,
                "std_dopo": round(float(dopo.std()), 6) if len(dopo) else None,
            })
    return righe


def esegui():
    X, y, nomi_feature = carica_dataset()
    n_campioni, n_feature = X.shape
    ks, adwin, ddm, ph = parametri()

    # --- Classificatore statico, addestrato una volta sola ---
    classificatore = GaussianNB()
    classificatore.fit(X[:ELEC2_TRAIN_SIZE], y[:ELEC2_TRAIN_SIZE])

    # --- Detector: feature, prediction, concept ---
    detector_feature = {
        "KS": FeatureDriftDetector(KSStrategy, n_feature, FEATURE_K_THRESHOLD,
                                   nomi_feature, **ks),
        "ADWIN": FeatureDriftDetector(ADWINStrategy, n_feature, FEATURE_K_THRESHOLD,
                                      nomi_feature, **adwin),
    }
    detector_prediction = {
        "KS": PredictionDriftDetector(KSStrategy, **ks),
        "ADWIN": PredictionDriftDetector(ADWINStrategy, **adwin),
    }
    detector_concept = {
        "ADWIN": ConceptDriftDetector(ADWINStrategy, **adwin),
        "DDM": ConceptDriftDetector(DDMStrategy, **ddm),
        "PageHinkley": ConceptDriftDetector(PageHinkleyStrategy, **ph),
        "Ensemble-MAJORITY": ConceptDriftDetector(
            EnsembleStrategy,
            strategy_specs=[
                (ADWINStrategy, adwin),
                (DDMStrategy, ddm),
                (PageHinkleyStrategy, ph),
            ],
            aggregation="MAJORITY",
        ),
    }

    # Verdetti passo per passo, per poterli poi collassare in episodi.
    verdetti = {f"feature|{k}": [] for k in detector_feature}
    verdetti.update({f"prediction|{k}": [] for k in detector_prediction})
    verdetti.update({f"concept|{k}": [] for k in detector_concept})

    # Verdetti per singola feature, solo per i detector di feature drift.
    verdetti_feature = {
        nome: {f: [] for f in nomi_feature} for nome in detector_feature
    }

    finestra_corretti = deque(maxlen=FINESTRA_ACCURACY)
    curva_accuracy = []       # (passo, accuracy mobile)
    warning_ddm = []

    print(f"Streaming su {n_campioni - ELEC2_TRAIN_SIZE} campioni...")

    for t in range(ELEC2_TRAIN_SIZE, n_campioni):
        x_t = X[t]
        y_pred = classificatore.predict(x_t.reshape(1, -1))[0]
        y_true = y[t]

        finestra_corretti.append(1 if y_pred == y_true else 0)
        if (t - ELEC2_TRAIN_SIZE + 1) % 100 == 0:
            curva_accuracy.append((t, sum(finestra_corretti) / len(finestra_corretti)))

        for nome, det in detector_feature.items():
            det.update(x_t)
            risultato = det.detect()
            verdetti[f"feature|{nome}"].append(risultato.drift_detected)
            drittate = set(risultato.metadata["drifted_features"])
            for f in nomi_feature:
                verdetti_feature[nome][f].append(f in drittate)

        for nome, det in detector_prediction.items():
            det.update(y_pred)
            verdetti[f"prediction|{nome}"].append(det.detect().drift_detected)

        for nome, det in detector_concept.items():
            det.update(y_pred, y_true)
            risultato = det.detect()
            verdetti[f"concept|{nome}"].append(risultato.drift_detected)
            if nome == "DDM" and risultato.metadata.get("warning_detected"):
                warning_ddm.append(t)

    return {
        "verdetti": verdetti,
        "verdetti_feature": verdetti_feature,
        "curva_accuracy": curva_accuracy,
        "warning_ddm": warning_ddm,
        "nomi_feature": nomi_feature,
        "n_campioni": n_campioni,
        "n_streaming": n_campioni - ELEC2_TRAIN_SIZE,
        "offset": ELEC2_TRAIN_SIZE,
    }


def analizza(dati):
    """Confronta il conteggio a passi con quello a episodi."""
    n_streaming = dati["n_streaming"]
    righe = []

    for chiave, sequenza in dati["verdetti"].items():
        tipo_drift, strategia = chiave.split("|")
        ep = conta_episodi(sequenza, inizio=INIZIO_VALUTAZIONE)
        fronti = estrai_eventi(sequenza, inizio=INIZIO_VALUTAZIONE)
        passi = int(ep["passi_in_drift"])
        righe.append({
            "tipo_drift": tipo_drift,
            "strategia": strategia,
            "passi_in_drift": passi,
            "percentuale_passi": round(100 * passi / n_streaming, 2),
            "episodi": int(ep["n_episodi"]),
            "durata_media_episodio": round(float(ep["durata_media"]), 1),
            # Segnala il caso in cui il detector era gia' in allarme quando la
            # valutazione e' cominciata, e quindi non ha prodotto alcun fronte
            # di salita: non e' silenzioso, e' permanentemente in drift.
            "gia_in_drift_allinizio": bool(ep["gia_in_drift"]),
            "primo_fronte": (
                int(fronti[0] + dati["offset"]) if fronti else None
            ),
        })

    righe_feature = []
    for strategia, per_feature in dati["verdetti_feature"].items():
        for nome_feature, sequenza in per_feature.items():
            ep = conta_episodi(sequenza, inizio=INIZIO_VALUTAZIONE)
            passi = int(ep["passi_in_drift"])
            righe_feature.append({
                "strategia": strategia,
                "feature": nome_feature,
                "passi_in_drift": passi,
                "percentuale_passi": round(100 * passi / n_streaming, 2),
                "episodi": int(ep["n_episodi"]),
                "durata_media_episodio": round(float(ep["durata_media"]), 1),
                "gia_in_drift_allinizio": bool(ep["gia_in_drift"]),
            })

    return righe, righe_feature


def figura_accuracy(dati):
    """Accuracy prequenziale nel tempo con i punti di rilevamento del concept drift."""
    passi = [p for p, _ in dati["curva_accuracy"]]
    valori = [a for _, a in dati["curva_accuracy"]]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(passi, valori, color="#333333", linewidth=1.2,
            label=f"Accuracy mobile ({FINESTRA_ACCURACY} campioni)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1,
               label="Baseline casuale (50%)")

    stili = {
        "ADWIN": ("#d62728", 0.94),
        "DDM": ("#2ca02c", 0.90),
        "PageHinkley": ("#9467bd", 0.86),
        "Ensemble-MAJORITY": ("#ff7f0e", 0.82),
    }
    for nome, (colore, altezza) in stili.items():
        sequenza = dati["verdetti"].get(f"concept|{nome}")
        if sequenza is None:
            continue
        episodi = [e + dati["offset"]
                   for e in estrai_eventi(sequenza, inizio=INIZIO_VALUTAZIONE)]
        if not episodi:
            continue
        ax.plot(episodi, [altezza] * len(episodi), "|", color=colore,
                markersize=9, markeredgewidth=1.4,
                label=f"{nome} ({len(episodi)} episodi)")

    ax.set_xlabel("Passo dello stream")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.2, 1.0)
    ax.set_title("ELEC2: degrado del classificatore statico ed episodi di concept drift")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()

    CARTELLA_FIGURE.mkdir(parents=True, exist_ok=True)
    for estensione in ("pdf", "png"):
        fig.savefig(CARTELLA_FIGURE / f"elec2_accuracy_concept_drift.{estensione}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  {CARTELLA_FIGURE / 'elec2_accuracy_concept_drift'}.pdf / .png")


def figura_confronto_conteggio(righe):
    """Passi in stato di drift contro episodi, in scala logaritmica.

    Rende visibile in un colpo d'occhio l'entita' della distorsione introdotta
    dal conteggio a passi sui detector level-triggered.
    """
    feature = [r for r in righe if r["tipo_drift"] == "feature"]
    prediction = [r for r in righe if r["tipo_drift"] == "prediction"]
    selezione = feature + prediction
    etichette = [f"{r['tipo_drift']}\n{r['strategia']}" for r in selezione]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    posizioni = range(len(selezione))
    larghezza = 0.38
    ax.bar([p - larghezza / 2 for p in posizioni],
           [max(r["passi_in_drift"], 0.5) for r in selezione],
           larghezza, label="Passi in stato di drift (conteggio precedente)",
           color="#c44e52", edgecolor="white")
    ax.bar([p + larghezza / 2 for p in posizioni],
           [max(r["episodi"], 0.5) for r in selezione],
           larghezza, label="Episodi (conteggio corretto)",
           color="#4c72b0", edgecolor="white")

    ax.set_yscale("log")
    ax.set_xticks(list(posizioni))
    ax.set_xticklabels(etichette, fontsize=8)
    ax.set_ylabel("Conteggio (scala logaritmica)")
    ax.set_title("ELEC2: effetto del conteggio a episodi sui detector level-triggered")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()

    for estensione in ("pdf", "png"):
        fig.savefig(CARTELLA_FIGURE / f"elec2_passi_vs_episodi.{estensione}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  {CARTELLA_FIGURE / 'elec2_passi_vs_episodi'}.pdf / .png")


def salva(righe, righe_feature, dati):
    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    with open(CARTELLA_RISULTATI / "episodi_per_detector.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        writer.writeheader()
        writer.writerows(righe)

    with open(CARTELLA_RISULTATI / "episodi_per_feature.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(righe_feature[0].keys()))
        writer.writeheader()
        writer.writerows(righe_feature)

    accuracy = [a for _, a in dati["curva_accuracy"]]
    with open(CARTELLA_RISULTATI / "riepilogo.json", "w", encoding="utf-8") as f:
        json.dump({
            "generato_il": datetime.now().isoformat(timespec="seconds"),
            "dataset": DATASET_ELEC2_PATH,
            "n_campioni_totali": dati["n_campioni"],
            "n_campioni_streaming": dati["n_streaming"],
            "train_size": ELEC2_TRAIN_SIZE,
            "inizio_valutazione": INIZIO_VALUTAZIONE,
            "accuracy_iniziale": accuracy[0] if accuracy else None,
            "accuracy_finale": accuracy[-1] if accuracy else None,
            "accuracy_minima": min(accuracy) if accuracy else None,
            "accuracy_media": sum(accuracy) / len(accuracy) if accuracy else None,
            "n_warning_ddm": int(len(dati["warning_ddm"])),
            "anomalie_qualita_dataset": dati.get("anomalie_qualita", []),
            "detector": righe,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRisultati in {CARTELLA_RISULTATI}/")


def main():
    print("=" * 78)
    print("ELEC2 — rianalisi con conteggio a episodi")
    print("=" * 78)

    X, _, nomi_feature = carica_dataset()
    anomalie = controlla_qualita_dataset(X, nomi_feature)
    if anomalie:
        print()
        print("-" * 78)
        print("QUALITA' DEL DATASET — feature costanti per un tratto iniziale")
        print("-" * 78)
        for a in anomalie:
            print(f"  {a['feature']:<11} costante fino al passo {a['fine_regione_costante']:>6} "
                  f"({a['quota_costante']:>4.1f}% del dataset), valore {a['valore_costante']}"
                  f"  ->  poi media={a['media_dopo']} std={a['std_dopo']}")
        print("  Le segnalazioni di drift su queste feature nel punto di transizione")
        print("  sono artefatti di imputazione, non drift del fenomeno osservato.")

    dati = esegui()
    righe, righe_feature = analizza(dati)
    dati["anomalie_qualita"] = anomalie

    print()
    print("-" * 78)
    print(f"{'tipo':<11}{'strategia':<20}{'passi':>9}{'%':>7}{'episodi':>9}{'durata':>9}  nota")
    print("-" * 78)
    for r in righe:
        nota = "PERMANENTE dall'inizio" if r["gia_in_drift_allinizio"] else ""
        print(f"{r['tipo_drift']:<11}{r['strategia']:<20}"
              f"{r['passi_in_drift']:>9}{r['percentuale_passi']:>7.1f}"
              f"{r['episodi']:>9}{r['durata_media_episodio']:>9.0f}  {nota}")

    accuracy = [a for _, a in dati["curva_accuracy"]]
    print()
    print(f"Accuracy: iniziale={accuracy[0]:.3f}  minima={min(accuracy):.3f}  "
          f"finale={accuracy[-1]:.3f}  media={sum(accuracy)/len(accuracy):.3f}")
    print(f"Passi in stato di warning DDM: {len(dati['warning_ddm'])}")

    print("\nFigure:")
    figura_accuracy(dati)
    figura_confronto_conteggio(righe)
    salva(righe, righe_feature, dati)


if __name__ == "__main__":
    main()
