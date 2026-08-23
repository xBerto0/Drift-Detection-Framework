"""Genera i dataset sintetici di regressione Friedman con drift e li salva su CSV.

Lo scopo e' avere i dataset FISSI dentro il repository, invece di rigenerarli
al volo a ogni esperimento. In questo modo:

- gli esperimenti leggono un file, esattamente come fanno con ELEC2;
- i risultati riportati in tesi sono riproducibili anche se in futuro cambia
  la versione di river;
- i punti di drift reali sono noti e documentati (ground truth), il che
  permette di calcolare latenza di rilevamento, falsi allarmi e mancate
  rilevazioni.

Il generatore di riferimento e' `river.datasets.synth.FriedmanDrift`, che
implementa le tre varianti di drift descritte da Ikonomovska et al. (2011).

Questo script va lanciato UNA VOLTA SOLA; i CSV prodotti vengono versionati.

    python data/generate_friedman.py
"""

import json
from pathlib import Path

import pandas as pd
from river.datasets import synth


# --- Parametri di generazione (fissi, per riproducibilita') ---
N_SAMPLES = 20000
SEED = 42

# Le tre varianti di drift previste dal generatore. Per ognuna indichiamo i
# punti in cui il concetto cambia: sono la ground truth con cui confronteremo
# le segnalazioni dei detector.
VARIANTS = {
    # Local Expanding Abrupt: il drift colpisce due regioni dello spazio delle
    # feature e a ogni cambio le regioni interessate si allargano.
    # Richiede esattamente tre punti di cambio.
    "lea": {
        "drift_type": "lea",
        "position": (5000, 10000, 15000),
        "transition_window": 0,
        "descrizione": "Local Expanding Abrupt drift (tre cambi bruschi locali)",
    },
    # Global Recurring Abrupt: il drift colpisce tutto lo spazio delle feature;
    # al secondo punto di cambio il concetto originale si ripresenta.
    # E' il caso di drift RICORRENTE.
    "gra": {
        "drift_type": "gra",
        "position": (7000, 14000),
        "transition_window": 0,
        "descrizione": "Global Recurring Abrupt drift (il concetto iniziale ritorna)",
    },
    # Global and Slow Gradual: il drift colpisce tutto lo spazio ma la
    # transizione e' graduale. Durante la finestra di transizione i campioni
    # provengono dal vecchio e dal nuovo concetto con uguale probabilita'.
    "gsg": {
        "drift_type": "gsg",
        "position": (7000, 14000),
        "transition_window": 1000,
        "descrizione": "Global Slow Gradual drift (transizione lenta di 1000 campioni)",
    },
}

OUTPUT_DIR = Path("data")


def genera_variante(nome, config):
    """Genera un singolo dataset e lo salva in CSV. Restituisce i metadati."""
    dataset = synth.FriedmanDrift(
        drift_type=config["drift_type"],
        position=config["position"],
        transition_window=config["transition_window"],
        seed=SEED,
    )

    righe = []
    for x, y in dataset.take(N_SAMPLES):
        riga = dict(x)
        riga["y"] = y
        righe.append(riga)

    df = pd.DataFrame(righe)
    # river nomina le feature con interi 0..9: le rinominiamo x0..x9 per
    # avere intestazioni di colonna leggibili nel CSV.
    df.columns = [f"x{c}" if c != "y" else "y" for c in df.columns]

    percorso = OUTPUT_DIR / f"friedman_{nome}.csv"
    df.to_csv(percorso, index=False, float_format="%.6f")

    dimensione_mb = percorso.stat().st_size / (1024 * 1024)
    print(f"  {percorso}  ({len(df)} righe, {dimensione_mb:.1f} MB)")

    return {
        "file": percorso.name,
        "drift_type": config["drift_type"],
        "descrizione": config["descrizione"],
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "drift_positions": list(config["position"]),
        "transition_window": config["transition_window"],
        "n_features": 10,
        "feature_rilevanti": ["x0", "x1", "x2", "x3", "x4"],
        "target": "y",
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Generazione dei dataset Friedman ({N_SAMPLES} campioni, seed={SEED}):")

    metadati = {}
    for nome, config in VARIANTS.items():
        metadati[nome] = genera_variante(nome, config)

    percorso_meta = OUTPUT_DIR / "friedman_metadata.json"
    with open(percorso_meta, "w", encoding="utf-8") as f:
        json.dump(metadati, f, indent=2, ensure_ascii=False)

    print(f"\nMetadati (ground truth dei drift) salvati in {percorso_meta}")


if __name__ == "__main__":
    main()
