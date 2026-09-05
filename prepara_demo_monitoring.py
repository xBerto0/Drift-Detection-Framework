"""Prepara i dati e il modello per la demo del servizio di monitoring.

Costruisce lo scenario realistico che il servizio si aspetta:

- un modello addestrato su dati storici e salvato su disco;
- il file dei dati di RIFERIMENTO, cioe' quelli su cui il modello e' stato
  addestrato;
- il file dei dati CORRENTI, cioe' quelli arrivati dopo il rilascio.

Come dati si usa ELEC2. Il riferimento sono i primi 5.000 campioni, i dati
correnti sono 5.000 campioni presi molto piu' avanti nel tempo: sappiamo dalla
validazione sperimentale che in quel tratto il classificatore si e' degradato,
quindi ci aspettiamo che il drift venga rilevato.

Va lanciato una volta sola, prima della demo:

    python prepara_demo_monitoring.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from config import DATASET_ELEC2_PATH

CARTELLA_DEMO = Path("data/demo")
CARTELLA_MODELLI = Path("models/addestrati")

# Il riferimento e' il periodo su cui il modello viene addestrato.
INIZIO_RIFERIMENTO, FINE_RIFERIMENTO = 0, 5000
# I dati correnti arrivano molto piu' avanti: e' il periodo in cui, dagli
# esperimenti, l'accuratezza del classificatore risulta crollata.
INIZIO_CORRENTE, FINE_CORRENTE = 38000, 43000


def main():
    CARTELLA_DEMO.mkdir(parents=True, exist_ok=True)
    CARTELLA_MODELLI.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_ELEC2_PATH)
    # La classe e' testuale (UP / DOWN): la si porta a 0/1.
    df["class"] = np.where(df["class"].values == "UP", 1, 0)

    riferimento = df.iloc[INIZIO_RIFERIMENTO:FINE_RIFERIMENTO].reset_index(drop=True)
    corrente = df.iloc[INIZIO_CORRENTE:FINE_CORRENTE].reset_index(drop=True)

    colonne_feature = [c for c in df.columns if c != "class"]

    # --- Modello addestrato SOLO sul riferimento ---
    modello = GaussianNB()
    modello.fit(riferimento[colonne_feature].values, riferimento["class"].values)

    accuratezza_rif = modello.score(
        riferimento[colonne_feature].values, riferimento["class"].values,
    )
    accuratezza_cur = modello.score(
        corrente[colonne_feature].values, corrente["class"].values,
    )

    percorso_modello = CARTELLA_MODELLI / "classificatore_elec2.pkl"
    with open(percorso_modello, "wb") as f:
        pickle.dump(modello, f)

    riferimento.to_csv(CARTELLA_DEMO / "riferimento.csv", index=False)
    corrente.to_csv(CARTELLA_DEMO / "corrente.csv", index=False)

    print("Scenario della demo pronto.")
    print()
    print(f"  Riferimento : campioni {INIZIO_RIFERIMENTO}-{FINE_RIFERIMENTO} "
          f"({len(riferimento)} righe)")
    print(f"  Correnti    : campioni {INIZIO_CORRENTE}-{FINE_CORRENTE} "
          f"({len(corrente)} righe)")
    print(f"  Modello     : GaussianNB addestrato sul solo riferimento")
    print()
    print(f"  Accuratezza sul riferimento : {accuratezza_rif:.3f}")
    print(f"  Accuratezza sui dati nuovi  : {accuratezza_cur:.3f}")
    print()
    print("  File prodotti:")
    print(f"    {CARTELLA_DEMO / 'riferimento.csv'}")
    print(f"    {CARTELLA_DEMO / 'corrente.csv'}")
    print(f"    {percorso_modello}")


if __name__ == "__main__":
    main()
