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

# Due scenari, per mostrare che il servizio sa dire anche di no.
#
#   "drift"     il modello e' addestrato sul periodo iniziale e valutato su un
#               periodo molto successivo, in cui dagli esperimenti risulta
#               fortemente degradato.
#   "stabile"   riferimento e dati correnti provengono dallo stesso periodo, a
#               due meta' consecutive: il modello lavora nelle condizioni per
#               cui e' stato addestrato.
SCENARI = {
    "drift":   {"riferimento": (0, 5000),    "corrente": (38000, 43000)},
    "stabile": {"riferimento": (0, 2500),    "corrente": (2500, 5000)},
}


def prepara(nome_scenario, df, colonne_feature):
    intervalli = SCENARI[nome_scenario]
    a, b = intervalli["riferimento"]
    c, d = intervalli["corrente"]

    riferimento = df.iloc[a:b].reset_index(drop=True)
    corrente = df.iloc[c:d].reset_index(drop=True)

    # Il modello vede SOLO il riferimento, come nella realta'.
    modello = GaussianNB()
    modello.fit(riferimento[colonne_feature].values, riferimento["class"].values)

    accuratezza_rif = modello.score(
        riferimento[colonne_feature].values, riferimento["class"].values,
    )
    accuratezza_cur = modello.score(
        corrente[colonne_feature].values, corrente["class"].values,
    )

    percorso_modello = CARTELLA_MODELLI / f"classificatore_{nome_scenario}.pkl"
    with open(percorso_modello, "wb") as f:
        pickle.dump(modello, f)

    riferimento.to_csv(CARTELLA_DEMO / f"{nome_scenario}_riferimento.csv", index=False)
    corrente.to_csv(CARTELLA_DEMO / f"{nome_scenario}_corrente.csv", index=False)

    print(f"  scenario '{nome_scenario}':")
    print(f"     riferimento campioni {a}-{b}, correnti {c}-{d}")
    print(f"     accuratezza sul riferimento {accuratezza_rif:.3f}  ->  "
          f"sui dati nuovi {accuratezza_cur:.3f}")


def main():
    CARTELLA_DEMO.mkdir(parents=True, exist_ok=True)
    CARTELLA_MODELLI.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_ELEC2_PATH)
    # La classe e' testuale (UP / DOWN): la si porta a 0/1.
    df["class"] = np.where(df["class"].values == "UP", 1, 0)
    colonne_feature = [c for c in df.columns if c != "class"]

    print("Scenari della demo:")
    print()
    for nome in SCENARI:
        prepara(nome, df, colonne_feature)
    print()
    print(f"File in {CARTELLA_DEMO}/ e {CARTELLA_MODELLI}/")


if __name__ == "__main__":
    main()
