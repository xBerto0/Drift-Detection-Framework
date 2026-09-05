"""Test del servizio di monitoring.

Coprono soprattutto la VALIDAZIONE della configurazione, perche' e' il punto in
cui un utente sbaglia piu' facilmente e in cui un messaggio d'errore chiaro
vale piu' di qualunque documentazione.

Lancio dalla radice del progetto:
    pytest tests/ -v
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB

from monitoring.drift_monitoring_service import (
    DriftMonitoringService, ErroreConfigurazione,
)
from monitoring.registry import risolvi, verifica_applicabilita


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

def test_registro_risolve_i_nomi_noti():
    for nome in ("ks", "adwin", "ddm", "page_hinkley"):
        assert risolvi(nome) is not None


def test_registro_rifiuta_nome_sconosciuto():
    with pytest.raises(ValueError):
        risolvi("algoritmo_inventato")


def test_ddm_non_applicabile_al_data_drift():
    # DDM lavora su uno stream binario di errori: sulle feature in ingresso
    # non produrrebbe nulla di interpretabile.
    with pytest.raises(ValueError):
        verifica_applicabilita("data_drift", "ddm")


def test_ddm_applicabile_al_concept_drift():
    verifica_applicabilita("concept_drift", "ddm")


# ---------------------------------------------------------------------------
# Validazione della configurazione
# ---------------------------------------------------------------------------

def config_minima(**modifiche):
    base = {
        "input": {"reference": "a.csv", "current": "b.csv"},
        "drift_da_rilevare": {"data_drift": ["ks"]},
    }
    base.update(modifiche)
    return base


def test_manca_input():
    with pytest.raises(ErroreConfigurazione):
        DriftMonitoringService({"drift_da_rilevare": {"data_drift": ["ks"]}})


def test_manca_current():
    with pytest.raises(ErroreConfigurazione):
        DriftMonitoringService(config_minima(input={"reference": "a.csv"}))


def test_tipo_di_drift_sconosciuto():
    with pytest.raises(ErroreConfigurazione):
        DriftMonitoringService(config_minima(
            drift_da_rilevare={"drift_inventato": ["ks"]},
        ))


def test_lista_strategie_vuota():
    with pytest.raises(ErroreConfigurazione):
        DriftMonitoringService(config_minima(drift_da_rilevare={"data_drift": []}))


def test_concept_drift_senza_etichette_e_rifiutato():
    # E' il vincolo piu' importante: il concept drift senza etichette vere non
    # e' calcolabile, e il servizio deve dirlo invece di inventarsi qualcosa.
    with pytest.raises(ErroreConfigurazione) as e:
        DriftMonitoringService({
            "input": {"reference": "a.csv", "current": "b.csv",
                      "model_path": "m.pkl"},
            "drift_da_rilevare": {"concept_drift": ["ddm"]},
        })
    assert "etichette" in str(e.value).lower()


def test_prediction_drift_senza_modello_e_rifiutato():
    with pytest.raises(ErroreConfigurazione) as e:
        DriftMonitoringService({
            "input": {"reference": "a.csv", "current": "b.csv"},
            "drift_da_rilevare": {"prediction_drift": ["adwin"]},
        })
    assert "model_path" in str(e.value)


def test_regola_di_aggregazione_sconosciuta():
    with pytest.raises(ErroreConfigurazione):
        DriftMonitoringService(config_minima(aggregazione="xor"))


def test_configurazione_valida_non_solleva():
    DriftMonitoringService(config_minima())


# ---------------------------------------------------------------------------
# Esecuzione completa su dati costruiti a tavolino
# ---------------------------------------------------------------------------

@pytest.fixture
def scenario(tmp_path):
    """Crea due file e un modello: riferimento pulito, correnti con drift netto."""
    rng = np.random.default_rng(7)

    def crea(nome, media, n=800):
        X = rng.normal(media, 1.0, size=(n, 3))
        y = (X[:, 0] > media).astype(int)
        df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
        df["y"] = y
        df.to_csv(tmp_path / nome, index=False)
        return df

    rif = crea("rif.csv", media=0.0)
    crea("cur.csv", media=4.0)   # spostamento molto marcato

    modello = GaussianNB()
    modello.fit(rif[["f0", "f1", "f2"]].values, rif["y"].values)
    with open(tmp_path / "modello.pkl", "wb") as f:
        pickle.dump(modello, f)

    configurazione = {
        "input": {
            "reference": "rif.csv", "current": "cur.csv",
            "target_column": "y", "model_path": "modello.pkl",
        },
        "drift_da_rilevare": {
            "data_drift": ["ks", "adwin"],
            "prediction_drift": ["adwin"],
            "concept_drift": ["ddm", "page_hinkley"],
        },
        "parametri": {"ks": {"window_size": 100, "alpha": 0.05},
                      "ddm": {"warm_start": 100}},
        "aggregazione": "or",
    }
    return configurazione, tmp_path


def test_esecuzione_rileva_drift_marcato(scenario):
    configurazione, radice = scenario
    esito = DriftMonitoringService(configurazione, radice=radice).esegui()

    assert set(esito) == {"data_drift", "prediction_drift", "concept_drift"}
    assert esito["data_drift"]["drift_detected"] is True


def test_struttura_dell_output(scenario):
    configurazione, radice = scenario
    esito = DriftMonitoringService(configurazione, radice=radice).esegui()

    voce = esito["data_drift"]["per_strategia"]["ks"]
    for campo in ("drift", "score", "tipo_score", "primo_rilevamento",
                  "frazione_in_allarme", "n_episodi"):
        assert campo in voce
    assert voce["drift"] in (0, 1)
    # Il p-value del KS va etichettato per quello che e': non e' la
    # probabilita' che ci sia drift.
    if voce["score"] is not None:
        assert voce["tipo_score"] == "p_value"


def test_output_serializzabile_in_json(scenario):
    # L'esito deve poter essere scritto su file senza conversioni manuali:
    # niente tipi numpy che sfuggono da scipy o river.
    configurazione, radice = scenario
    esito = DriftMonitoringService(configurazione, radice=radice).esegui()
    json.dumps(esito)


def test_colonna_esclusa_non_viene_monitorata(scenario):
    configurazione, radice = scenario
    configurazione["input"]["exclude_columns"] = ["f2"]
    esito = DriftMonitoringService(configurazione, radice=radice).esegui()
    coinvolte = esito["data_drift"]["per_strategia"]["ks"]["feature_coinvolte"]
    assert "f2" not in coinvolte


def test_colonna_esclusa_inesistente_e_rifiutata(scenario):
    configurazione, radice = scenario
    configurazione["input"]["exclude_columns"] = ["colonna_che_non_esiste"]
    with pytest.raises(ErroreConfigurazione):
        DriftMonitoringService(configurazione, radice=radice).esegui()


def test_aggregazione_and_e_piu_severa_di_or(scenario):
    configurazione, radice = scenario
    configurazione["drift_da_rilevare"] = {"concept_drift": ["ddm", "page_hinkley"]}

    configurazione["aggregazione"] = "or"
    con_or = DriftMonitoringService(configurazione, radice=radice).esegui()
    configurazione["aggregazione"] = "and"
    con_and = DriftMonitoringService(configurazione, radice=radice).esegui()

    if con_and["concept_drift"]["drift_detected"]:
        assert con_or["concept_drift"]["drift_detected"]
