"""Servizio di monitoring: orchestra il rilevamento del drift a partire da una configurazione.

E' il livello 3 dell'architettura, quello che finora mancava. Collega il file di
configurazione, i dati, il modello monitorato e i detector, e restituisce un
verdetto strutturato.

Modalita' di funzionamento: A LOTTI
-----------------------------------
Il servizio riceve due insiemi di dati:

- il RIFERIMENTO, cioe' i dati su cui il modello monitorato e' stato addestrato,
  che descrivono il comportamento considerato normale;
- i dati CORRENTI, cioe' quelli arrivati dopo il rilascio del modello.

Fa scorrere entrambi nei detector — prima il riferimento, per fissare la
baseline, poi i dati correnti — e raccoglie i verdetti solo sulla seconda
parte. In uscita produce, per ogni tipo di drift richiesto, un verdetto binario
piu' il dettaglio per singola strategia.

Nota: la modalita' a lotti risponde alla domanda "in questo blocco di dati c'e'
drift?", non alla domanda "quando e' cominciato". Il motore sottostante resta a
flusso, quindi l'istante di primo rilevamento e' comunque riportato, ma le
metriche di latenza hanno senso solo nella valutazione sperimentale.

Esempio d'uso
-------------
    from monitoring.drift_monitoring_service import DriftMonitoringService

    servizio = DriftMonitoringService.da_file("config/drift_config.json")
    esito = servizio.esegui()
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from detectors.concept_drift_detector import (
    ConceptDriftDetector, errore_assoluto, errore_binario,
)
from detectors.feature_drift_detector import FeatureDriftDetector
from detectors.prediction_drift_detector import PredictionDriftDetector
from monitoring.registry import (
    REQUISITI, STRATEGIE_AMMESSE, TIPI_DRIFT, TIPO_SCORE,
    risolvi, verifica_applicabilita,
)

REGOLE_AGGREGAZIONE = ("or", "and", "majority")


class ErroreConfigurazione(Exception):
    """La configurazione fornita non e' utilizzabile."""


class DriftMonitoringService:
    """Esegue il rilevamento del drift secondo una configurazione dichiarativa."""

    def __init__(self, configurazione: dict, radice: Path = None):
        self.config = configurazione
        # I percorsi nella configurazione sono relativi alla radice del
        # progetto, non alla posizione del file di configurazione.
        self.radice = Path(radice) if radice else Path(".")
        self._valida()

    @classmethod
    def da_file(cls, percorso, radice: Path = None):
        percorso = Path(percorso)
        with open(percorso, encoding="utf-8") as f:
            configurazione = json.load(f)
        return cls(configurazione, radice=radice)

    # -- validazione ------------------------------------------------------

    def _valida(self):
        """Controlla la configurazione PRIMA di leggere i dati.

        Fallire subito con un messaggio chiaro e' molto meglio che accorgersi a
        meta' elaborazione che mancava un campo.
        """
        if "input" not in self.config:
            raise ErroreConfigurazione("Manca la sezione 'input'.")
        if "drift_da_rilevare" not in self.config:
            raise ErroreConfigurazione("Manca la sezione 'drift_da_rilevare'.")

        ingresso = self.config["input"]
        for campo in ("reference", "current"):
            if campo not in ingresso:
                raise ErroreConfigurazione(
                    f"Manca 'input.{campo}': servono sia i dati di riferimento "
                    f"sia quelli correnti."
                )

        richiesti = self.config["drift_da_rilevare"]
        if not richiesti:
            raise ErroreConfigurazione(
                "'drift_da_rilevare' e' vuoto: non c'e' niente da fare."
            )

        for tipo, strategie in richiesti.items():
            if tipo not in TIPI_DRIFT:
                raise ErroreConfigurazione(
                    f"Tipo di drift '{tipo}' sconosciuto. "
                    f"Ammessi: {', '.join(TIPI_DRIFT)}."
                )
            if not strategie:
                raise ErroreConfigurazione(
                    f"Nessuna strategia indicata per '{tipo}'. "
                    f"Ammesse: {', '.join(STRATEGIE_AMMESSE[tipo])}."
                )
            for nome in strategie:
                verifica_applicabilita(tipo, nome)

            # Controlla che ci sia tutto il necessario per quel tipo di drift.
            requisiti = REQUISITI[tipo]
            if requisiti["modello"] and not ingresso.get("model_path"):
                raise ErroreConfigurazione(
                    f"Il '{tipo}' richiede il modello monitorato, ma "
                    f"'input.model_path' non e' indicato."
                )
            if requisiti["etichette"] and not ingresso.get("target_column"):
                raise ErroreConfigurazione(
                    f"Il '{tipo}' richiede le etichette vere, ma "
                    f"'input.target_column' non e' indicato. Senza sapere quale "
                    f"fosse la risposta corretta il concept drift non e' "
                    f"calcolabile."
                )

        regola = self.config.get("aggregazione", "or").lower()
        if regola not in REGOLE_AGGREGAZIONE:
            raise ErroreConfigurazione(
                f"Regola di aggregazione '{regola}' non supportata. "
                f"Ammesse: {', '.join(REGOLE_AGGREGAZIONE)}."
            )

    # -- caricamento ------------------------------------------------------

    def _carica_dati(self):
        ingresso = self.config["input"]
        riferimento = pd.read_csv(self.radice / ingresso["reference"])
        corrente = pd.read_csv(self.radice / ingresso["current"])

        colonna_target = ingresso.get("target_column")
        if colonna_target:
            for nome, df in (("reference", riferimento), ("current", corrente)):
                if colonna_target not in df.columns:
                    raise ErroreConfigurazione(
                        f"La colonna target '{colonna_target}' non esiste in "
                        f"'{nome}'. Colonne disponibili: {list(df.columns)}."
                    )

        colonne_feature = [c for c in riferimento.columns if c != colonna_target]
        mancanti = set(colonne_feature) - set(corrente.columns)
        if mancanti:
            raise ErroreConfigurazione(
                f"Le colonne {sorted(mancanti)} sono presenti nel riferimento "
                f"ma non nei dati correnti."
            )

        return riferimento, corrente, colonne_feature, colonna_target

    def _carica_modello(self):
        percorso = self.config["input"].get("model_path")
        if not percorso:
            return None
        percorso = self.radice / percorso
        if not percorso.exists():
            raise ErroreConfigurazione(f"Modello non trovato: {percorso}")
        with open(percorso, "rb") as f:
            return pickle.load(f)

    def _parametri(self, nome_strategia):
        return dict(self.config.get("parametri", {}).get(nome_strategia, {}))

    # -- esecuzione -------------------------------------------------------

    def esegui(self) -> dict:
        """Esegue tutti i rilevamenti richiesti e restituisce l'esito."""
        riferimento, corrente, colonne_feature, colonna_target = self._carica_dati()
        modello = self._carica_modello()

        X_rif = riferimento[colonne_feature].values.astype(float)
        X_cur = corrente[colonne_feature].values.astype(float)

        y_rif = y_cur = None
        if colonna_target:
            y_rif = riferimento[colonna_target].values
            y_cur = corrente[colonna_target].values

        pred_rif = pred_cur = None
        if modello is not None:
            pred_rif = np.asarray(modello.predict(X_rif))
            pred_cur = np.asarray(modello.predict(X_cur))

        esito = {}
        for tipo, nomi_strategie in self.config["drift_da_rilevare"].items():
            if tipo == "data_drift":
                esito[tipo] = self._data_drift(
                    nomi_strategie, X_rif, X_cur, colonne_feature,
                )
            elif tipo == "prediction_drift":
                esito[tipo] = self._stream_singolo(
                    tipo, nomi_strategie, pred_rif, pred_cur,
                )
            elif tipo == "concept_drift":
                esito[tipo] = self._concept_drift(
                    nomi_strategie, pred_rif, y_rif, pred_cur, y_cur,
                )

        return esito

    def _data_drift(self, nomi_strategie, X_rif, X_cur, colonne_feature):
        """Una strategia indipendente per feature, poi aggregazione."""
        correzione = self.config.get("correzione_test_multipli")
        per_strategia = {}

        for nome in nomi_strategie:
            cls = risolvi(nome)
            parametri = self._parametri(nome)
            extra = {}
            if correzione:
                extra = {
                    "correzione": correzione,
                    "alpha_correzione": parametri.get("alpha", 0.05),
                }
            detector = FeatureDriftDetector(
                cls, n_features=len(colonne_feature), k=1,
                feature_names=colonne_feature, **extra, **parametri,
            )
            detector.imposta_riferimento(X_rif)

            rilevato = False
            primo = None
            feature_coinvolte = set()
            for i, riga in enumerate(X_cur):
                detector.update(riga)
                risultato = detector.detect()
                if risultato.drift_detected:
                    if not rilevato:
                        primo = i
                    rilevato = True
                    feature_coinvolte.update(risultato.metadata["drifted_features"])

            per_strategia[nome] = {
                "drift": int(rilevato),
                "score": None,
                "tipo_score": None,
                "primo_rilevamento": primo,
                "feature_coinvolte": sorted(feature_coinvolte),
            }

        return self._aggrega(per_strategia)

    def _stream_singolo(self, tipo, nomi_strategie, stream_rif, stream_cur):
        """Prediction drift: un solo stream, quello delle predizioni."""
        per_strategia = {}
        for nome in nomi_strategie:
            detector = PredictionDriftDetector(risolvi(nome), **self._parametri(nome))
            detector.imposta_riferimento(stream_rif)
            per_strategia[nome] = self._scorri(
                nome, stream_cur, detector.update, detector.detect,
            )
        return self._aggrega(per_strategia)

    def _concept_drift(self, nomi_strategie, pred_rif, y_rif, pred_cur, y_cur):
        """Concept drift: stream degli errori del modello."""
        # Il tipo di errore dipende dal problema. Se le etichette non sono
        # numeri interi assumiamo una regressione e usiamo l'errore assoluto.
        e_regressione = np.issubdtype(np.asarray(y_cur).dtype, np.floating)
        funzione_errore = errore_assoluto if e_regressione else errore_binario

        per_strategia = {}
        for nome in nomi_strategie:
            if nome == "ddm" and e_regressione:
                raise ErroreConfigurazione(
                    "DDM non e' applicabile a un modello di regressione: "
                    "presuppone un errore binario. Per la regressione usare "
                    "page_hinkley, adwin o ks."
                )
            detector = ConceptDriftDetector(
                risolvi(nome), error_fn=funzione_errore, **self._parametri(nome),
            )
            detector.imposta_riferimento(pred_rif, y_rif)
            per_strategia[nome] = self._scorri(
                nome, list(zip(pred_cur, y_cur)),
                lambda coppia: detector.update(coppia[0], coppia[1]),
                detector.detect,
            )
        return self._aggrega(per_strategia)

    def _scorri(self, nome, elementi, aggiorna, verifica):
        """Fa scorrere i dati correnti e riassume l'esito della strategia."""
        rilevato = False
        primo = None
        score_finale = None
        score_al_rilevamento = None

        for i, elemento in enumerate(elementi):
            aggiorna(elemento)
            risultato = verifica()
            score_finale = risultato.score
            if risultato.drift_detected and not rilevato:
                rilevato = True
                primo = i
                score_al_rilevamento = risultato.score

        return {
            "drift": int(rilevato),
            # Se il drift e' stato rilevato si riporta il valore in quel
            # momento, altrimenti l'ultimo osservato.
            "score": score_al_rilevamento if rilevato else score_finale,
            "tipo_score": TIPO_SCORE.get(nome),
            "primo_rilevamento": primo,
        }

    def _aggrega(self, per_strategia):
        """Applica la regola di aggregazione ai verdetti delle strategie."""
        regola = self.config.get("aggregazione", "or").lower()
        verdetti = [bool(v["drift"]) for v in per_strategia.values()]

        if regola == "or":
            complessivo = any(verdetti)
        elif regola == "and":
            complessivo = all(verdetti)
        else:
            complessivo = sum(verdetti) > len(verdetti) / 2

        return {
            "drift_detected": complessivo,
            "regola_aggregazione": regola,
            "per_strategia": per_strategia,
        }
