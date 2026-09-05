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
from evaluation.drift_metrics import estrai_eventi
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

        # Attenzione a non confondere due cose diverse:
        #
        #   colonne_feature    tutte le feature che il MODELLO si aspetta in
        #                      ingresso. Sono fissate al momento
        #                      dell'addestramento e non si possono togliere.
        #   colonne_monitorate quelle su cui si vuole cercare il data drift.
        #                      Qui ha senso escludere le colonne che producono
        #                      un drift strutturale garantito ma privo di
        #                      significato, tipicamente i timestamp, che
        #                      crescono in modo monotono per costruzione.
        colonne_feature = [c for c in riferimento.columns if c != colonna_target]
        da_escludere = set(ingresso.get("exclude_columns", []))
        sconosciute = da_escludere - set(colonne_feature)
        if sconosciute:
            raise ErroreConfigurazione(
                f"Le colonne da escludere {sorted(sconosciute)} non esistono "
                f"nei dati. Disponibili: {colonne_feature}."
            )
        colonne_monitorate = [c for c in colonne_feature if c not in da_escludere]
        if not colonne_monitorate:
            raise ErroreConfigurazione(
                "Dopo le esclusioni non resta alcuna feature da monitorare."
            )
        mancanti = set(colonne_feature) - set(corrente.columns)
        if mancanti:
            raise ErroreConfigurazione(
                f"Le colonne {sorted(mancanti)} sono presenti nel riferimento "
                f"ma non nei dati correnti."
            )

        return (riferimento, corrente, colonne_feature,
                colonne_monitorate, colonna_target)

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
        (riferimento, corrente, colonne_feature,
         colonne_monitorate, colonna_target) = self._carica_dati()
        modello = self._carica_modello()

        # Al modello vanno TUTTE le feature su cui e' stato addestrato.
        X_rif = riferimento[colonne_feature].values.astype(float)
        X_cur = corrente[colonne_feature].values.astype(float)

        # Al data drift solo quelle che si vuole monitorare.
        M_rif = riferimento[colonne_monitorate].values.astype(float)
        M_cur = corrente[colonne_monitorate].values.astype(float)

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
                    nomi_strategie, M_rif, M_cur, colonne_monitorate,
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

            primo = None
            score = None
            n_allarmi = 0
            feature_coinvolte = set()
            verdetti_nel_tempo = []
            for i, riga in enumerate(X_cur):
                detector.update(riga)
                risultato = detector.detect()
                verdetti_nel_tempo.append(risultato.drift_detected)
                if risultato.drift_detected:
                    n_allarmi += 1
                    if primo is None:
                        primo = i
                        score = self._score_piu_estremo(
                            nome, risultato.metadata.get("punteggi"),
                        )
                    feature_coinvolte.update(risultato.metadata["drifted_features"])

            frazione = n_allarmi / len(X_cur) if len(X_cur) else 0.0
            voce = self._verdetto(nome, frazione, primo, score,
                                  n_eventi=len(estrai_eventi(verdetti_nel_tempo)))
            voce["feature_coinvolte"] = sorted(feature_coinvolte)
            per_strategia[nome] = voce

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
        primo = None
        n_allarmi = 0
        score_finale = None
        score_al_rilevamento = None
        totale = 0
        verdetti_nel_tempo = []

        for i, elemento in enumerate(elementi):
            totale += 1
            aggiorna(elemento)
            risultato = verifica()
            score_finale = risultato.score
            verdetti_nel_tempo.append(risultato.drift_detected)
            if risultato.drift_detected:
                n_allarmi += 1
                if primo is None:
                    primo = i
                    score_al_rilevamento = risultato.score

        frazione = n_allarmi / totale if totale else 0.0
        return self._verdetto(
            nome, frazione, primo,
            score_al_rilevamento if primo is not None else score_finale,
            n_eventi=len(estrai_eventi(verdetti_nel_tempo)),
        )

    def _verdetto(self, nome, frazione, primo, score, n_eventi=None):
        """Trasforma la frazione di tempo in allarme in un verdetto binario.

        Il criterio ingenuo "ha segnalato almeno una volta nel lotto" e' troppo
        permissivo: su un blocco di migliaia di campioni un detector sequenziale
        ha migliaia di occasioni per scattare, e su dati reali finisce quasi
        sempre per farlo. E' lo stesso problema di molteplicita' che si presenta
        sulle feature, ma lungo l'asse del tempo.

        Si riporta quindi anche la FRAZIONE di lotto trascorsa in allarme, e il
        verdetto binario si ottiene confrontandola con una soglia
        configurabile. Con soglia 0 si ricade nel comportamento "basta una
        segnalazione".
        """
        soglia = float(self.config.get("soglia_frazione_allarme", 0.0))
        rilevato = frazione > soglia if soglia > 0 else primo is not None
        return {
            "drift": int(rilevato),
            "frazione_in_allarme": round(frazione, 4),
            "n_episodi": n_eventi,
            # Se il drift e' stato rilevato si riporta il valore in quel
            # momento, altrimenti l'ultimo osservato.
            "score": score,
            "tipo_score": TIPO_SCORE.get(nome),
            "primo_rilevamento": primo,
        }

    @staticmethod
    def _score_piu_estremo(nome, punteggi):
        """Riassume in un solo numero i punteggi delle singole feature.

        Con piu' feature non esiste un punteggio unico: se ne riporta quello
        piu' estremo, cioe' l'evidenza piu' forte trovata. Per il KS, che
        produce p-value, il piu' estremo e' il piu' piccolo. Per le strategie
        che non producono un punteggio confrontabile si restituisce None.
        """
        if not punteggi:
            return None
        validi = [p for p in punteggi if p is not None]
        if not validi:
            return None
        if nome == "ks":
            return min(validi)
        return None

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
