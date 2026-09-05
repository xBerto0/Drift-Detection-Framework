"""Registro delle strategie: dal nome nella configurazione alla classe.

E' il punto in cui il file di configurazione, che contiene solo stringhe,
incontra il codice. Aggiungere una strategia al framework significa
implementarne la classe e aggiungere una riga qui: nessun altro file cambia.

Il modulo si occupa anche di dire QUALI strategie sono applicabili a QUALE tipo
di drift. Non e' una restrizione arbitraria: dipende dalla natura dello stream
che ciascun tipo di drift produce.
"""

from detectors.adwin_strategy import ADWINStrategy
from detectors.ddm_strategy import DDMStrategy
from detectors.ks_strategy import KSStrategy
from detectors.page_hinkley_strategy import PageHinkleyStrategy


# Nome usato nella configurazione -> classe che lo implementa.
STRATEGIE = {
    "ks": KSStrategy,
    "adwin": ADWINStrategy,
    "ddm": DDMStrategy,
    "page_hinkley": PageHinkleyStrategy,
}


# Tipi di drift riconosciuti e cosa serve per calcolarli.
#
#   data_drift        cambia la distribuzione delle feature in ingresso, P(X).
#                     Non servono ne' il modello ne' le etichette vere.
#   prediction_drift  cambia la distribuzione delle predizioni del modello.
#                     Serve il modello, non servono le etichette vere.
#   concept_drift     cambia la relazione fra input e output, P(y|X).
#                     Servono il modello E le etichette vere: senza sapere
#                     quale fosse la risposta giusta non e' calcolabile.
TIPI_DRIFT = ("data_drift", "prediction_drift", "concept_drift")

REQUISITI = {
    "data_drift": {"modello": False, "etichette": False},
    "prediction_drift": {"modello": True, "etichette": False},
    "concept_drift": {"modello": True, "etichette": True},
}


# Quali strategie hanno senso su quale tipo di drift.
#
# DDM lavora su uno stream binario di ERRORI e presuppone il modello di
# Bernoulli: e' quindi applicabile al solo concept drift, e solo quando il
# modello monitorato e' un classificatore. Sulle feature in ingresso o sulle
# predizioni non produrrebbe risultati interpretabili, anche quando i valori
# fossero per caso binari.
STRATEGIE_AMMESSE = {
    "data_drift": ("ks", "adwin", "page_hinkley"),
    "prediction_drift": ("ks", "adwin", "page_hinkley"),
    "concept_drift": ("ks", "adwin", "page_hinkley", "ddm"),
}


# Cosa rappresenta il numero restituito in `score` da ciascuna strategia.
#
# Nessuna delle strategie implementate produce una "probabilita' che ci sia
# drift": non esiste, in questi algoritmi, una quantita' con quel significato.
# Il p-value del KS in particolare NON e' la probabilita' che il drift ci sia,
# ma la probabilita' di osservare una differenza cosi' grande SE il drift non
# ci fosse. Etichettare il campo evita che il numero venga letto per quello che
# non e'.
TIPO_SCORE = {
    "ks": "p_value",
    "adwin": "stima_media",
    "ddm": None,
    "page_hinkley": None,
}


def risolvi(nome):
    """Restituisce la classe corrispondente al nome usato in configurazione."""
    if nome not in STRATEGIE:
        raise ValueError(
            f"Strategia '{nome}' sconosciuta. "
            f"Disponibili: {', '.join(sorted(STRATEGIE))}"
        )
    return STRATEGIE[nome]


def verifica_applicabilita(tipo_drift, nome_strategia):
    """Controlla che la strategia abbia senso per quel tipo di drift."""
    if tipo_drift not in TIPI_DRIFT:
        raise ValueError(
            f"Tipo di drift '{tipo_drift}' sconosciuto. "
            f"Ammessi: {', '.join(TIPI_DRIFT)}"
        )
    ammesse = STRATEGIE_AMMESSE[tipo_drift]
    if nome_strategia not in ammesse:
        raise ValueError(
            f"La strategia '{nome_strategia}' non e' applicabile al "
            f"'{tipo_drift}'. Per questo tipo di drift sono ammesse: "
            f"{', '.join(ammesse)}."
        )
