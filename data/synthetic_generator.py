"""Generatore di dati sintetici per testare i drift detector.

Il punto di questo modulo e' produrre stream in cui la posizione del drift e'
NOTA PER COSTRUZIONE. Senza quella conoscenza non e' possibile calcolare la
latenza di rilevamento, il tasso di falsi allarmi e il tasso di mancate
rilevazioni: si puo' solo contare quante volte un detector ha segnalato
qualcosa, che e' una descrizione, non una valutazione.

Per questo motivo tutte le funzioni introdotte qui restituiscono la coppia
`(valori, punti_drift)`: la ground truth viaggia insieme ai dati e non puo'
disallinearsi.

I quattro tipi di drift implementati corrispondono alla caratterizzazione
temporale descritta in Gama et al. (2014) e ripresa nel Capitolo 2 della tesi:

- IMPROVVISO (abrupt):     il concetto cambia da un campione all'altro
- GRADUALE (gradual):      due concetti coesistono per un periodo, con
                           probabilita' crescente del nuovo
- INCREMENTALE (incremental): il concetto si deforma con continuita'
                           attraverso stati intermedi
- RICORRENTE (recurring):  un concetto gia' visto si ripresenta

La differenza fra GRADUALE e INCREMENTALE e' sottile e viene spesso confusa,
quindi vale la pena fissarla:

- nel graduale i concetti sono DUE e distinti; durante la transizione ogni
  campione proviene o dall'uno o dall'altro, con probabilita' che si sposta
  progressivamente dal vecchio al nuovo. Non esistono stati intermedi.
- nell'incrementale il concetto e' UNO che si deforma; i campioni provengono
  da stati intermedi che non coincidono ne' con quello iniziale ne' con
  quello finale.

A questi si aggiunge lo stream STAZIONARIO, che non e' un tipo di drift ma e'
indispensabile: e' l'unico modo per misurare il tasso di falsi allarmi, cioe'
quante volte un detector segnala drift quando non c'e' nulla da segnalare.
"""

import numpy as np


def bernoulli_with_abrupt_drift(n_samples, p_before, p_after, drift_point, seed=None):
    """Genera un flusso Bernoulli con cambio brusco di p al passo drift_point.

    Replica lo scenario della Figura 2 del paper Bifet & Gavalda 2006:
    un cambiamento improvviso del parametro mu della distribuzione.

    Funzione storica, mantenuta invariata per compatibilita' con gli
    esperimenti gia' esistenti. Per il nuovo protocollo di valutazione usare
    `bernoulli_stream`, che restituisce anche la ground truth.
    """
    rng = np.random.default_rng(seed)
    before = rng.binomial(n=1, p=p_before, size=drift_point)
    after = rng.binomial(n=1, p=p_after, size=n_samples - drift_point)
    return np.concatenate([before, after]).astype(float)


# ---------------------------------------------------------------------------
# Generatori con ground truth
# ---------------------------------------------------------------------------


def bernoulli_stazionario(n_samples, p=0.5, seed=None):
    """Stream Bernoulli senza alcun drift.

    Serve a misurare i falsi allarmi: ogni segnalazione prodotta su questo
    stream e' per definizione un falso positivo.

    Restituisce (valori, punti_drift) con punti_drift vuoto.
    """
    rng = np.random.default_rng(seed)
    valori = rng.binomial(n=1, p=p, size=n_samples).astype(float)
    return valori, []


def bernoulli_improvviso(n_samples, p_before, p_after, drift_point, seed=None):
    """Drift improvviso: p passa da p_before a p_after in un solo passo."""
    rng = np.random.default_rng(seed)
    prima = rng.binomial(n=1, p=p_before, size=drift_point)
    dopo = rng.binomial(n=1, p=p_after, size=n_samples - drift_point)
    valori = np.concatenate([prima, dopo]).astype(float)
    return valori, [drift_point]


def bernoulli_graduale(n_samples, p_before, p_after, drift_point,
                       transition_window, seed=None):
    """Drift graduale: due concetti coesistono per `transition_window` passi.

    Durante la transizione ogni campione viene estratto dal vecchio concetto
    (p_before) oppure dal nuovo (p_after); la probabilita' di pescare dal
    nuovo cresce linearmente da 0 a 1 lungo la finestra.

    Il punto di drift di riferimento e' l'INIZIO della transizione: e' il
    primo istante in cui lo stream smette di essere quello vecchio.
    """
    rng = np.random.default_rng(seed)
    valori = np.zeros(n_samples)
    fine_transizione = drift_point + transition_window

    for t in range(n_samples):
        if t < drift_point:
            p = p_before
        elif t >= fine_transizione:
            p = p_after
        else:
            # Probabilita' di provenire dal nuovo concetto, da 0 a 1.
            peso_nuovo = (t - drift_point) / transition_window
            # Si sceglie DA QUALE concetto pescare, non si media fra i due:
            # e' questa la differenza rispetto al drift incrementale.
            p = p_after if rng.random() < peso_nuovo else p_before
        valori[t] = rng.binomial(n=1, p=p)

    return valori, [drift_point]


def bernoulli_incrementale(n_samples, p_before, p_after, drift_point,
                           transition_window, seed=None):
    """Drift incrementale: p si sposta con continuita' da p_before a p_after.

    A differenza del graduale, qui il concetto e' uno solo che si deforma:
    durante la transizione il parametro p assume valori INTERMEDI, che non
    corrispondono ne' al concetto iniziale ne' a quello finale.
    """
    rng = np.random.default_rng(seed)
    valori = np.zeros(n_samples)
    fine_transizione = drift_point + transition_window

    for t in range(n_samples):
        if t < drift_point:
            p = p_before
        elif t >= fine_transizione:
            p = p_after
        else:
            # Interpolazione lineare del parametro: stati intermedi reali.
            avanzamento = (t - drift_point) / transition_window
            p = p_before + (p_after - p_before) * avanzamento
        valori[t] = rng.binomial(n=1, p=p)

    return valori, [drift_point]


def bernoulli_ricorrente(n_samples, p_a, p_b, drift_points, seed=None):
    """Drift ricorrente: i concetti A e B si alternano.

    Lo stream parte dal concetto A e ad ogni punto di `drift_points` passa
    all'altro concetto. Con due punti si ottiene A -> B -> A, cioe' il
    ritorno del concetto iniziale.

    E' il caso tipico della stagionalita': un comportamento gia' osservato in
    passato si ripresenta. Un detector senza memoria lo tratta come un drift
    nuovo; e' uno degli scenari in cui la differenza fra gli algoritmi si
    vede meglio.
    """
    rng = np.random.default_rng(seed)
    valori = np.zeros(n_samples)
    confini = [0] + list(drift_points) + [n_samples]

    for i in range(len(confini) - 1):
        inizio, fine = confini[i], confini[i + 1]
        # I segmenti pari usano il concetto A, quelli dispari il concetto B.
        p = p_a if i % 2 == 0 else p_b
        valori[inizio:fine] = rng.binomial(n=1, p=p, size=fine - inizio)

    return valori, list(drift_points)


# Mappa dei tipi di drift disponibili, usata dal runner degli esperimenti per
# iterare sugli scenari senza duplicare codice.
TIPI_DRIFT = {
    "stazionario": bernoulli_stazionario,
    "improvviso": bernoulli_improvviso,
    "graduale": bernoulli_graduale,
    "incrementale": bernoulli_incrementale,
    "ricorrente": bernoulli_ricorrente,
}
