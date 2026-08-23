"""Test del modulo di metriche di valutazione.

Il modulo `evaluation/drift_metrics.py` e' il punto su cui poggia tutta la
validazione sperimentale: se sbaglia a contare, ogni numero riportato in tesi
e' sbagliato. Per questo e' la parte del framework che va testata per prima.

Lancio dalla radice del progetto:
    pytest tests/ -v
"""

from evaluation.drift_metrics import aggrega, conta_episodi, estrai_eventi, valuta


# ---------------------------------------------------------------------------
# estrai_eventi: collasso dei verdetti in eventi
# ---------------------------------------------------------------------------

def test_detector_edge_triggered_resta_invariato():
    # ADWIN, DDM e Page-Hinkley producono True isolati: il collasso in eventi
    # deve essere l'identita'.
    verdetti = [False] * 10
    verdetti[3] = True
    verdetti[7] = True
    assert estrai_eventi(verdetti) == [3, 7]


def test_detector_level_triggered_viene_collassato():
    # Il KS resta True per molti passi consecutivi: ogni sequenza contigua
    # deve contare come un solo evento, registrato sul primo passo.
    verdetti = [False] * 3 + [True] * 5 + [False] * 2 + [True] * 4
    assert estrai_eventi(verdetti) == [3, 10]


def test_warming_up_escluso_senza_falso_fronte():
    # Se il verdetto e' gia' True quando comincia la valutazione, non deve
    # essere registrato un evento spurio al primo passo utile.
    verdetti = [True] * 5 + [False] * 3 + [True] * 2
    assert estrai_eventi(verdetti, inizio=5) == [8]


def test_nessuna_segnalazione_nessun_evento():
    assert estrai_eventi([False] * 100) == []


# ---------------------------------------------------------------------------
# valuta: abbinamento con la ground truth
# ---------------------------------------------------------------------------

def test_rilevamento_corretto():
    m = valuta(eventi=[1050], punti_drift=[1000], n_campioni=2000, tolleranza=500)
    assert m.n_rilevati == 1
    assert m.latenze == [50]
    assert m.latenza_media == 50
    assert m.n_falsi_allarmi == 0
    assert m.n_mancati == 0


def test_evento_fuori_finestra_e_falso_allarme():
    # L'evento a t=300 precede il drift: non puo' riferirsi ad esso.
    m = valuta(eventi=[300, 1050], punti_drift=[1000], n_campioni=2000, tolleranza=500)
    assert m.n_falsi_allarmi == 1
    assert m.n_rilevati == 1


def test_rilevamento_oltre_tolleranza_conta_come_mancato():
    # Rilevare 900 passi dopo, con tolleranza 500, non e' un rilevamento:
    # e' una mancata rilevazione piu' un falso allarme.
    m = valuta(eventi=[1900], punti_drift=[1000], n_campioni=2000, tolleranza=500)
    assert m.n_mancati == 1
    assert m.n_falsi_allarmi == 1
    assert m.tasso_mancate == 1.0


def test_eventi_ripetuti_nella_stessa_finestra_sono_duplicati():
    # Il secondo evento si riferisce a un drift gia' coperto: non e' un
    # successo in piu' ne' un falso allarme.
    m = valuta(eventi=[1050, 1200], punti_drift=[1000], n_campioni=2000, tolleranza=500)
    assert m.n_duplicati == 1
    assert m.n_rilevati == 1
    assert m.n_falsi_allarmi == 0
    assert m.latenze == [50]


def test_stream_stazionario_ogni_evento_e_falso_allarme():
    m = valuta(eventi=[100, 500, 900], punti_drift=[], n_campioni=2000, tolleranza=500)
    assert m.n_falsi_allarmi == 3
    assert m.tasso_mancate is None          # non ci sono drift da mancare
    assert m.falsi_allarmi_per_1000 == 1.5  # 3 falsi su 2000 campioni


def test_due_drift_uno_solo_rilevato():
    m = valuta(eventi=[1100], punti_drift=[1000, 5000], n_campioni=8000, tolleranza=500)
    assert m.n_rilevati == 1
    assert m.n_mancati == 1
    assert m.tasso_mancate == 0.5


def test_nessun_evento_tutti_i_drift_mancati():
    m = valuta(eventi=[], punti_drift=[1000, 3000], n_campioni=5000, tolleranza=500)
    assert m.n_rilevati == 0
    assert m.n_mancati == 2
    assert m.tasso_mancate == 1.0
    assert m.latenza_media is None


# ---------------------------------------------------------------------------
# aggrega: sintesi su piu' ripetizioni
# ---------------------------------------------------------------------------

def test_aggregazione_su_piu_seed():
    metriche = [
        valuta([1050], [1000], 2000, 500),
        valuta([1080], [1000], 2000, 500),
        valuta([300, 1040], [1000], 2000, 500),
    ]
    a = aggrega(metriche)
    assert a["n_ripetizioni"] == 3
    assert a["latenza_media"] == (50 + 80 + 40) / 3
    assert a["latenza_std"] > 0
    # Un solo seed su tre ha prodotto un falso allarme: e' esattamente
    # l'informazione che una singola esecuzione non avrebbe mostrato.
    assert a["seed_con_falsi_allarmi"] == 1
    assert a["n_rilevati_totali"] == 3


def test_aggregazione_lista_vuota():
    assert aggrega([]) == {}


# ---------------------------------------------------------------------------
# conta_episodi: blocchi contigui, incluso uno gia' aperto
# ---------------------------------------------------------------------------

def test_conta_episodi_blocchi_separati():
    verdetti = [False] * 3 + [True] * 5 + [False] * 2 + [True] * 4
    r = conta_episodi(verdetti)
    assert r["n_episodi"] == 2
    assert r["gia_in_drift"] is False
    assert r["passi_in_drift"] == 9
    assert r["durata_media"] == 4.5


def test_conta_episodi_gia_in_drift_allinizio():
    # Il caso che si presenta su ELEC2: il verdetto e' gia' True quando
    # comincia la valutazione e non torna mai False. Non e' "zero episodi":
    # e' un unico episodio permanente.
    verdetti = [True] * 100
    r = conta_episodi(verdetti, inizio=10)
    assert r["n_episodi"] == 1
    assert r["gia_in_drift"] is True
    assert r["frazione_tempo"] == 1.0
    # estrai_eventi, che cerca i fronti di salita, non ne trova nessuno.
    assert estrai_eventi(verdetti, inizio=10) == []


def test_conta_episodi_nessuna_segnalazione():
    r = conta_episodi([False] * 50)
    assert r["n_episodi"] == 0
    assert r["frazione_tempo"] == 0.0
    assert r["durata_media"] == 0.0
