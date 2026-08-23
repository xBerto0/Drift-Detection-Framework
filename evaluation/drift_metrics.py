"""Metriche di valutazione dei drift detector rispetto a una ground truth nota.

Questo modulo risolve due problemi metodologici distinti.

--------------------------------------------------------------------------
PROBLEMA 1 — Confrontare segnali di natura diversa
--------------------------------------------------------------------------
I detector implementati non segnalano tutti allo stesso modo:

- ADWIN, DDM e Page-Hinkley sono EDGE-TRIGGERED: `drift_detected` vale True
  solo nell'istante in cui l'algoritmo taglia la finestra o supera la soglia.
  Contano EVENTI.
- Il KS e' LEVEL-TRIGGERED: il test viene rifatto a ogni passo su finestre che
  si sovrappongono quasi completamente, e resta True finche' la condizione
  persiste. Conta PASSI IN STATO DI DRIFT.

Confrontare direttamente i due conteggi non ha senso: dire che "il KS produce
44.413 segnalazioni e ADWIN 558, quindi ADWIN e' 80 volte meno rumoroso"
significa dividere mele per arance. Quelle 44.413 non sono 44.413 drift
distinti: sono pochi episodi che durano a lungo, moltiplicati per test
ripetuti su dati quasi identici e quindi statisticamente dipendenti.

La soluzione adottata e' contare EVENTI e non PASSI: si registra solo la
transizione da "non in drift" a "in drift" (rilevamento di fronte di salita).
Su un detector edge-triggered l'operazione e' l'identita'; su uno
level-triggered collassa ogni sequenza contigua di True nel suo primo passo.
Dopo questa trasformazione i due tipi di detector sono confrontabili.

--------------------------------------------------------------------------
PROBLEMA 2 — Valutare invece di descrivere
--------------------------------------------------------------------------
Contare le segnalazioni non dice se il detector ha lavorato bene. Servono
metriche riferite a una ground truth:

- LATENZA (detection delay): quanti passi fra il drift reale e la prima
  segnalazione corretta. Misura la prontezza.
- FALSI ALLARMI: segnalazioni che non corrispondono ad alcun drift reale.
  Misura l'affidabilita' operativa.
- MANCATE RILEVAZIONI: drift reali che nessuna segnalazione ha coperto.
  Misura la copertura.

Il protocollo di abbinamento segue la prassi consolidata negli studi
comparativi sui drift detector (Bifet & Gavalda 2007; Barros & Santos 2018):
attorno a ogni drift reale si definisce una finestra di tolleranza, e una
segnalazione conta come corretta solo se cade dentro quella finestra.
"""

from dataclasses import dataclass, field
from statistics import mean, pstdev


@dataclass
class MetricheRilevamento:
    """Esito della valutazione di un detector su un singolo stream."""

    n_drift_reali: int
    n_eventi: int                 # segnalazioni dopo il collasso in eventi
    n_rilevati: int               # drift reali correttamente coperti
    n_falsi_allarmi: int
    n_mancati: int
    n_duplicati: int              # eventi extra dentro una finestra gia' coperta
    latenze: list = field(default_factory=list)
    campioni_valutati: int = 0
    campioni_fuori_finestra: int = 0

    @property
    def latenza_media(self):
        """Latenza media sui soli drift effettivamente rilevati."""
        return mean(self.latenze) if self.latenze else None

    @property
    def tasso_mancate(self):
        """Missed Detection Rate: frazione di drift reali non rilevati."""
        if self.n_drift_reali == 0:
            return None
        return self.n_mancati / self.n_drift_reali

    @property
    def falsi_allarmi_per_1000(self):
        """Falsi allarmi ogni 1000 campioni in cui non c'era drift da rilevare.

        Normalizzare sulla lunghezza e' necessario: un detector valutato su
        uno stream di 20.000 campioni non e' confrontabile con uno valutato su
        2.000 se si guardano i conteggi assoluti.
        """
        if self.campioni_fuori_finestra <= 0:
            return None
        return self.n_falsi_allarmi / (self.campioni_fuori_finestra / 1000)


def estrai_eventi(segnalazioni, inizio=0):
    """Collassa uno stream booleano di verdetti in una lista di eventi.

    Viene registrato solo il passo in cui il verdetto passa da False a True
    (fronte di salita). E' la trasformazione che rende confrontabili detector
    edge-triggered e level-triggered: vedere la nota in testa al modulo.

    `inizio` permette di ignorare i primi passi, tipicamente la fase di
    warming up del KS, durante la quale nessun verdetto e' significativo.
    """
    eventi = []
    precedente = False
    for t, corrente in enumerate(segnalazioni):
        if t < inizio:
            # Durante il warming up non si valuta, ma si tiene comunque
            # traccia dello stato per non registrare un falso fronte al
            # primo passo utile.
            precedente = bool(corrente)
            continue
        if corrente and not precedente:
            eventi.append(t)
        precedente = bool(corrente)
    return eventi


def valuta(eventi, punti_drift, n_campioni, tolleranza=500, inizio=0):
    """Confronta gli eventi rilevati con i punti di drift reali.

    Protocollo di abbinamento:

    - attorno a ogni drift reale d si apre la finestra [d, d + tolleranza];
    - un evento che cade nella finestra di un drift non ancora coperto e' un
      RILEVAMENTO CORRETTO, e la sua distanza da d e' la latenza;
    - un evento che cade in una finestra gia' coperta e' un DUPLICATO: non
      viene contato ne' come successo ne' come falso allarme, perche' si
      riferisce a un drift gia' segnalato;
    - un evento fuori da ogni finestra e' un FALSO ALLARME;
    - un drift reale che nessun evento ha coperto e' una MANCATA RILEVAZIONE.

    La `tolleranza` va scelta piu' piccola della distanza minima fra due drift
    consecutivi, altrimenti le finestre si sovrappongono e l'abbinamento
    diventa ambiguo.
    """
    drift_ordinati = sorted(punti_drift)
    coperti = set()

    n_corretti = 0
    n_falsi = 0
    n_duplicati = 0
    latenze = []

    for t in eventi:
        # Cerca il drift reale piu' recente che precede l'evento.
        candidato = None
        for d in drift_ordinati:
            if d <= t:
                candidato = d
            else:
                break

        if candidato is not None and (t - candidato) <= tolleranza:
            if candidato in coperti:
                n_duplicati += 1
            else:
                coperti.add(candidato)
                latenze.append(t - candidato)
                n_corretti += 1
        else:
            n_falsi += 1

    # Campioni in cui una segnalazione sarebbe stata un falso allarme: tutti
    # quelli valutati, meno quelli coperti dalle finestre di tolleranza.
    campioni_valutati = max(0, n_campioni - inizio)
    campioni_in_finestra = 0
    for d in drift_ordinati:
        fine = min(n_campioni, d + tolleranza)
        campioni_in_finestra += max(0, fine - max(d, inizio))
    campioni_fuori = max(0, campioni_valutati - campioni_in_finestra)

    return MetricheRilevamento(
        n_drift_reali=len(drift_ordinati),
        n_eventi=len(eventi),
        n_rilevati=n_corretti,
        n_falsi_allarmi=n_falsi,
        n_mancati=len(drift_ordinati) - n_corretti,
        n_duplicati=n_duplicati,
        latenze=latenze,
        campioni_valutati=campioni_valutati,
        campioni_fuori_finestra=campioni_fuori,
    )


def aggrega(lista_metriche):
    """Aggrega le metriche di piu' ripetizioni (seed diversi) in media e dev.std.

    Una singola esecuzione su un solo seed non dice quasi nulla: un falso
    allarme puo' comparire o sparire cambiando soltanto il seme del generatore
    casuale. Riportare media e deviazione standard su N ripetizioni e' il
    minimo perche' un risultato sia difendibile.
    """
    if not lista_metriche:
        return {}

    def media_dev(valori):
        puliti = [v for v in valori if v is not None]
        if not puliti:
            return (None, None)
        return (mean(puliti), pstdev(puliti) if len(puliti) > 1 else 0.0)

    latenze_medie = [m.latenza_media for m in lista_metriche]
    far = [m.falsi_allarmi_per_1000 for m in lista_metriche]
    mdr = [m.tasso_mancate for m in lista_metriche]

    lat_m, lat_s = media_dev(latenze_medie)
    far_m, far_s = media_dev(far)
    mdr_m, mdr_s = media_dev(mdr)

    return {
        "n_ripetizioni": len(lista_metriche),
        "latenza_media": lat_m,
        "latenza_std": lat_s,
        "falsi_allarmi_per_1000_media": far_m,
        "falsi_allarmi_per_1000_std": far_s,
        "tasso_mancate_media": mdr_m,
        "tasso_mancate_std": mdr_s,
        "n_eventi_media": mean([m.n_eventi for m in lista_metriche]),
        "n_falsi_allarmi_totali": sum(m.n_falsi_allarmi for m in lista_metriche),
        "n_drift_reali_totali": sum(m.n_drift_reali for m in lista_metriche),
        "n_rilevati_totali": sum(m.n_rilevati for m in lista_metriche),
        "seed_con_falsi_allarmi": sum(
            1 for m in lista_metriche if m.n_falsi_allarmi > 0
        ),
    }
