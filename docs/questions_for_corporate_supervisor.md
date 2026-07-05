# Domande per il tutor aziendale (Daniele Fakhoury)

Domande utili da fare al tutor aziendale per chiarire scope, requisiti e
direzioni del lavoro di tesi. Sono raggruppate per area, in ordine di
priorità decrescente per ciascun gruppo.

---

## 1. Contesto produttivo

Senza queste informazioni è difficile orientare la parte sperimentale e la
pipeline MLOps verso scenari realistici.

1. **Quali sono i modelli ML reali su cui il framework potenzialmente
   andrà a operare in azienda?** Classificazione binaria, multiclasse,
   regressione, NLP, vision? Mi aiuta a capire se devo prevedere strategie
   diverse per output diversi.

2. **Quanti modelli ci sono in produzione contemporaneamente?** Decine?
   Centinaia? Mille? Cambia le scelte sulla scalabilità del monitoring.

3. **Avete già un'esperienza concreta di degradazione di modelli in
   produzione che potete raccontarmi?** Un case study reale che ha portato
   all'avvio di questa tesi sarebbe materiale prezioso per il Capitolo 1.

4. **Avete una preferenza tra approcci batch (drift valutato a fine
   giornata/settimana) e streaming (drift valutato in tempo reale, valore
   per valore)?** Cambia molto l'architettura.

---

## 2. Scenari di drift di interesse

5. **C'è un tipo di drift che vi preoccupa di più nel vostro contesto?**
   Feature drift (distribuzione input), prediction drift (distribuzione
   output), concept drift (relazione input-output)?

6. **Avete scenari ricorrenti di drift specifici?** Es: drift stagionale,
   drift legato a release di nuove versioni del prodotto, drift legato a
   shock esterni (crisi, cambi normativi). Il framework dovrebbe gestirne
   alcuni in particolare?

7. **Quanto è critico il "concept drift" rispetto al "feature drift" nella
   vostra pratica?** Le label `y_true` sono disponibili rapidamente
   (qualche ora/giorno) o c'è un forte label delay (settimane/mesi)?

---

## 3. Infrastruttura MLOps esistente

8. **Avete già una stack MLOps in produzione?** MLflow è già usato? Argo
   Workflows è già installato o va integrato da zero?

9. **Su quale piattaforma cloud/on-premise gira la produzione?** AWS,
   Azure, GCP, Kubernetes on-prem? Influenza le scelte per la
   containerizzazione.

10. **Avete uno standard interno per il monitoring/alerting?** Grafana?
    Prometheus? Slack? Email? Mi serve sapere come integrare gli alert
    di drift.

11. **Esiste già un sistema interno di feature store o di gestione dataset
    di training?** Il framework dovrà integrarsi con esso?

---

## 4. Scope del framework

12. **Il framework che sto costruendo deve essere production-ready** (cioè
    effettivamente desplyabile in azienda), oppure è inteso come
    **proof-of-concept** per dimostrare la fattibilità del monitoring di
    drift? Cambia il livello di robustezza, gestione errori, logging,
    osservabilità richiesto.

13. **Quanto custom-code vs librerie commerciali?** La proposta cita
    Evidently AI e Alibi Detect, ma noi ne stiamo costruendo uno custom.
    Mi confermi la direzione: framework custom con possibilità di
    integrazione futura con quelle librerie?

14. **Quali sono le 2-3 cose minime che il framework deve fare** per
    essere considerato un successo dal punto di vista aziendale?
    (Esempio: "rilevare drift su 5 feature in tempo reale + dashboard").
    Mi aiuta a calibrare le priorità.

---

## 5. Dati e validazione

15. **Avete dataset reali aziendali (anonimizzati) che posso usare per
    testare il framework?** O devo lavorare solo con dataset pubblici
    (ELEC2, KDDCup99, ecc.)? Validare su dati aziendali darebbe peso
    industriale ai risultati.

16. **Avete benchmark interni di riferimento** rispetto ai quali il mio
    framework dovrebbe essere confrontato?

17. **C'è uno scenario sperimentale specifico** che vorreste vedere
    nella tesi (es. "vorremmo che testaste il framework su 30 giorni di
    log di un modello esistente con un drift noto al giorno 14")?

---

## 6. Retraining strategies

L'Objective 4 della proposta parla esplicitamente di retraining triggers,
ed è un'area che al momento non ho ancora affrontato.

18. **Il retraining deve essere effettivamente automatico** (cioè il sistema
    addestra un nuovo modello da solo) **oppure il framework deve solo
    segnalare la necessità di retraining** a un operatore umano?

19. **Avete un processo aziendale di retraining già definito?** Frequenza,
    dataset usati, validazione del nuovo modello, deployment del nuovo
    modello.

20. **Vi interessa che la tesi esplori strategie come shadow models o
    A/B testing?** O preferite una linea più conservativa (segnalazione
    + intervento manuale)?

---

## 7. Aspetti pratici

21. **Quanto coordinamento c'è tra te (tutor aziendale) e il prof La
    Cascia (relatore accademico)?** Vi parlerete in autonomia o devo io
    fare da ponte sulle scelte importanti?

22. **Avete vincoli temporali "duri"** entro cui il framework deve essere
    pronto in azienda? (Es: "vorremmo usarlo internamente entro
    settembre 2026").

23. **Ci sono brevetti/proprietà intellettuale** o riservatezza su parti
    del lavoro? Devo evitare di pubblicare alcuni dettagli nella tesi?

24. **Per la stesura della tesi**: la parte di descrizione dell'azienda e
    del contesto produttivo va validata da voi? C'è una versione "standard"
    della descrizione aziendale che usate per le tesi?

---

## 8. Per il prossimo incontro

Domande "pratiche" da preparare per la prossima call:

25. **Posso avere accesso a un ambiente di test interno** dove sperimentare
    il framework su un caso reale?

26. **Possiamo organizzare un kick-off tecnico** dove tu o un collega mi
    mostra l'attuale processo di monitoring (anche solo a alto livello),
    così so dove la mia tesi si va a inserire?

27. **Ci sono colleghi tecnici** (data scientist, ML engineer, MLOps
    engineer) con cui posso interfacciarmi per domande pratiche durante
    lo sviluppo?

---

## Suggerimento di priorità

Se non riesci a fare tutte le 27 domande in una sola call, queste sono le
**5 più critiche** da chiarire subito:

1. **Q12** — Production-ready o proof-of-concept?
2. **Q5** — Quale tipo di drift è più critico?
3. **Q15** — Dataset reali disponibili per validazione?
4. **Q18** — Retraining automatico o segnalazione manuale?
5. **Q1** — Tipi di modelli su cui il framework dovrà operare?

Le risposte a queste cinque domande ti permettono di prendere tutte le
decisioni architetturali importanti per il resto del lavoro.
