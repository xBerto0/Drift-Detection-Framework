# Outline della tesi — struttura completa

**Titolo provvisorio**: *Robust Monitoring and Drift Detection in Machine Learning
Systems: Methods, Metrics, and MLOps Integration*

**Autore**: Alberto Conti
**Università**: Università degli Studi di Palermo, Ingegneria Informatica Magistrale,
curriculum Intelligenza Artificiale
**Relatore**: Prof. Marco La Cascia
**Tutor aziendale**: Daniele Fakhoury (Engineering Ingegneria Informatica S.p.A.)
**Anno accademico**: 2025-2026

---

## Configurazione editoriale

- **Lingua**: italiano
- **Stile citazioni**: IEEE numerato `[1]` — *da confermare con il relatore*
- **Editor**: LaTeX
- **Lunghezza massima**: 120 pagine (vincolo stretto)
- **Font**: Times New Roman 12pt (corpo), 14pt grassetto (titoli capitolo)
- **Interlinea**: 1.5
- **Margini**: superiore/inferiore/destro 2 cm, sinistro 3.5 cm
- **Allineamento**: giustificato

---

## Struttura generale

Struttura accademica classica, con i 6 obiettivi della proposta aziendale mappati
all'interno dei capitoli di metodologia/implementazione/validazione.

### Stato delle sezioni

- ✅ **SCRIVIBILE ORA**: abbiamo materiale sufficiente
- 🟡 **PARZIALMENTE SCRIVIBILE**: alcuni paragrafi sì, altri placeholder
- ⏳ **PLACEHOLDER**: da riempire mano a mano che procediamo col lavoro

---

# Indice della tesi

## Frontespizio

(da template Unipa, da adattare al Dipartimento di Ingegneria)

## Abstract

🟡 — versione italiana e inglese, ~1 pagina ciascuno.
- Italiano: motivazione, obiettivi, contributo, principali risultati.
- Inglese: traduzione fedele dell'abstract italiano.

## Indice generale, indice delle figure, indice delle tabelle

(generati automaticamente con LaTeX)

## Eventuali ringraziamenti

(facoltativi, da inserire alla fine della redazione)

---

## CAPITOLO 1 — Introduzione

**Pagine stimate**: 8-10
**Stato**: ✅ SCRIVIBILE ORA

### 1.1 Contesto e motivazione
Cosa è il Machine Learning in produzione, perché i modelli si degradano nel
tempo, quale problema affronta questa tesi. Cenni al concetto di drift senza
ancora formalizzarlo (verrà fatto nel capitolo 2).

### 1.2 Contesto aziendale
La collaborazione con Engineering Ingegneria Informatica S.p.A., i requisiti
industriali del drift monitoring in scenari MLOps reali.

### 1.3 Obiettivi della tesi
Mappatura dei 6 obiettivi della proposta tutorata:
1. Caratterizzazione e impatto del drift
2. Metriche statistiche e metodi di detection
3. Design e sviluppo dei servizi di monitoring
4. Valutazione delle strategie di retraining
5. Pipeline end-to-end e containerizzazione
6. Documentazione

### 1.4 Contributo
Il valore aggiunto del lavoro: un framework MLOps modulare e model-agnostic per
drift detection, con architettura estendibile basata su Strategy Pattern.

### 1.5 Struttura della tesi
Breve descrizione di cosa contiene ogni capitolo.

---

## CAPITOLO 2 — Stato dell'arte e background

**Pagine stimate**: 18-22
**Stato**: 🟡 PARZIALMENTE SCRIVIBILE

### 2.1 Machine Learning in produzione: il ciclo di vita di un modello
Cenni introduttivi al concetto di ML pipeline, training vs serving, il problema
della stazionarietà dei dati.

### 2.2 Il fenomeno del drift
**🟡** Formalizzazione del concetto di drift. Distinzione fra dataset
stazionario e non stazionario.

### 2.3 Tassonomia del drift
**🟡** Riferimento a [Gama et al. 2014], [Lu et al. 2019]:

- **2.3.1 Feature drift (data drift)** — `P(X)` cambia nel tempo
- **2.3.2 Prediction drift** — `P(Ŷ)` cambia nel tempo
- **2.3.3 Concept drift** — `P(Y|X)` cambia nel tempo
- **2.3.4 Drift improvviso, incrementale, ricorrente** — caratterizzazione
  temporale del cambiamento

### 2.4 Tecniche di drift detection
**🟡** Panoramica dei principali approcci, citando dove rilevante:

- **2.4.1 Test statistici a due campioni** (Kolmogorov-Smirnov, Chi², Mann-Whitney)
- **2.4.2 Misure di divergenza** (Jensen-Shannon, KL, Wasserstein)
- **2.4.3 Metodi error-based** (DDM, EDDM, Page-Hinkley) [Sethi e Kantardzic 2018]
- **2.4.4 Metodi window-based adattivi** (ADWIN, ADWIN2) [Bifet e Gavaldà 2006, 2007]
- **2.4.5 Confronto fra le famiglie di metodi**

### 2.5 MLOps: integrazione del drift monitoring
**⏳** Cenni a MLOps come disciplina, posizionamento del drift monitoring
nel contesto MLOps. Da scrivere meglio quando avremo affrontato la parte di
pipeline.

### 2.6 Librerie e strumenti esistenti
**🟡** Panoramica di:
- Evidently AI (citato dalla proposta)
- Alibi Detect (citato dalla proposta)
- river (usato nel framework)
- scipy.stats (usato nel framework)

Confronto e motivazione delle scelte fatte nella tesi.

---

## CAPITOLO 3 — Architettura del framework

**Pagine stimate**: 12-15
**Stato**: ✅ SCRIVIBILE ORA

### 3.1 Requisiti e principi di design
Modularità, model-agnosticità, estensibilità, separation of concerns.

### 3.2 Visione d'insieme
Schema architetturale a tre livelli: strategie, detector per tipo di drift,
servizio di monitoring centrale.

### 3.3 Il pattern Strategy come scelta di design
**[Gamma et al. 1994]** — perché il Strategy Pattern è la scelta naturale per
un framework di drift detection con algoritmi intercambiabili. Vantaggi e
trade-off rispetto ad alternative (Template Method, ereditarietà multipla).

### 3.4 Le interfacce base
- **3.4.1 `BaseDriftDetector`**: contratto comune per strategie e detector
- **3.4.2 `BaseModel`**: astrazione che garantisce la model-agnosticità
- **3.4.3 `DriftResult`**: dataclass per il verdetto strutturato

### 3.5 Principio Open/Closed nel framework
Discussione di come l'architettura sia chiusa alla modifica ma aperta
all'estensione, con riferimento alle nuove strategie aggiungibili.

---

## CAPITOLO 4 — Strategie di drift detection

**Pagine stimate**: 18-22
**Stato**: ✅ SCRIVIBILE ORA (per KS e ADWIN); ⏳ per le altre

### 4.1 La strategia Kolmogorov-Smirnov

#### 4.1.1 Fondamenti teorici
- Test di Kolmogorov-Smirnov a due campioni
- ECDF e statistica D
- Calcolo del p-value
- Confronto con soglia di significatività α
- Natura non-parametrica e adeguatezza per feature continue

#### 4.1.2 Adattamento allo streaming
Definizione di reference window (fissa) e current window (FIFO scorrevole).
Discussione del trade-off fra dimensione delle finestre e potere statistico.

#### 4.1.3 Implementazione: `KSStrategy`
Codice essenziale, parametri esposti (`window_size`, `alpha`), dettagli
implementativi (uso di `collections.deque`, integrazione con
`scipy.stats.ks_2samp`).

#### 4.1.4 Limiti e considerazioni
Potere statistico ridotto su dati discreti, finestre statiche, segnale
continuo dopo il rilevamento.

### 4.2 La strategia ADWIN

#### 4.2.1 Fondamenti teorici
- Adaptive Windowing: l'idea di finestra a dimensione variabile
- Riferimento al paper [Bifet e Gavaldà 2006]
- Distinzione fra ADWIN base (sez. 3.2 del paper) e ADWIN2 (sez. 3.3 del paper)
- Motivazione dell'utilizzo di ADWIN2 in pratica

#### 4.2.2 La struttura a istogrammi esponenziali
- Bucket: capacità e contenuto
- Regola dei `M` bucket per dimensione
- Meccanismo di fusione e cascata
- Costi di memoria O(M·log(W/M)) e tempo O(log W)

#### 4.2.3 Il meccanismo del taglio
- Tagli candidati ai confini dei bucket
- La formula `ε_cut` (Equazione 3.1 del paper)
- Confronto `|μ(W₀) − μ(W₁)| > ε_cut`
- Eliminazione della porzione vecchia

#### 4.2.4 Garanzie teoriche
- Teorema 3.1 del paper: limiti su falsi positivi e falsi negativi
- Confronto con l'assenza di garanzie analoghe in molti altri detector

#### 4.2.5 Implementazione: `ADWINStrategy`
Codice essenziale, wrapping di `river.drift.ADWIN`, parametro `delta`,
motivazione della scelta di utilizzare una libreria consolidata anziché
re-implementare l'algoritmo.

#### 4.2.6 Limiti e considerazioni
Sensibilità solo alla media, segnale a impulso, parametri di river non
esposti dal wrapper.

### 4.3 Confronto fra le strategie implementate

Tabella comparativa: forma di confronto, requisiti di memoria, latenza tipica,
adeguatezza per dati continui vs discreti, garanzie teoriche.

### 4.4 Strategie future (placeholder)

**⏳** Brevi cenni alle strategie che verranno aggiunte:
- Jensen-Shannon Divergence (per feature/prediction drift)
- DDM, EDDM, Page-Hinkley (per concept drift)

---

## CAPITOLO 5 — Detector e orchestrazione

**Pagine stimate**: 12-15
**Stato**: 🟡 PARZIALMENTE SCRIVIBILE (FeatureDriftDetector pronto; gli altri placeholder)

### 5.1 Architettura dei detector per tipo di drift
Cosa fa un "detector" rispetto a una "strategia". Pattern di smistamento del
flusso e aggregazione dei verdetti.

### 5.2 `FeatureDriftDetector`
**✅** Implementazione, gestione delle N strategie indipendenti (una per
feature), regola di aggregazione `k`, output strutturato in `DriftResult`.

### 5.3 `PredictionDriftDetector`
**⏳** Da progettare e implementare. Cenni al riuso delle stesse strategie
sullo stream delle predizioni, con discussione del trade-off tra monitorare
classi predette vs probabilità predette (per modelli di classificazione).

### 5.4 `ConceptDriftDetector`
**⏳** Da progettare e implementare. Cenni alla necessità di `y_true`,
calcolo dello stream di errori, gestione di scenari di label delay.

### 5.5 `DriftMonitoringService` — orchestratore centrale
**⏳** Architettura del servizio centrale, gestione del modello via `BaseModel`,
smistamento dei dati ai vari detector, aggregazione finale dei risultati.

---

## CAPITOLO 6 — Validazione sperimentale

**Pagine stimate**: 18-22
**Stato**: 🟡 PARZIALMENTE SCRIVIBILE

### 6.1 Metodologia di validazione
Generale: come si valuta un drift detector. Metriche: latenza di rilevamento,
tasso di falsi positivi, tasso di falsi negativi, robustezza.

### 6.2 Generatore di dati sintetici
**✅** Descrizione del modulo `synthetic_generator.py`, scelte del scenario
Bernoulli, fedeltà al paper di Bifet & Gavaldà.

### 6.3 Esperimento 1: KS su dati sintetici Bernoulli
**✅** Setup, parametri, risultati, interpretazione. Discussione della
latenza ~100 passi e dell'assenza di falsi positivi.

### 6.4 Esperimento 2: ADWIN su dati sintetici Bernoulli
**✅** Setup, parametri, risultati, interpretazione. Discussione della
latenza ~55 passi e del comportamento a impulso.

### 6.5 Confronto KS vs ADWIN sullo stesso scenario
**✅** Analisi comparativa dei risultati, conferma sperimentale delle
proprietà teoriche, discussione delle scelte di design del framework.

### 6.6 Esperimento 3: validazione su dataset reale ELEC2
**⏳** Da fare nelle prossime settimane. Descrizione del dataset, applicazione
delle due strategie, confronto con risultati attesi dalla letteratura.

### 6.7 Esperimenti futuri
**⏳** Validazione di prediction drift, concept drift, su dataset reali
multipli (da definire mano a mano).

---

## CAPITOLO 7 — Pipeline MLOps e integrazione (future)

**Pagine stimate**: 10-15
**Stato**: ⏳ PLACEHOLDER COMPLETO

### 7.1 Architettura della pipeline end-to-end
**⏳** Da progettare. Cenni a: ingestion → modello → drift monitoring →
alerting/retraining.

### 7.2 Tracking degli esperimenti con MLflow
**⏳**

### 7.3 Orchestrazione con Argo Workflows e Argo Events
**⏳**

### 7.4 Containerizzazione con Docker
**⏳**

### 7.5 Strategie di retraining
**⏳** Trigger basati su drift detection, valutazione di shadow models,
rollback strategies.

### 7.6 Test end-to-end della pipeline
**⏳**

---

## CAPITOLO 8 — Conclusioni e sviluppi futuri

**Pagine stimate**: 5-7
**Stato**: ⏳ SCRIVIBILE A LAVORO COMPLETO

### 8.1 Riepilogo del lavoro svolto

### 8.2 Risultati principali e contributo

### 8.3 Limiti del lavoro

### 8.4 Sviluppi futuri

Direzioni interessanti per estensioni della tesi, alcune già identificate:
- Ensemble di detector
- Drift scoring continuo
- Monitoraggio per-feature avanzato
- Supporto a label delay per concept drift
- Windowing adattivo per il KS

---

## Bibliografia

**🟡** Da costruire e completare in itinere. Riferimenti già identificati:

### Articoli di rivista e atti di convegno
1. Gama, J. et al. (2014). *A Survey on Concept Drift Adaptation*. ACM Computing Surveys.
2. Lu, J. et al. (2019). *Learning under Concept Drift: A Review*. IEEE TKDE.
3. Sethi, T., Kantardzic, M. (2018). *On the reliable detection of concept drift from streaming unlabeled data*. Expert Systems with Applications.
4. Baier, L. et al. (2020). *Detecting Concept Drift With Neural Network Model Uncertainty*.
5. Bifet, A., Gavaldà, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing*. SIAM Int. Conf. on Data Mining.
6. Bifet, A., Gavaldà, R. (2006). *Learning from Time-Changing Data with Adaptive Windowing* (technical report, UPC).
7. Datar, M., Gionis, A., Indyk, P., Motwani, R. (2002). *Maintaining stream statistics over sliding windows*. SIAM Journal on Computing.
8. Montiel, J., Halford, M., Mastelini, S. M., et al. (2021). *River: machine learning for streaming data in Python*. JMLR.

### Libri
9. Gamma, E., Helm, R., Johnson, R., Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.

### Documentazione tecnica e siti web
10. MLflow Documentation. https://mlflow.org/
11. Argo Project Documentation. https://argoproj.github.io/
12. Alibi Detect Documentation. https://docs.seldon.ai/alibi-detect
13. Evidently AI Documentation. https://www.evidentlyai.com/
14. Docker Documentation. https://docs.docker.com/

### Riferimenti recenti da aggiungere
**⏳** Da cercare e integrare: lavori 2023-2025 sullo stato dell'arte del
drift detection, ricerche specifiche su streaming ML production.

---

## Appendici (opzionali)

**⏳** Eventualmente:
- A. Codice sorgente integrale del framework
- B. Dettagli matematici della formula ε_cut di ADWIN
- C. Istruzioni per la riproducibilità degli esperimenti

---

## Stime di lunghezza

| Capitolo | Pagine stimate |
|---|---|
| Frontespizio + abstract + indici | ~10 |
| Cap. 1 — Introduzione | 8-10 |
| Cap. 2 — Stato dell'arte | 18-22 |
| Cap. 3 — Architettura del framework | 12-15 |
| Cap. 4 — Strategie di drift detection | 18-22 |
| Cap. 5 — Detector e orchestrazione | 12-15 |
| Cap. 6 — Validazione sperimentale | 18-22 |
| Cap. 7 — Pipeline MLOps | 10-15 |
| Cap. 8 — Conclusioni | 5-7 |
| Bibliografia | 5-8 |
| **Totale stimato** | **116-146** |

Considerato il vincolo massimo di **120 pagine**, sarà necessario contenere
lo stato dell'arte (cap. 2) e la pipeline MLOps (cap. 7) entro la fascia
bassa delle stime, oppure spostare alcuni dettagli implementativi in appendice.

---

## Cosa è scrivibile ORA con il materiale che abbiamo

Tutti i capitoli marcati ✅ o 🟡 hanno almeno alcune sezioni scrivibili
immediatamente. Concretamente:

- **Capitolo 1** — completo
- **Capitolo 2** — sezioni 2.1, 2.2, 2.3, 2.4.1, 2.4.4, 2.6
- **Capitolo 3** — completo
- **Capitolo 4** — sezioni 4.1, 4.2, 4.3 (le sezioni su KS e ADWIN)
- **Capitolo 5** — solo sezione 5.2 (FeatureDriftDetector)
- **Capitolo 6** — sezioni 6.1, 6.2, 6.3, 6.4, 6.5 (manca ELEC2)

I capitoli marcati ⏳ vanno aperti come placeholder vuoti, con i titoli e
sotto-titoli già piazzati, da riempire mano a mano che procediamo col lavoro
(ELEC2, prediction drift, concept drift, MLOps pipeline).

---

## Prossimi passi suggeriti

1. **Confermare con La Cascia** le scelte di formato (citazioni IEEE,
   eventuali norme specifiche di Ingegneria) e la struttura proposta
2. **Scrivere il Capitolo 1** (Introduzione) — è il più semplice da chiudere
   e dà subito un punto di riferimento
3. **Procedere col Capitolo 4 sezioni 4.1 e 4.2** — abbiamo già la
   documentazione tecnica completa in `docs/adwin_overview.md`, basta
   adattarla allo stile della tesi
4. **Capitolo 2** in parallelo, costruendo lo stato dell'arte mentre si
   procede col lavoro tecnico (alcune referenze possono emergere via via)
