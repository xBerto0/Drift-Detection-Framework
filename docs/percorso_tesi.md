# Percorso del lavoro di tesi — narrazione completa

**Autore**: Alberto Conti
**Anno accademico**: 2025-2026
**Data compilazione documento**: 2026-07
**Obiettivo del documento**: fornire a un LLM (o a un lettore umano) tutto il
contesto, i ragionamenti, le decisioni di design e i risultati del lavoro di
tesi, in modo da poter cominciare a scrivere la tesi in modo informato e
coerente. Non è un manuale tecnico (per quello vedere
`docs/thesis_writing_briefing.md` e `docs/adwin_overview.md`) ma il **racconto
del percorso**, con particolare attenzione al *perché* delle scelte fatte.

---

# INDICE

0. Come usare questo documento
1. Contesto della tesi
2. Fase 1 — Le fondamenta architetturali
3. Fase 2 — Le interfacce base
4. Fase 3 — La strategia Kolmogorov-Smirnov (KS)
5. Fase 4 — Il FeatureDriftDetector
6. Fase 5 — La strategia ADWIN
7. Fase 6 — Prediction e Concept Drift Detector
8. Fase 7 — Il classificatore GaussianNB
9. Fase 8 — L'esperimento su ELEC2
10. Fase 9 — Analisi trasversale dei risultati
11. Fase 10 — Esternalizzazione dei parametri
12. Riferimenti bibliografici emersi
13. Cosa manca e prossimi passi
14. Come usare questo documento per la tesi

---

# 0. Come usare questo documento

Questo file racconta l'intero percorso di sviluppo del framework
`drift_framework` come è emerso in una serie di conversazioni fra Alberto
Conti e un assistente AI. Ogni sezione documenta:

- **Cosa è stato deciso** (la decisione operativa)
- **Perché** (il ragionamento che ha portato alla decisione, incluse le
  alternative valutate)
- **Cosa è stato implementato** (in termini di codice o documenti prodotti)
- **Cosa è emerso di significativo** (spunti utili in fase di redazione tesi)

Quando questo file viene passato a un LLM per assistere nella scrittura della
tesi, l'LLM dovrebbe usarlo come *guida narrativa*: sa quali capitoli servono
(vedi `docs/thesis_outline.md`), sa cosa metterci dentro (vedi
`docs/thesis_writing_briefing.md`), ma **da questo documento apprende
il perché e la storia delle scelte fatte**.

---

# 1. Contesto della tesi

## 1.1 Il progetto

Il lavoro di tesi consiste nella progettazione e implementazione di un
framework MLOps modulare per **drift detection** — cioè per rilevare
cambiamenti nel tempo nel comportamento di un modello di Machine Learning
in produzione. Il progetto è svolto in collaborazione con **Engineering
Ingegneria Informatica S.p.A.**, con tutor aziendale **Daniele Fakhoury**.
Il relatore accademico è il **Prof. Marco La Cascia** presso l'Università
degli Studi di Palermo, Corso di Laurea Magistrale in Ingegneria
Informatica curriculum Intelligenza Artificiale.

## 1.2 Il problema affrontato

Il "drift" è un problema silenzioso e insidioso dei modelli ML in
produzione: nel tempo, i dati di input possono cambiare distribuzione, o
la relazione fra input e output può evolvere. Un modello addestrato su
dati storici degrada senza generare errori esplicit — semplicemente
produce predizioni sempre meno accurate. Rilevarlo tempestivamente è
critico per sistemi che prendono decisioni operative (credit scoring,
fraud detection, manutenzione predittiva, ecc.).

## 1.3 L'ambizione della tesi

Non implementare l'ennesimo algoritmo di drift detection, ma costruire
un **framework architetturale** che:

- Sia **modulare** (nuovi algoritmi si aggiungono senza toccare il core)
- Sia **model-agnostic** (funziona con qualunque modello ML)
- Copra i **tre tipi principali di drift** (feature, prediction, concept)
- Sia **testabile e riproducibile** (esperimenti su dataset sintetici e reali)
- Possa essere integrato in una **pipeline MLOps industriale** (in
  prospettiva: MLflow, Argo, Docker)

Il **valore aggiunto della tesi è l'architettura**, non gli algoritmi: le
tecniche di drift detection scelte (KS, ADWIN) sono consolidate in
letteratura. La novità sta nel **come vengono composte** in un sistema
riutilizzabile.

---

# 2. Fase 1 — Le fondamenta architetturali

## 2.1 Il Strategy Pattern come scelta guida

**Decisione**: adottare il *Strategy Pattern* (Gamma et al., 1994) come
principio architetturale portante.

**Perché**:

Sono state discusse alternative:

- **Ereditarietà multipla**: un unico detector con più metodi di detection.
  Rifiutata perché legherebbe il detector a un algoritmo specifico e
  renderebbe difficile aggiungere nuove tecniche.
- **Chain-of-Responsibility**: una catena di detector. Interessante ma
  eccessiva per il nostro caso — non ci serve una gerarchia di
  handler.
- **Template Method**: un algoritmo scheletro con hook. Non adatto perché
  le tecniche di detection sono strutturalmente diverse (KS batch statistico
  vs ADWIN adaptive streaming), non variazioni dello stesso scheletro.

Il Strategy Pattern si è imposto come la scelta naturale perché:

1. Ogni tecnica di detection è **incapsulata in una classe** con la stessa
   interfaccia (`BaseDriftDetector`)
2. Le strategie sono **intercambiabili a runtime** (basta cambiare
   `strategy_cls`)
3. Aggiungere una nuova strategia **non richiede modifiche al codice
   esistente** (principio Open/Closed)

Questa scelta si è rivelata corretta in modo empirico: quando abbiamo
aggiunto ADWIN dopo KS, non è stata necessaria alcuna modifica al
`FeatureDriftDetector`.

## 2.2 I tre livelli dell'architettura

**Decisione**: separare il framework in tre livelli.

```
Livello 3 — DriftMonitoringService (orchestrazione globale) [futuro]
                │
Livello 2 — FeatureDriftDetector, PredictionDriftDetector, ConceptDriftDetector
                │
Livello 1 — KSStrategy, ADWINStrategy, ... (algoritmi su singolo stream)
```

**Perché**:

- **Livello 1 (strategie)**: sanno fare una sola cosa — dato uno stream di
  numeri, dire se è cambiato. Non sanno cosa siano quei numeri.
- **Livello 2 (detector per tipo di drift)**: orchestrano più strategie o
  gestiscono la specificità del tipo di drift (es. il concept drift
  calcola l'errore internamente).
- **Livello 3 (servizio centrale)**: futuro — collegherà modello e
  detector, gestirà l'input, aggregerà i risultati.

Questa separazione garantisce che **la stessa strategia possa servire più
tipi di drift**: `ADWINStrategy` viene usata sia per feature drift (via
`FeatureDriftDetector`) sia per prediction drift sia per concept drift,
senza duplicazione di codice.

## 2.3 Il principio della model-agnosticità

**Decisione**: il framework non deve fare alcuna assunzione sulla natura
del modello ML che monitora.

**Perché**:

Un framework MLOps industriale non può essere legato a un tipo specifico
di modello. Il modello viene visto come un'entità che espone unicamente
un metodo `predict()`. Tutte le informazioni per il drift detection
vengono ricavate dagli **stream di dati** che attraversano il modello:

- Le feature in ingresso (per feature drift)
- Le predizioni in uscita (per prediction drift)
- Gli errori (per concept drift, quando `y_true` è disponibile)

Questa scelta ha reso possibile testare il framework con un `GaussianNB`
di sklearn senza aver bisogno di alcun adattamento per la sua natura
probabilistica.

---

# 3. Fase 2 — Le interfacce base

## 3.1 `BaseDriftDetector`

**Cosa**: classe astratta che definisce il contratto comune a tutte le
strategie e a tutti i detector.

```python
class BaseDriftDetector(ABC):
    def __init__(self, detector_name, drift_type):
        self.detector_name = detector_name
        self.drift_type = drift_type

    @abstractmethod
    def update(self, value): ...

    @abstractmethod
    def detect(self) -> DriftResult: ...

    @abstractmethod
    def reset(self): ...
```

**Il perché di ogni metodo**:

- **`update(value)`**: riceve un nuovo dato e aggiorna lo stato interno.
  Non decide, non produce output. È lo strumento con cui la strategia
  "vede" i dati man mano che arrivano.
- **`detect()`**: legge lo stato corrente e restituisce un verdetto
  strutturato (`DriftResult`). È **idempotente**: chiamarlo due volte di
  fila senza `update` in mezzo dà lo stesso risultato.
- **`reset()`**: riporta la strategia allo stato iniziale. Utile dopo un
  retraining del modello.

**Ragionamento sulla separazione update/detect**: durante una delle prime
discussioni è emersa la domanda "perché non un solo metodo che aggiorna
e decide?". La risposta: separare **scrittura** e **lettura** dello
stato garantisce che:

1. Si può chiamare `detect()` più volte fra un `update()` e l'altro
2. Si può testare `detect()` in isolamento
3. Il chiamante controlla esplicitamente *quando* interrogare il verdetto

## 3.2 `BaseModel`

**Cosa**: astrazione minimale del modello ML.

```python
class BaseModel(ABC):
    @abstractmethod
    def predict(self, x): ...

    @abstractmethod
    def get_name(self) -> str: ...
```

**Il perché**: garantire la **model-agnosticità** tramite un'interfaccia
minimale. Il framework non conosce (né gli interessa) come è fatto
internamente il modello, sa solo che espone `predict()`.

## 3.3 `DriftResult`

**Cosa**: dataclass che rappresenta il verdetto strutturato prodotto da
un detector.

```python
@dataclass
class DriftResult:
    detector_name: str
    drift_detected: bool
    drift_type: str
    score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
```

**Il perché di `metadata` come dict libero**: discussione importante. È
emerso il dubbio "perché tenere `metadata` invece di campi specifici?".
La conclusione: `metadata` è il **contenitore libero** che permette a
ogni strategia di aggiungere informazioni specifiche (statistica D e alpha
per KS, delta per ADWIN, lista di feature drittate per il
`FeatureDriftDetector`) **senza modificare la dataclass**. È
applicazione diretta del principio Open/Closed.

---

# 4. Fase 3 — La strategia Kolmogorov-Smirnov (KS)

## 4.1 Il concetto teorico

Il **test di Kolmogorov-Smirnov a due campioni** è un test statistico
**non-parametrico** che verifica se due campioni di dati provengono dalla
stessa distribuzione di probabilità. L'aspetto non-parametrico è
fondamentale: non richiede assunzioni sulla forma della distribuzione
(es. normalità).

## 4.2 Il ragionamento pedagogico sviluppato in chat

Durante le conversazioni è emersa la necessità di chiarire diversi
concetti statistici che sono utili per la tesi:

### 4.2.1 L'ECDF (Empirical Cumulative Distribution Function)

Dato un campione `X = {x_1, ..., x_n}`, l'ECDF è la funzione:
`F(x) = (numero di x_i ≤ x) / n`. È una funzione a gradini, che parte da 0 e
arriva a 1, con un salto di `1/n` in corrispondenza di ogni valore del
campione. Rappresenta la **"forma cumulativa"** della distribuzione
osservata.

### 4.2.2 La statistica D

Dato due campioni con ECDF `F_1` e `F_2`, la statistica D del KS test è:
`D = max_x | F_1(x) − F_2(x) |`.

È la **massima distanza verticale** fra le due ECDF sovrapposte.
Interpretazione:
- D = 0 → le due ECDF coincidono → distribuzioni indistinguibili
- D = 1 → le due ECDF sono massimamente distanti → distribuzioni
  completamente disgiunte

### 4.2.3 Il p-value (concetto delicato)

Grande discussione dedicata al p-value, che è spesso frainteso. La
formulazione corretta:

> Il p-value è la probabilità di osservare una distanza D almeno
> grande quanto quella misurata, **assumendo che l'ipotesi nulla sia vera**
> (cioè che le due finestre vengano dalla stessa distribuzione).

Non è "la probabilità che le due distribuzioni siano uguali". Non è "la
probabilità che ci siamo sbagliati". È una **probabilità condizionata**:
"quanto sarebbe stato improbabile vedere quello che ho visto, se non
ci fosse davvero un cambiamento?"

- p-value alto → il D osservato è compatibile con il rumore → nessuna
  prova di drift
- p-value basso → il D osservato è troppo grande per essere caso →
  concludo che c'è drift

### 4.2.4 La soglia alpha

`alpha` è il **livello di significatività** scelto a priori. Se
`p-value < alpha`, si rifiuta l'ipotesi nulla e si dichiara drift.
Con `alpha = 0.05` accettiamo un tasso di falsi positivi massimo del 5%.

## 4.3 L'adattamento del KS allo streaming

**Decisione**: usare uno schema a **due finestre**:
- `reference`: finestra fissa dei primi N valori dello stream (rappresenta
  la "distribuzione attesa")
- `current`: finestra scorrevole degli ultimi N valori (rappresenta "cosa
  sta succedendo adesso")

**Perché la reference è fissa**: dopo una lunga discussione, si è deciso
che la reference deve essere **congelata** dopo il primo riempimento. Se
scorresse anche lei, il confronto perderebbe di senso — staremmo
confrontando "gli ultimi 100 valori" con "altri 100 recenti", perdendo il
punto di riferimento assoluto. Il drift graduale ci sfuggirebbe.

**Perché la current scorre come FIFO**: il `deque(maxlen=n)` di Python
implementa naturalmente il comportamento FIFO — quando arriva un nuovo
valore e la finestra è piena, il più vecchio esce automaticamente.

**Trade-off della dimensione delle finestre**:
- Finestre grandi → potere statistico alto, ma reattività bassa (drift
  rilevato tardi)
- Finestre piccole → reattività alta, ma più falsi positivi

**Valore scelto di default**: `window_size = 200`. Poi è stato reso
configurabile via `.env`.

## 4.4 L'implementazione: `KSStrategy`

Vedi `detectors/ks_strategy.py`. La logica è:

- `update(value)`: se la reference non è piena, aggiungi lì; altrimenti
  aggiungi alla current (FIFO automatico)
- `detect()`: se la current non è ancora piena, restituisci
  `warming_up`; altrimenti esegui `scipy.stats.ks_2samp` fra reference e
  current, confronta il p-value con alpha, restituisci il verdetto

**Punto sottile**: `detect()` restituisce sempre un `DriftResult`, mai
`None`. Anche durante il warming up. Questo evita al chiamante di dover
gestire un caso speciale.

## 4.5 Perché scipy e non implementazione manuale

**Decisione**: usare `scipy.stats.ks_2samp` invece di implementare a mano
il calcolo di D e del p-value.

**Perché**: scipy è la libreria standard di riferimento in Python per la
statistica. Il valore aggiunto della tesi è nell'architettura, non nel
reimplementare test statistici classici. `scipy.stats.ks_2samp`
restituisce sia la statistica D sia il p-value in una singola chiamata.

## 4.6 Validazione su dataset sintetico Bernoulli

**Setup**: 3 stream Bernoulli, 2000 campioni ciascuno:
- `f0_stable`: p=0.2 costante
- `f1_drift`: p=0.8 per i primi 1000 passi, poi p=0.4 (cambio brusco)
- `f2_stable`: p=0.5 costante

Replica dello scenario della **Figura 2 del paper di Bifet & Gavaldà 2006**,
che è un benchmark canonico per la validazione dei detector di drift a
cambio brusco.

**Risultato**: KS rileva il drift solo su `f1_drift`, al passo t≈1100,
con una latenza di circa 100 passi. Nessun falso positivo sulle feature
stabili. Il ritardo è coerente con la teoria: la current window scorrevole
deve popolarsi di valori "nuovi" prima che il p-value scenda sotto alpha.

**Osservazione critica**: sono stati emessi warning di scipy sul fatto
che il metodo esatto per il calcolo del p-value non è applicabile su
dati binari (che sono l'output di una Bernoulli) — scipy fa fallback
sull'approssimazione asintotica. Questo è un primo indizio pratico del
fatto che il **KS non è ottimale sui dati discreti/binari**, tema che
riemergerà su ELEC2.

---

# 5. Fase 4 — Il FeatureDriftDetector

## 5.1 Il problema che risolve

Una singola `KSStrategy` (o qualunque altra strategia) lavora su
**un solo stream numerico**. Ma in un dataset ML tipico ci sono più
feature. Serve un componente che:

1. Riceva un vettore multi-feature
2. Smisti ogni componente alla strategia giusta
3. Aggreghi i verdetti individuali in un unico verdetto globale

## 5.2 Il design

**Decisione**: creare la classe `FeatureDriftDetector` che riceve una
**classe di strategia** (non un'istanza) più il numero di feature, e
istanzia internamente **N copie indipendenti** della strategia.

```python
FeatureDriftDetector(
    strategy_cls=KSStrategy,       # o ADWINStrategy, o altro
    n_features=8,
    k=1,                           # soglia di aggregazione
    feature_names=[...],           # opzionali per output leggibile
    **strategy_kwargs              # passati alle istanze di strategia
)
```

**La discussione emersa**: perché ricevere `strategy_cls` invece di
un'istanza pre-costruita? Perché così il detector può crearsi **N
istanze indipendenti** delle strategie, una per feature. Passare una
sola istanza costringerebbe l'utente a costruirsi le N istanze
manualmente prima di passarle.

## 5.3 Il parametro k — regola di aggregazione

**Decisione**: la regola di aggregazione è "drift globale se **almeno k
feature** hanno drittato".

**Perché**:

- **k=1** (default): sensibile — basta una feature drittata per allarme
  globale
- **k=n/2**: conservativo — richiede la maggioranza
- **k=n**: molto conservativo — tutte le feature devono drittare

Questo parametro rende il detector calibrabile in base allo scenario
operativo. Nei nostri esperimenti abbiamo usato sempre k=1 per la
sensibilità massima.

## 5.4 L'output aggregato

Il `FeatureDriftDetector` restituisce un unico `DriftResult` che
riassume tutte le strategie. Il campo `metadata` contiene:

- `drifted_features`: lista dei nomi (o indici) delle feature che hanno
  drittato in questo update
- `n_drifted`: numero totale di feature drittate
- `k_threshold`: la soglia usata (per tracciabilità)

---

# 6. Fase 5 — La strategia ADWIN

## 6.1 Il contesto

Dopo aver validato KS su Bernoulli, ci si è chiesti se aggiungere una
seconda strategia. La scelta è ricaduta su **ADWIN** (ADaptive
WINdowing) di Bifet e Gavaldà, per diversi motivi:

1. È uno **standard nella letteratura** dello streaming ML
2. È stato validato dagli stessi autori del paper che stavamo usando
   come riferimento (Bifet & Gavaldà 2006)
3. Offre **garanzie formali** su falsi positivi e falsi negativi (raro
   fra i drift detector)
4. È molto **complementare al KS**: KS guarda la forma della
   distribuzione, ADWIN guarda la media

## 6.2 Le due versioni di ADWIN

Discussione importante: il paper presenta **ADWIN base** (concettuale)
e **ADWIN2** (efficiente).

- **ADWIN base**: memorizza tutti i valori esplicitamente e prova tutti
  i possibili tagli. Costo `O(W)` in memoria e tempo per update.
  Inutilizzabile su stream lunghi.
- **ADWIN2**: usa una struttura a **istogrammi esponenziali** (Datar et
  al., 2002) che riduce a `O(M · log(W/M))` la memoria e `O(log W)` il
  tempo per update, mantenendo le stesse garanzie statistiche.

**Decisione**: usiamo ADWIN2 (che è quello che tutte le implementazioni
pratiche usano, inclusa quella di river).

## 6.3 La struttura a bucket esponenziali — spiegazione ricostruita

Grande discussione pedagogica per capire come funziona la struttura
interna di ADWIN2:

- Ogni **bucket** contiene due informazioni: **capacità** (numero di
  valori rappresentati, potenza di 2) e **contenuto** (somma dei
  valori). I singoli valori non vengono conservati.
- Il parametro **M** (default 5) è il **numero massimo di bucket
  per ogni dimensione** 2^i, non il numero totale di bucket. Questo
  punto ha richiesto chiarimenti espliciti.
- Quando arrivano nuovi valori: si crea un bucket di dimensione 1;
  se ci sono M+1 bucket della stessa dimensione, i due più vecchi si
  fondono in un bucket di dimensione doppia. Il meccanismo si propaga
  a cascata.
- Con `M=5`, su 100.000 valori la struttura totale contiene circa 70
  bucket. Ordine di grandezza logaritmico.

## 6.4 Il meccanismo del taglio

ADWIN, ad ogni nuovo valore, prova i possibili **tagli ai confini dei
bucket** (non fra singoli valori). Per ogni taglio candidato:

1. Divide la finestra in `W_0` (parte vecchia) e `W_1` (parte nuova)
2. Calcola le medie `μ(W_0)` e `μ(W_1)`
3. Se `|μ(W_0) − μ(W_1)| > ε_cut`, scarta `W_0` e segnala drift

## 6.5 La formula ε_cut (Equazione 3.1 del paper)

Discussione lunga sul ruolo esatto di `delta`. La formula corretta è:

```
ε_cut = √( (2/m) · σ²_W · ln(2/δ') ) + (2/3m) · ln(2/δ')
```

dove:
- **δ (delta)**: parametro di confidenza scelto dall'utente
- **δ' = δ / ln(n)**: correzione per il multiple hypothesis testing
- **m**: media armonica delle dimensioni di `W_0` e `W_1`
- **σ²_W**: varianza osservata nella finestra

**Chiarimento fondamentale emerso**: `delta` **non viene mai confrontato
direttamente** con la differenza delle medie. `delta` entra dentro la
formula come parametro che modula la soglia adattiva `ε_cut`. La regola
di decisione confronta `|μ(W_0) − μ(W_1)|` con `ε_cut`, non con `delta`.

Interpretazione di `delta`:
- Piccolo (default 0.002) → soglia più alta → detector più prudente
- Grande (es. 0.05) → soglia più bassa → detector più sensibile

## 6.6 Il Teorema 3.1

Il paper dimostra formalmente due garanzie:

1. **Bound sui falsi positivi**: se la distribuzione è stazionaria, la
   probabilità che ADWIN segnali drift in un dato passo è al massimo
   `delta`.
2. **Bound sui falsi negativi**: se esiste una partizione dove la
   differenza delle medie supera `2·ε_cut`, ADWIN la rileva con
   probabilità almeno `1 − delta`.

Queste garanzie sono **non asintotiche** e valgono ad ogni passo
temporale. È una proprietà distintiva di ADWIN rispetto ad altri
detector (DDM, EDDM, Page-Hinkley) che sono giustificati solo
empiricamente.

## 6.7 Perché wrappare river e non reimplementare

**Decisione**: `ADWINStrategy` è un thin wrapper attorno a
`river.drift.ADWIN`, non un'implementazione da zero.

**Perché**: reimplementare ADWIN2 a mano richiederebbe 150-200 righe
di codice intricato (bucket, fusioni, formula ε_cut). river ha
un'implementazione **testata, ottimizzata e fedele al paper**. Il
valore della tesi non è nella reimplementazione — anzi, wrappare river
è una scelta di design difendibile che dimostra maturità ingegneristica.

## 6.8 Differenze comportamentali rispetto al KS

- **Segnale a impulso vs continuo**: ADWIN segnala `drift_detected=True`
  **solo nell'istante del taglio**. Al successivo update torna False (a
  meno di un altro taglio). Il KS invece resta True finché reference e
  current restano diverse.
- **Sensibilità solo alla media**: ADWIN non vede cambi di varianza o
  forma della distribuzione, solo cambi di media. Il KS invece vede
  l'intera forma via ECDF.
- **Dati binari**: ADWIN è nativo per stream binari. KS su binari perde
  potere statistico.

## 6.9 Validazione su Bernoulli sintetico

Sullo stesso scenario del KS (`f1_drift` cambia p da 0.8 a 0.4 al passo
1000), ADWIN rileva il drift a t=1055 — latenza di soli **55 passi**,
circa la metà del KS. Un solo segnale, coerente con la natura impulsiva.

---

# 7. Fase 6 — Prediction e Concept Drift Detector

## 7.1 Chiarimento sui tre tipi di drift

Discussione articolata sui tre tipi di drift e su cosa serve per rilevarli:

| Tipo | Cosa cambia | Cosa serve |
|---|---|---|
| Feature | `P(X)` | Solo le feature; niente modello |
| Prediction | `P(Ŷ)` | Le predizioni del modello |
| Concept | `P(Y\|X)` | Predizioni + `y_true` (ground truth) |

**Insight emerso**: il **feature drift è completamente indipendente dal
modello**. Il framework può monitorarlo anche senza avere alcun modello
addestrato. Prediction e concept drift invece richiedono il modello (e
il concept drift richiede anche `y_true`).

## 7.2 Il problema del label delay nel concept drift

Discussione importante per motivare i limiti del concept drift in
produzione reale:

- **Scenario ideale**: `y_true` disponibile immediatamente (es.
  raccomandazioni, trading). Il concept drift funziona perfettamente in
  streaming.
- **Scenario reale**: `y_true` disponibile con **ritardo** (label delay).
  Esempi: fraud detection (settimane), credit scoring (mesi). Il concept
  drift funziona ma con lag strutturale.
- **Scenario problematico**: `y_true` **mai disponibile**. Il concept
  drift classico non è applicabile. Servono approcci alternativi
  (uncertainty-based, pseudo-labels).

**Implicazione per la tesi**: il feature drift e il prediction drift
(che non richiedono `y_true`) sono operativamente più affidabili in
scenari MLOps reali. Il concept drift è il gold standard ma spesso
inutilizzabile senza label delay.

## 7.3 Il design dei due nuovi detector

**`PredictionDriftDetector`**: wrapper minimale che riceve un singolo
stream (le predizioni `y_pred`) e lo passa a una strategia. Riclassifica
il tipo di drift come `"prediction"` nel `DriftResult`.

**`ConceptDriftDetector`**: differente perché il suo input è **la coppia
`(y_pred, y_true)`**. Internamente calcola l'errore binario
`error = 1 if y_pred != y_true else 0` e lo passa alla strategia. Questo
mantiene l'interfaccia della strategia (che lavora sempre su un singolo
valore alla volta) senza sporcarla di logica di errore.

## 7.4 La scelta di limitare KS al feature/prediction, ADWIN a tutti e tre

**Decisione**: il concept drift viene rilevato **solo con ADWIN**, non
con KS. Ragione:

- Il concept drift lavora su uno stream binario (errori 0/1)
- Il KS su binari perde potere statistico (discussione precedente)
- ADWIN è nativo per binari

Includere KS sul concept drift avrebbe prodotto solo rumore. Includere
solo ADWIN dà un risultato pulito.

---

# 8. Fase 7 — Il classificatore GaussianNB

## 8.1 La scelta del classificatore

**Decisione**: usare `sklearn.naive_bayes.GaussianNB` come classificatore
per gli esperimenti su ELEC2.

**Perché**:

- **Fedeltà ai paper di riferimento**: sia Bifet & Gavaldà 2006 (Sezione
  5.3) sia dos Reis et al. 2016 usano Naive Bayes su ELEC2. Usarlo
  garantisce confrontabilità.
- **È incrementale per natura** (anche se noi lo useremo statico)
- **Non richiede tuning di iperparametri** — non introduce rumore
  sperimentale
- **È veloce**: non aggiunge overhead computazionale allo streaming

Non ci interessa "il modello migliore possibile" per ELEC2 — ci interessa
un modello ragionevole su cui misurare il drift.

## 8.2 Il concetto: cosa fa GaussianNB

**Naive Bayes** è un classificatore probabilistico basato sul teorema di
Bayes:

```
P(classe | x) ∝ P(classe) · P(x | classe)
```

Il "naive" si riferisce all'assunzione (falsa in generale, ma spesso
efficace) che le feature siano **condizionalmente indipendenti** data
la classe.

**Gaussian** significa che si assume che ogni feature all'interno di una
classe sia **distribuita normalmente**. Il modello memorizza quindi
media e varianza di ogni feature per ogni classe.

## 8.3 Cosa succede internamente (spiegazione pedagogica sviluppata)

**Durante `fit(X, y)`**:
1. Per ogni classe c, calcola:
   - `class_prior_[c]` = frazione di esempi di classe c
   - `theta_[c]` = media di ogni feature sui campioni di classe c
   - `var_[c]` = varianza di ogni feature sui campioni di classe c
2. Nient'altro. Non ci sono pesi, non c'è ottimizzazione.

**Durante `predict(x)`**:
1. Calcola internamente `_joint_log_likelihood(x)`, che è un
   **vettore** con un log-score per ogni classe. La formula è
   `log P(classe) + Σ log P(x_i | classe)`, dove `P(x_i | classe)` è
   la densità gaussiana valutata in `x_i`.
2. Applica `argmax` sul vettore → restituisce l'indice della classe più
   probabile.

**Alternativa**: `predict_proba(x)` applica la softmax invece di argmax,
restituendo le probabilità di ciascuna classe.

## 8.4 La scelta "statica" del classificatore

**Decisione**: il classificatore viene addestrato **una volta** sui primi
500 campioni di ELEC2 e **mai riaddestrato** durante lo streaming.

**Perché**: è cruciale per far emergere il concept drift. Se il
classificatore si aggiornasse continuamente, si adatterebbe al drift e
lo nasconderebbe al detector. Un modello statico invece degrada
visibilmente quando la relazione input-output cambia, ed è quello che
vogliamo osservare.

È lo stesso setup della colonna "statico" delle Tabelle 9-10 del paper
Bifet & Gavaldà.

## 8.5 La domanda emersa: "quali sono le y_true?"

Una discussione pedagogica ha chiarito che ELEC2 è un dataset
**supervisionato completo**: ogni riga ha sia le 8 feature sia la label
(`class` = UP/DOWN). Non c'è alcuno split "dentro" i 500 campioni di
training: tutti e 500 sono usati per addestrare.

Le `y_true` per la fase di streaming sono semplicemente le label delle
righe successive nel dataset — sono già lì, non le inventiamo.

---

# 9. Fase 8 — L'esperimento su ELEC2

## 9.1 Perché ELEC2

Discussione articolata sui possibili dataset reali:

- **ELEC2** (mercato elettrico australiano): usato dal paper ADWIN
  (Bifet & Gavaldà 2006, Sez. 5.3) **e** dal paper dos Reis 2016 sul
  KS. Coincidenza fortunata: **un solo dataset copre entrambi i
  riferimenti**.
- **Airlines**, **Forest Cover Type**, **PowerSupply**: candidati per
  future estensioni.

**Decisione**: iniziare con ELEC2. Dataset scaricato da Kaggle
(`yashsharan/the-elec2-dataset`), file `data/electricity-normalized.csv`
di 3 MB, 45.312 istanze, 8 feature numeriche + label `UP`/`DOWN`.

## 9.2 Le feature di ELEC2

| Feature | Descrizione |
|---|---|
| `date` | Timestamp normalizzato 0-1 (monotona crescente su 2.5 anni) |
| `day` | Giorno della settimana (1-7) |
| `period` | Periodo di 30 min all'interno del giorno (1-48) |
| `nswprice` | Prezzo elettricità NSW (normalizzato) |
| `nswdemand` | Domanda elettricità NSW |
| `vicprice` | Prezzo elettricità Victoria |
| `vicdemand` | Domanda elettricità Victoria |
| `transfer` | Trasferimento programmato fra stati |

## 9.3 Il concetto di stazionarietà — spiegazione emersa

Una discussione ha chiarito un concetto chiave per interpretare i
risultati:

- **Feature stazionaria**: le sue proprietà statistiche non cambiano nel
  tempo. Se prendi una finestra all'inizio e una alla fine, sono simili.
- **Feature non stazionaria**: le proprietà cambiano. Le due finestre
  sono diverse.

**Esempi da ELEC2**:

- `period` è **stazionaria per costruzione**: cicla 1-48 ogni giorno.
  Qualsiasi finestra di 200 campioni contiene ~4 giorni di dati, quindi
  copre tutti i 48 valori possibili uniformemente. Distribuzione uguale
  in ogni finestra.
- `date` è **non stazionaria per costruzione**: cresce monotonicamente
  da 0 a 1. Finestre in punti diversi contengono intervalli
  completamente diversi. **Falso positivo strutturale** per il drift
  detector — è un drift "vero" statisticamente ma inutile
  operativamente (sappiamo già che il tempo passa).
- `nswprice`, `nswdemand`, `vicprice`, ecc.: **non stazionarie
  realistiche** — hanno trend e stagionalità reali di mercato.

## 9.4 Setup dell'esperimento

- Classificatore Naive Bayes addestrato sui primi 500 campioni
- Streaming dei restanti 44.812 campioni
- Detector eseguiti in parallelo:
  - `FeatureDriftDetector` con KS e con ADWIN (uno per esperimento)
  - `PredictionDriftDetector` con KS e con ADWIN
  - `ConceptDriftDetector` con solo ADWIN

## 9.5 Risultati KS su ELEC2

**Feature drift**: 44.413 segnalazioni su 44.812 (99.1% dei passi utili).

Distribuzione per feature:
- `date`: 100% (falso positivo strutturale)
- `period`: 0% (correttamente stazionaria)
- Le altre 6: 62-99%

**Interpretazione**:

Il framework funziona (period=0 conferma), ma con parametri standard
(window=200, alpha=0.05) il KS è **troppo sensibile** su un dataset
reale così non-stazionario. La per-feature analysis fornisce comunque
informazione diagnostica utile, ma l'aggregato è troppo denso per uso
operativo diretto.

**Prediction drift**: 44.180 segnalazioni (98.6%). Sostanzialmente
inutile. Warning continui di scipy sul fallback all'approssimazione
asintotica. Evidenza empirica del fatto che il **KS non è adatto a
stream binari**.

## 9.6 Risultati ADWIN su ELEC2

**Feature drift**: 558 segnalazioni totali (~80× meno del KS).

Distribuzione per feature:
- `date`: 11 (contro le 44.413 del KS — ADWIN taglia una volta e non
  urla più)
- `nswprice`: 10 (contro 44.373 del KS)
- `period`: 0 (concorde con KS)
- **`vicprice`: 0** ← discussione centrale
- Le altre: da 45 a 467

Il caso `vicprice = 0 per ADWIN vs 27.856 per KS` è il **momento
più significativo** dell'intero esperimento (vedi discussione dedicata
sotto).

**Prediction drift**: solo **2 segnalazioni** (t=723 e t=1075),
all'inizio dello streaming. Dopo l'assestamento del modello statico, le
predizioni restano distribuite in modo stabile per il resto del flusso.
Nota critica: **stabilità della distribuzione delle predizioni ≠
correttezza del modello**. Le predizioni sono stabili anche quando il
modello sta sbagliando sistematicamente.

**Concept drift**: **66 segnalazioni** distribuite lungo il flusso.
Primo segnale al passo t=1619, che correla temporalmente con il crollo
di accuracy dal 83% (t=1499) al 58% (t=2499). Cluster di segnalazioni
nei periodi di forte instabilità del classificatore (es. t=2931-3891,
t=27091-27763, t=44371-44979).

## 9.7 Il caso `vicprice` — la scoperta più importante

**Osservazione**: sul KS `vicprice` risulta drittata al 62% dei passi.
Su ADWIN è a 0.

**Spiegazione**: le due tecniche misurano cose diverse.

- Il **KS misura la forma della distribuzione** (via ECDF). Se la forma
  cambia, il KS lo vede.
- L'**ADWIN misura la media** dello stream. Se la media resta stabile,
  ADWIN non vede nulla.

Quindi la distribuzione di `vicprice` **cambia forma nel tempo ma
mantiene la stessa media**. Entrambi gli algoritmi hanno ragione dal
loro punto di vista.

**Implicazione**: KS e ADWIN non sono in competizione — sono
**complementari**. Un framework maturo li affianca entrambi, offrendo
due lenti diverse sul drift.

Questa osservazione è **la gemma sperimentale della tesi**: dimostra
empiricamente il valore dell'architettura multi-strategia.

## 9.8 Correlazione fra ADWIN concept drift e accuracy del classificatore

Confrontando i 66 punti di segnalazione del `ConceptDriftDetector` con
la timeline dell'accuracy (media mobile su 1000 campioni):

- Primo segnale a t=1619 → accuracy passa da 83% (t=1499) a 58%
  (t=2499). ADWIN anticipa il crollo.
- Cluster di 4 segnali a t=2931-3891 → accuracy ~55%.
- Cluster finale a t=44371-44979 → accuracy scesa a 43-45%
  (fase peggiore dell'intero flusso).

**Conclusione**: ADWIN sul concept drift **funziona come da Teorema
3.1**. I punti di segnalazione **correlano con eventi statistici
concreti** nel comportamento del modello.

---

# 10. Fase 9 — Analisi trasversale dei risultati

## 10.1 KS e ADWIN sono complementari, non alternativi

Insight centrale emerso da tutti gli esperimenti:

| Aspetto | KS | ADWIN |
|---|---|---|
| Cosa misura | Forma della distribuzione | Media |
| Volume di segnalazioni | Alto | Basso |
| Dati continui | Ottimo | Buono |
| Dati binari | Sub-ottimo | Nativo |
| Segnale nel tempo | Continuo | A impulso |
| Latenza | Alta | Bassa |
| Garanzie formali | Test KS classico | Teorema 3.1 |

## 10.2 La proposta operativa emersa

Dalla discussione dei risultati è emersa una proposta operativa
concreta:

- **ADWIN come detector di allarme primario in tempo reale**: basso
  volume di segnali, alta significatività statistica. Ogni segnalazione
  può ragionevolmente diventare un evento nel sistema di monitoring.
- **KS come strumento di analisi diagnostica offline**: esplorazione di
  quale feature specifica sta contribuendo al drift, dove il rumore
  aggregato del KS non è un problema perché stiamo indagando
  analiticamente.

Le due strategie **coesistono nella stessa pipeline**, non si
escludono.

## 10.3 Sulla stazionarietà e i falsi positivi strutturali

Il caso `date` (100% drift) è un esempio di **falso positivo
strutturale**: la feature è genuinamente non stazionaria (drift statistico
reale) ma il "drift" non è informativo (sappiamo già che il tempo passa,
il dataset ha un ordinamento temporale).

**Implicazione per la tesi**: la **selezione delle feature da monitorare
è essa stessa una scelta di design**. Un framework maturo dovrebbe
consentire di escludere le feature temporali dal drift monitoring.
Nella nostra implementazione non l'abbiamo fatto (le monitoriamo tutte)
proprio per esporre questo comportamento come discussione metodologica.

## 10.4 Sul limite del prediction drift per il monitoring del modello

Osservazione critica emersa: **la stabilità della distribuzione delle
predizioni non implica che il modello sia corretto**. Nel nostro
esperimento, le predizioni del Naive Bayes statico sono rimaste
stabilmente distribuite (ADWIN prediction ha rilevato solo 2 drift
all'inizio), **mentre l'accuracy è crollata dal 71% al 55%**.

Il prediction drift monitora la **distribuzione dell'output** — se il
modello sbaglia in modo consistente (stessa distribuzione, ma
sistematicamente errata), il prediction drift non lo vede. Serve il
concept drift.

**Conclusione**: i tre tipi di drift sono **davvero complementari**.
Nessuno da solo copre tutto. Solo aggregandoli si ha una visione
completa dello stato del sistema.

---

# 11. Fase 10 — Esternalizzazione dei parametri

## 11.1 Il problema

Nella prima versione, i parametri degli esperimenti (window_size, alpha,
delta, train_size) erano **hardcoded** all'inizio di ogni script
sperimentale. Cambiarli richiedeva editing del codice.

## 11.2 La soluzione: file `.env` + `config.py`

**Decisione**: esternalizzare tutti i parametri configurabili in un file
`.env` alla radice del progetto, caricato da `config.py` che li espone
agli script con i tipi corretti (int/float/str).

**Perché `.env` e non altre alternative**:

- **YAML**: più espressivo ma richiede la libreria PyYAML come
  dipendenza aggiuntiva e non è così "standard MLOps"
- **JSON**: nativo Python ma meno human-friendly
- **argparse (CLI)**: buono per override puntuali ma scomodo per
  set completi di parametri
- **`.env`**: standard industriale per MLOps, semplice da editare, si
  integra con Docker/CI/CD

**Perché committare il `.env` in repo**: contiene **solo parametri di
esperimento** (non segreti). Committarlo garantisce **riproducibilità**
del setup dell'esperimento. Chiunque clona la repo può rilanciare gli
esperimenti con gli stessi parametri usati per i risultati salvati.

Se in futuro il progetto avrà segreti reali (API key, credenziali DB),
si aggiungerà un `.env.local` gitignored per quelli.

## 11.3 Cosa è configurabile

Vedi `.env` per la lista completa. In sintesi:
- Path del dataset ELEC2
- Parametri esperimento Bernoulli (N_samples, drift_point, seed)
- Parametri esperimento ELEC2 (train_size)
- Parametri KS (window_size, alpha)
- Parametri ADWIN (delta)
- Parametri FeatureDriftDetector (k_threshold)

---

# 12. Riferimenti bibliografici emersi

Paper e libri effettivamente discussi o citati durante il lavoro:

## Paper primari
- **Bifet, A., Gavaldà, R. (2006/2007)** — *Learning from Time-Changing
  Data with Adaptive Windowing*. Riferimento principale di ADWIN.
  Contiene teoria, teorema 3.1, esperimenti su ELEC2 (Tabelle 9-10).
- **dos Reis, D. M., Flach, P., Matwin, S., Batista, G. (2016)** —
  *Fast Unsupervised Online Drift Detection Using Incremental
  Kolmogorov-Smirnov Test*. KDD 2016. Riferimento per l'uso del KS in
  drift detection streaming.

## Paper di background
- **Gama, J. et al. (2014)** — *A Survey on Concept Drift Adaptation*.
  Riferimento fondamentale per la tassonomia del drift.
- **Lu, J. et al. (2019)** — *Learning under Concept Drift: A Review*.
  Aggiornamento sistematico dello stato dell'arte.
- **Sethi, T., Kantardzic, M. (2018)** — *On the reliable detection of
  concept drift from streaming unlabeled data*. Discussione dei metodi
  error-based (DDM, EDDM, Page-Hinkley).
- **Baier, L. et al. (2020)** — *Detecting Concept Drift With Neural
  Network Model Uncertainty*.

## Struttura dati e algoritmi
- **Datar, M., Gionis, A., Indyk, P., Motwani, R. (2002)** —
  *Maintaining stream statistics over sliding windows*. Riferimento
  per gli istogrammi esponenziali usati in ADWIN2.

## Software e librerie
- **Montiel, J. et al. (2021)** — *River: machine learning for
  streaming data in Python*. JMLR. Libreria usata per ADWIN.
- **Virtanen, P. et al. (2020)** — *SciPy 1.0*. Libreria usata per KS
  (`scipy.stats.ks_2samp`).

## Design software
- **Gamma, E., Helm, R., Johnson, R., Vlissides, J. (1994)** —
  *Design Patterns: Elements of Reusable Object-Oriented Software*.
  Riferimento canonico per il Strategy Pattern.

## Documentazione tecnica MLOps
- MLflow, Argo Workflows, Alibi Detect, Evidently AI, Docker —
  citati dalla proposta di tesi come strumenti dell'ecosistema
  MLOps con cui il framework dovrà integrarsi in prospettiva.

---

# 13. Cosa manca e prossimi passi

## 13.1 Componenti architetturali non ancora implementati

- **`DriftMonitoringService`**: l'orchestratore centrale che collega
  modello e detector in un flusso unificato. Attualmente esiste solo
  come cartella `monitoring/drift_monitoring_service.py` vuota.
- **Strategie aggiuntive**: Jensen-Shannon Divergence (per feature e
  prediction drift), DDM e Page-Hinkley (per concept drift come
  alternative ad ADWIN).

## 13.2 Esperimenti mancanti

- **Sweep dei parametri**: usare la parametrizzazione `.env` per
  esplorare come cambiano i risultati al variare di `window_size`,
  `alpha`, `delta`. Utile per giustificare le scelte di default in
  tesi.
- **Rilancio dei Bernoulli con nuova configurazione**: dopo l'ultimo
  commit sull'esternalizzazione parametri, i file di output dei due
  esperimenti Bernoulli in `results/KS/` e `results/ADWIN/` non sono
  ancora stati rigenerati.
- **Secondo dataset reale**: Airlines o Forest Cover Type, per
  validare che i risultati di ELEC2 siano generalizzabili.
- **Modello diverso**: provare oltre a Naive Bayes anche un modello
  più complesso (es. Random Forest, Logistic Regression) per verificare
  la model-agnosticità in modo empirico.

## 13.3 Integrazione MLOps

Ancora tutta da fare, prevista per le fasi finali del progetto:
- Integrazione con MLflow per il tracking degli esperimenti
- Orchestrazione via Argo Workflows
- Containerizzazione con Docker
- Design della pipeline end-to-end con trigger di retraining

## 13.4 Redazione tesi

- **Capitolo 1**: già scritto (`thesis/chapter1_introduzione.tex`)
- **Capitolo 2-8**: da scrivere, con materiale già presente in
  `docs/thesis_writing_briefing.md` e `docs/adwin_overview.md`

---

# 14. Come usare questo documento per la tesi

## 14.1 Struttura dei file di supporto disponibili

Il progetto contiene già una serie di documenti che coprono aspetti
diversi del lavoro:

- **`docs/percorso_tesi.md`** (questo file) — narrazione, ragionamenti,
  storia delle scelte
- **`docs/thesis_writing_briefing.md`** — briefing tecnico strutturato
  per capitolo con contenuti, formule, code snippet, chiavi bibtex
- **`docs/thesis_outline.md`** — struttura dei capitoli della tesi
  (indice ragionato)
- **`docs/adwin_overview.md`** — deep-dive su ADWIN, materiale pronto
  per il capitolo dedicato
- **`docs/slides_briefing.md`** — versione compatta per slide di
  presentazione
- **`docs/flow_diagrams_briefing.md`** — descrizione dei diagrammi di
  flusso di KS e ADWIN
- **`results/analisi_risultati.md`** — analisi dettagliata dei
  risultati sperimentali
- **`thesis/chapter1_introduzione.tex`** — Capitolo 1 già scritto
- **`HANDOFF.md`** — sintesi essenziale del progetto per riprendere in
  chat future

## 14.2 Come combinare i documenti quando si scrive la tesi

Ordine consigliato di consultazione quando si scrive un capitolo:

1. **`docs/thesis_outline.md`** per capire la struttura del capitolo
2. **`docs/thesis_writing_briefing.md`** per il contenuto tecnico
   specifico
3. **`docs/percorso_tesi.md`** (questo file) per il ragionamento e
   la giustificazione delle scelte da inserire nel testo
4. **`docs/adwin_overview.md`** e **`results/analisi_risultati.md`**
   per approfondimenti specifici quando serve

## 14.3 Se dai questo file a un LLM per scrivere

Se questo documento viene passato a un LLM per assistere nella
scrittura:

1. L'LLM deve **leggere l'intero documento** prima di scrivere
2. Deve mantenere **stile italiano accademico ma leggibile** (vedi
   `thesis/chapter1_introduzione.tex` come esempio di stile già
   validato)
3. Deve usare le **chiavi bibtex** definite in
   `docs/thesis_writing_briefing.md` per le citazioni
4. Deve rispettare i **limiti** discussi (es. concept drift solo con
   ADWIN, KS non testato su concept drift) senza aggiungere risultati
   inventati
5. Le decisioni di design (Strategy Pattern, model-agnosticità,
   classificatore statico) sono **giustificate qui** — l'LLM può
   riprenderle e rielaborarle, non ha bisogno di reinventarle
6. Il **caso `vicprice`** è il momento più memorabile — dedicare
   uno spazio adeguato nel capitolo di validazione sperimentale

## 14.4 Cose che l'LLM deve evitare

- **Non inventare risultati** che non sono qui documentati
- **Non usare toni troppo enfatici** (l'utente preferisce prosa
  sobria e diretta)
- **Non introdurre concetti** non discussi qui senza segnalare che
  è materiale nuovo da validare
- **Non contraddire** le decisioni di design documentate

---

**Fine documento.** Buon lavoro con la scrittura della tesi.
