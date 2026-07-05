# Briefing di scrittura della tesi — self-contained

Documento di **passaggio di consegne per la scrittura della tesi** di Alberto
Conti. È pensato per essere passato a un LLM che deve produrre capitoli della
tesi, sia nella modalità *"scrivi tutto in una passata"* sia nella modalità
*"un capitolo alla volta man mano che aggiungiamo lavoro"*. Il documento è
**auto-contenuto**: contiene tutte le informazioni tecniche, i risultati
sperimentali, i riferimenti bibliografici e le convenzioni di scrittura
necessarie.

**Ultimo aggiornamento**: 2026-06-16

---

# PARTE 0 — Come usare questo documento

## 0.1 Obiettivo

Questo briefing serve a **scrivere la tesi in LaTeX** per il corso di Laurea
Magistrale in Ingegneria Informatica, curriculum Intelligenza Artificiale,
Università degli Studi di Palermo.

L'LLM che riceve questo documento deve:

1. Leggere **tutte le parti** prima di iniziare a scrivere
2. Rispettare il format tecnico (Parte 2) e lo stile (Parte 3)
3. Attenersi al contenuto specifico di ciascun capitolo (Parte 5)
4. Usare le key bibtex definite (Parte 4) per le citazioni

## 0.2 Modalità d'uso

Due modalità tipiche:

- **Modalità A — scrittura completa**: "Scrivi tutti i capitoli marcati come
  ✅ o 🟡 nella Parte 5, in file LaTeX separati (`chapter2.tex`,
  `chapter3.tex`, …). Segui esattamente il contenuto specificato."

- **Modalità B — capitolo singolo**: "Sulla base di questo briefing, scrivi
  solo il Capitolo X. Segui esattamente il contenuto della sezione 5.X."

Il documento va **aggiornato manualmente** man mano che il lavoro di tesi
avanza: quando si implementa un nuovo detector o si esegue un nuovo
esperimento, si aggiungono i risultati alla sezione corrispondente della
Parte 5.

---

# PARTE 1 — Metadati

- **Titolo**: *Robust Monitoring and Drift Detection in Machine Learning
  Systems: Methods, Metrics, and MLOps Integration*
- **Autore**: Alberto Conti
- **Università**: Università degli Studi di Palermo
- **Corso di Studi**: Ingegneria Informatica Magistrale, curriculum
  Intelligenza Artificiale
- **Anno accademico**: 2025-2026
- **Relatore**: Prof. Marco La Cascia
- **Tutor aziendale**: Daniele Fakhoury, Engineering Ingegneria Informatica
  S.p.A.
- **CFU**: 24 (600 ore, 17 settimane di lavoro previste)

---

# PARTE 2 — Format tecnico

## 2.1 Lingua e strumenti

- **Lingua della tesi**: italiano
- **Editor**: LaTeX
- **Class LaTeX suggerita**: `report` o `book` (verificare con il relatore
  se esiste un template Unipa specifico per Ingegneria)

## 2.2 Formattazione (norme Unipa)

- **Font**: Times New Roman
- **Dimensione corpo del testo**: 12 pt
- **Dimensione titoli di capitolo**: 14 pt grassetto, centrati
- **Dimensione note a piè di pagina**: 10 pt
- **Interlinea**: 1.5
- **Margini**: superiore/inferiore/destro 2 cm, sinistro 3.5 cm (per la
  rilegatura)
- **Allineamento**: giustificato
- **Rientro prima riga di ogni paragrafo**: 1 cm
- **Corsivo**: limitato a titoli di opere, vocaboli stranieri e di lingue
  morte
- **Enfasi**: preferire virgolette apicali (`"..."`) al corsivo
- **Citazioni**: fra virgolette a sergente (`«...»`)

## 2.3 Citazioni bibliografiche

**Stile**: IEEE numerato, in forma `[N]`. In LaTeX usare `\cite{key}` con
`\bibliographystyle{IEEEtran}` (o `\usepackage{cite}` per la gestione dei
range).

## 2.4 Lunghezza

**Massimo 120 pagine** totali, incluso frontespizio, abstract, indice e
bibliografia. Vincolo rigido.

## 2.5 Struttura del file principale

Struttura consigliata:

```
main.tex
├── frontespizio.tex
├── abstract.tex
├── \tableofcontents
├── \listoffigures
├── \listoftables
├── chapter1_introduzione.tex
├── chapter2_stato_arte.tex
├── chapter3_architettura.tex
├── chapter4_strategie.tex
├── chapter5_detector.tex
├── chapter6_validazione.tex
├── chapter7_mlops.tex
├── chapter8_conclusioni.tex
├── bibliografia.bib
└── appendici.tex
```

---

# PARTE 3 — Stile di scrittura

## 3.1 Tono

Accademico ma leggibile. Formale, in terza persona impersonale ("si
osserva", "il framework consente"). Evitare eccessi di formalismo tipo
"si è ritenuto opportuno" o "il presente elaborato". Preferire prosa
diretta.

## 3.2 Convenzioni terminologiche

- Termini tecnici in inglese al primo uso: **corsivi**, seguiti dalla
  spiegazione. Alle occorrenze successive: tondo (non corsivo).
- Nomi di software/librerie/classi: `\texttt{}` monospaziato
  (es. `\texttt{river.drift.ADWIN}`, `\texttt{KSStrategy}`)
- Formule matematiche: numerate solo se referenziate nel testo
- Riferimenti bibliografici: `\cite{key}` (mai `[Autore anno]` a mano)

## 3.3 Esempio di stile

Il **Capitolo 1** è già stato scritto e si trova in
`thesis/chapter1_introduzione.tex`. Usare quel file come **reference di
stile** per la scrittura dei capitoli successivi. Il tono, il registro,
la lunghezza dei paragrafi e l'uso del corsivo devono essere coerenti
con quello.

---

# PARTE 4 — Bibliografia con key bibtex

Ogni riferimento è preceduto dalla **chiave bibtex** da usare in
`\cite{}`. La bibliografia va poi costruita in un file `.bib`.

## 4.1 Articoli e atti di convegno

- `gama2014survey` — Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M.,
  Bouchachia, A. (2014). *A survey on concept drift adaptation*. ACM
  Computing Surveys 46(4), 44:1–44:37.
- `lu2019learning` — Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G.
  (2019). *Learning under concept drift: A review*. IEEE Transactions on
  Knowledge and Data Engineering 31(12), 2346–2363.
- `sethi2018reliable` — Sethi, T. S., Kantardzic, M. (2018). *On the
  reliable detection of concept drift from streaming unlabeled data*.
  Expert Systems with Applications 82, 77–99.
- `baier2020detecting` — Baier, L., Kellner, V., Kühl, N., Satzger, G.
  (2020). *Detecting concept drift with neural network model uncertainty*.
- `bifet2007adwin` — Bifet, A., Gavaldà, R. (2007). *Learning from
  time-changing data with adaptive windowing*. Proceedings of the 2007
  SIAM International Conference on Data Mining, 443–448.
- `bifet2006adwin` — Bifet, A., Gavaldà, R. (2006). *Learning from
  time-changing data with adaptive windowing*. Technical Report,
  Universitat Politècnica de Catalunya.
- `datar2002maintaining` — Datar, M., Gionis, A., Indyk, P., Motwani, R.
  (2002). *Maintaining stream statistics over sliding windows*. SIAM
  Journal on Computing 31(6), 1794–1813.
- `montiel2021river` — Montiel, J., Halford, M., Mastelini, S. M., Bolmier,
  G., Sourty, R., Vaysse, R., Zouitine, A., Gomes, H. M., Read, J.,
  Abdessalem, T., Bifet, A. (2021). *River: machine learning for streaming
  data in Python*. Journal of Machine Learning Research 22(110), 1–8.
- `virtanen2020scipy` — Virtanen, P., Gommers, R., Oliphant, T. E., et al.
  (2020). *SciPy 1.0: fundamental algorithms for scientific computing in
  Python*. Nature Methods 17, 261–272.

## 4.2 Libri

- `gof1994patterns` — Gamma, E., Helm, R., Johnson, R., Vlissides, J.
  (1994). *Design Patterns: Elements of Reusable Object-Oriented
  Software*. Addison-Wesley.

## 4.3 Documentazione tecnica e siti web

- `mlflow_docs` — MLflow Documentation. https://mlflow.org/
- `argo_docs` — Argo Project Documentation. https://argoproj.github.io/
- `alibi_detect_docs` — Alibi Detect Documentation.
  https://docs.seldon.ai/alibi-detect
- `evidently_docs` — Evidently AI Documentation. https://www.evidentlyai.com/
- `docker_docs` — Docker Documentation. https://docs.docker.com/

---

# PARTE 5 — Briefing capitolo per capitolo

Per ogni capitolo:

- **Titolo** e stato di scrivibilità
- **Contenuto**: bullet completi di **fatti, formule, numeri concreti,
  riferimenti** — non prosa da copiare, ma materiale sufficiente perché
  un LLM produca prosa di qualità
- **`\cite{}` da inserire dove**
- **Note stilistiche**

Stato:
- ✅ **PRONTO** — contenuto completo, si può scrivere ora
- 🟡 **PARZIALE** — alcune sezioni pronte, altre no
- ⏳ **PLACEHOLDER** — da riempire quando il lavoro sarà svolto

---

## 5.1 CAPITOLO 1 — Introduzione ✅

**Stato**: GIÀ SCRITTO. Vedere `thesis/chapter1_introduzione.tex`.
Usare quel file come riferimento di stile per gli altri capitoli.

---

## 5.2 CAPITOLO 2 — Stato dell'arte e background 🟡

**Pagine target**: 18-22

### 5.2.1 Sezione 2.1 — Machine Learning in produzione: il ciclo di vita di un modello ✅

**Contenuto da trattare**:
- Definire cos'è una *ML pipeline*: raccolta dati → preprocessing → training
  → validazione → deployment → serving → monitoring
- Distinguere fra *training* (fase offline) e *serving/inference* (fase
  online). In training i dati sono statici; in serving arrivano nel tempo
  come stream.
- Il problema fondamentale della **non-stazionarietà** dei dati in
  produzione: il modello è stato addestrato su una distribuzione, ma
  in produzione può ricevere dati da una distribuzione diversa.
- Citare `\cite{gama2014survey}` per la sistematizzazione del ciclo.

### 5.2.2 Sezione 2.2 — Il fenomeno del drift ✅

**Contenuto**:
- Definizione formale di drift: cambiamento nel tempo di una o più
  distribuzioni di probabilità associate al problema di apprendimento.
- Notazione: `P_t(X,Y)` per la distribuzione congiunta al tempo `t`.
  Se `P_t(X,Y) ≠ P_{t+Δt}(X,Y)` si parla di drift.
- Il drift può manifestarsi su:
  - `P(X)`: distribuzione delle feature di input
  - `P(Y)`: distribuzione della variabile target
  - `P(Y|X)`: relazione condizionale input-output
  - `P(X|Y)`: relazione input dato output (raro in pratica)
- Citare `\cite{gama2014survey, lu2019learning}` per la formalizzazione.

### 5.2.3 Sezione 2.3 — Tassonomia del drift ✅

**Sotto-sezioni**:

**2.3.1 Feature drift (data drift)** — `P(X)` cambia nel tempo.
Esempio: le caratteristiche demografiche degli utenti di un servizio
cambiano nel tempo. Il modello riceve input da una distribuzione
diversa da quella di training.

**2.3.2 Prediction drift** — `P(Ŷ)` cambia nel tempo. Non è un drift
"vero" nel senso stretto, ma un sintomo osservabile: se le predizioni
del modello si spostano nel tempo, verosimilmente c'è drift sui dati
di input o nel concetto. Utile perché richiede solo l'output del modello,
non ground truth.

**2.3.3 Concept drift** — `P(Y|X)` cambia nel tempo. La relazione
input-output cambia: gli stessi input producono output diversi. Esempio:
un modello addestrato a rilevare frodi con certi pattern non riesce più
a riconoscerle quando i truffatori cambiano metodo. Richiede `y_true`
per essere rilevato.

**2.3.4 Caratterizzazione temporale del drift**:
- *Drift improvviso* (sudden/abrupt): salto discreto della distribuzione
  ad un dato istante temporale
- *Drift graduale* (gradual): la nuova distribuzione emerge lentamente
  in coesistenza con la vecchia
- *Drift incrementale* (incremental): la distribuzione varia in modo
  continuo e progressivo nel tempo
- *Drift ricorrente* (recurring): pattern che si alternano nel tempo
  (esempio: stagionalità)

**Citazioni**: `\cite{gama2014survey, lu2019learning}` per la tassonomia
completa.

### 5.2.4 Sezione 2.4 — Tecniche di drift detection 🟡

**2.4.1 Test statistici a due campioni** ✅

- **Kolmogorov-Smirnov** (a due campioni): test non-parametrico che
  confronta le funzioni di distribuzione cumulativa empiriche (ECDF) di
  due campioni. Utile per feature continue.
- **Chi-quadro** (χ²): confronta le frequenze osservate con quelle attese.
  Utile per feature categoriche.
- **Mann-Whitney U**: test non-parametrico che verifica se un campione
  tende ad avere valori sistematicamente più grandi dell'altro.

**2.4.2 Misure di divergenza** ✅

- **Jensen-Shannon Divergence (JSD)**: misura simmetrica e limitata di
  divergenza fra due distribuzioni. Adatta a distribuzioni discrete o
  continue binned.
- **KL Divergence**: non simmetrica, può essere infinita se una
  distribuzione ha supporto strettamente contenuto nell'altra.
- **Wasserstein distance** (o Earth Mover's Distance): misura il "costo"
  di trasformare una distribuzione nell'altra. Robusta e interpretabile
  geometricamente.

**2.4.3 Metodi error-based** ✅

- **DDM (Drift Detection Method)** [Gama et al., 2004]: monitora il
  tasso di errore del modello e la sua deviazione standard. Segnala
  warning e drift quando queste superano soglie fisse.
- **EDDM (Early Drift Detection Method)**: variante di DDM che monitora
  la distanza fra errori consecutivi anziché il solo tasso. Più sensibile
  ai drift graduali.
- **Page-Hinkley test**: change detection su una serie temporale,
  monitorando la deviazione cumulativa rispetto alla media corrente.
- Citare `\cite{sethi2018reliable}` per una rassegna di questi metodi.

**2.4.4 Metodi window-based adattivi** ✅

- **ADWIN** [`\cite{bifet2007adwin}`]: mantiene una finestra scorrevole
  di dimensione **variabile** che si allunga quando i dati sono stazionari
  e si accorcia bruscamente quando viene rilevato un cambiamento. Offre
  garanzie formali sui tassi di falsi positivi e falsi negativi
  (Teorema 3.1 del paper).
- **ADWIN2**: versione efficiente di ADWIN che utilizza istogrammi
  esponenziali `\cite{datar2002maintaining}` per ridurre memoria a
  `O(M·log(W/M))` e tempo per update a `O(log W)`, mantenendo le stesse
  garanzie statistiche del Teorema 3.1.

**2.4.5 Confronto fra le famiglie di metodi** ✅

Tabella comparativa da inserire:

| Categoria | Adatti a | Vantaggi | Limiti |
|---|---|---|---|
| Test statistici a due campioni | Feature drift, prediction drift | Rigorosi, ben studiati | Richiedono finestre fisse |
| Misure di divergenza | Feature drift | Interpretabili geometricamente | Richiedono binning per continui |
| Metodi error-based | Concept drift | Rilevano degradazione diretta | Richiedono `y_true` |
| Metodi window-based adattivi | Tutti | Online, gestiscono stream infiniti | Solo cambi di media (per ADWIN) |

### 5.2.5 Sezione 2.5 — MLOps: integrazione del drift monitoring ⏳

**Da scrivere in profondità solo quando avremo la parte MLOps della tesi
più definita**. Per ora si può dare una definizione di MLOps come
disciplina che estende DevOps al ciclo di vita dei modelli ML,
e posizionare il drift monitoring come uno dei suoi componenti chiave.

### 5.2.6 Sezione 2.6 — Librerie e strumenti esistenti 🟡

**Da menzionare**:

- **Evidently AI**: libreria open-source per drift detection con dashboard
  integrate. Offre report interattivi HTML/JSON.
- **Alibi Detect** (Seldon): libreria specializzata in outlier detection,
  adversarial detection, drift detection. Include implementazioni di
  Kolmogorov-Smirnov, Chi-Squared, MMD (Maximum Mean Discrepancy).
- **river**: libreria Python per streaming ML. Include implementazioni
  di ADWIN, DDM, EDDM, KSWIN, Page-Hinkley. Utilizzata in questo lavoro
  per l'implementazione di ADWIN. `\cite{montiel2021river}`.
- **scipy.stats**: modulo statistico standard, contiene `ks_2samp` per
  il test di Kolmogorov-Smirnov a due campioni. `\cite{virtanen2020scipy}`.

**Motivazione delle scelte fatte**: il lavoro utilizza scipy e river
invece di Evidently AI/Alibi Detect perché l'obiettivo della tesi è
costruire un framework architettonicamente originale, non aggregare
strumenti esistenti. L'integrazione futura con Evidently AI e Alibi
Detect resta possibile grazie all'architettura basata su Strategy Pattern.

---

## 5.3 CAPITOLO 3 — Architettura del framework ✅

**Pagine target**: 12-15

### 5.3.1 Sezione 3.1 — Requisiti e principi di design ✅

**Principi**:
- **Modularità**: componenti indipendenti, interfacce chiare
- **Model-agnosticità**: il framework non fa assunzioni sulla natura del
  modello ML monitorato
- **Estensibilità**: aggiungere nuove strategie o nuovi detector senza
  modificare il codice esistente
- **Separation of concerns**: ogni classe ha una responsabilità
  ben definita e limitata

### 5.3.2 Sezione 3.2 — Visione d'insieme ✅

**Architettura a tre livelli**:

```
Livello 3 — Servizio di monitoring
     DriftMonitoringService (orchestrazione globale) [FUTURO]
             │
Livello 2 — Detector per tipo di drift
     FeatureDriftDetector  |  PredictionDriftDetector  |  ConceptDriftDetector
             │
Livello 1 — Strategie (singolo stream)
     KSStrategy  |  ADWINStrategy  |  altre strategie future
```

Ogni livello dipende solo dal livello sottostante tramite un'**interfaccia
astratta** (`BaseDriftDetector`).

### 5.3.3 Sezione 3.3 — Il pattern Strategy come scelta di design ✅

**Riferimento**: `\cite{gof1994patterns}` — libro Gang of Four.

**Definizione del pattern**: "Define a family of algorithms, encapsulate
each one, and make them interchangeable. Strategy lets the algorithm
vary independently from clients that use it."

**Applicazione al framework**:
- Le tecniche di drift detection sono la "famiglia di algoritmi"
- Ogni tecnica è incapsulata in una classe (`KSStrategy`, `ADWINStrategy`)
- Il "client" è il `FeatureDriftDetector`, che non conosce la tecnica
  specifica

**Vantaggi rispetto ad alternative**:
- Rispetto all'ereditarietà: le strategie possono essere sostituite a
  runtime, l'ereditarietà legherebbe il detector a un algoritmo specifico
- Rispetto al Template Method: le strategie non condividono uno
  scheletro comune, sono algoritmi completamente diversi
- Rispetto a un semplice `if/else`: aggiungere una nuova strategia non
  richiede modifiche al codice esistente

### 5.3.4 Sezione 3.4 — Le interfacce base ✅

**3.4.1 `BaseDriftDetector`**:

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

Contratto: ogni detector/strategia riceve valori uno alla volta con
`update()`, produce un verdetto con `detect()`, può essere azzerato con
`reset()`.

**3.4.2 `BaseModel`**:

```python
class BaseModel(ABC):
    @abstractmethod
    def predict(self, x): ...
    
    @abstractmethod
    def get_name(self) -> str: ...
```

Astrazione minimale del modello: espone solo `predict()`. Nessun accesso
agli internals. Questo è ciò che garantisce la **model-agnosticità**.

**3.4.3 `DriftResult`**:

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

Dataclass per il verdetto strutturato. Il campo `metadata` è un
contenitore libero che ogni strategia usa per informazioni
specifiche (statistica D, p-value per KS; delta per ADWIN;
feature drittate per il `FeatureDriftDetector`).

### 5.3.5 Sezione 3.5 — Principio Open/Closed ✅

**Enunciato del principio** (Meyer, 1988): "Software entities should be
open for extension, but closed for modification."

**Applicazione**:
- `BaseDriftDetector` è **chiusa** alla modifica (l'interfaccia non
  cambia mai)
- Il framework è **aperto** all'estensione: aggiungere una nuova
  strategia richiede solo la creazione di una nuova classe, senza
  toccare `FeatureDriftDetector` o altre parti del codice

Questo è dimostrabile empiricamente: l'aggiunta di `ADWINStrategy` dopo
`KSStrategy` non ha richiesto modifiche al `FeatureDriftDetector`.

---

## 5.4 CAPITOLO 4 — Strategie di drift detection ✅

**Pagine target**: 18-22

### 5.4.1 Sezione 4.1 — La strategia Kolmogorov-Smirnov ✅

**4.1.1 Fondamenti teorici**:

- Il **test di Kolmogorov-Smirnov a due campioni** è un test statistico
  **non-parametrico** che verifica se due campioni di dati provengono
  dalla stessa distribuzione di probabilità.
- L'ipotesi nulla H₀ è: "i due campioni provengono dalla stessa
  distribuzione".
- Il test si basa sulla **funzione di distribuzione cumulativa empirica**
  (ECDF). Per un campione `X = {x_1, ..., x_n}` l'ECDF è definita come:
  `F(x) = (numero di x_i ≤ x) / n`
- Data la ECDF di due campioni, `F_1` e `F_2`, la **statistica D del
  KS** è la massima distanza verticale fra le due:
  `D = max_x |F_1(x) − F_2(x)|`
- Da `D` e dalle dimensioni dei campioni si ricava un **p-value**, che
  rappresenta la probabilità di osservare una distanza almeno pari a
  `D` sotto l'ipotesi nulla.
- **Regola di decisione**: se il p-value scende sotto una soglia di
  significatività α (tipicamente 0.05), si rifiuta H₀ e si conclude
  che le due distribuzioni sono statisticamente diverse.
- **Natura non-parametrica**: il KS non assume alcuna forma specifica
  della distribuzione (non richiede normalità). Questo lo rende
  particolarmente adatto a feature di natura sconosciuta in scenari
  MLOps.

**4.1.2 Adattamento allo streaming**:

- Il KS classico è un test **batch**: prende due campioni fissi e li
  confronta una sola volta. Per adattarlo al drift monitoring in
  streaming si utilizza uno **schema a due finestre**:
  - **Reference window**: finestra fissa dei primi `n` valori osservati,
    rappresenta la distribuzione "attesa". Nel framework corrente si
    riempie con i primi `n` valori dello stream e resta congelata.
  - **Current window**: finestra di dimensione fissa `n`, contiene gli
    ultimi `n` valori del flusso. Si comporta come una coda FIFO: quando
    un nuovo valore entra, il più vecchio esce.
- Il KS viene ricalcolato ad ogni chiamata di `detect()` fra le due
  finestre correnti.
- **Trade-off dimensione finestra**:
  - Finestre grandi → potere statistico elevato ma reattività bassa
    (il drift viene rilevato tardi)
  - Finestre piccole → reattività alta ma più falsi positivi

**4.1.3 Implementazione — `KSStrategy`**:

```python
class KSStrategy(BaseDriftDetector):
    def __init__(self, window_size: int = 100, alpha: float = 0.05):
        super().__init__(detector_name="KS", drift_type="feature")
        self.window_size = window_size
        self.alpha = alpha
        self.reference = deque(maxlen=window_size)
        self.current = deque(maxlen=window_size)
    
    def update(self, value):
        if len(self.reference) < self.window_size:
            self.reference.append(value)
        else:
            self.current.append(value)
    
    def detect(self):
        if len(self.current) < self.window_size:
            return DriftResult(drift_detected=False,
                               metadata={"status": "warming_up"})
        statistic, p_value = ks_2samp(list(self.reference),
                                       list(self.current))
        return DriftResult(
            drift_detected=(p_value < self.alpha),
            score=p_value,
            metadata={"statistic": statistic, "alpha": self.alpha},
        )
    
    def reset(self):
        self.reference.clear()
        self.current.clear()
```

**Dettagli chiave**:
- L'uso di `collections.deque(maxlen=n)` fornisce automaticamente il
  comportamento FIFO senza gestione manuale
- Durante il *warming up* (prime `2n` chiamate di `update`), `detect()`
  restituisce sempre `drift_detected=False` con status esplicito
- Si sfrutta `scipy.stats.ks_2samp` `\cite{virtanen2020scipy}` per il
  calcolo effettivo della statistica

**4.1.4 Limiti e considerazioni**:

- **Potere statistico ridotto su dati discreti**: la ECDF di variabili
  binarie (es. Bernoulli) ha solo due gradini, molto meno informativa
  di quella di variabili continue. Il KS resta applicabile ma con
  potere statistico inferiore.
- **Finestre statiche**: la reference window resta congelata, non si
  adatta a drift graduali che si consolidano dopo la fase iniziale.
- **Segnale continuo dopo il rilevamento**: una volta rilevato drift,
  la strategia continua a segnalarlo ad ogni `detect()` successivo,
  finché le finestre non tornano compatibili.
- **Reference dallo stream vs training data**: idealmente la reference
  dovrebbe essere popolata con dati di training, non con i primi valori
  in ingresso. Nel framework corrente si accetta questa semplificazione
  come limite consapevole.

### 5.4.2 Sezione 4.2 — La strategia ADWIN ✅

**4.2.1 Fondamenti teorici**:

- **ADWIN** (**ADaptive WINdowing**) è un algoritmo di drift detection
  introdotto da Bifet e Gavaldà nel 2006-2007 `\cite{bifet2006adwin,
  bifet2007adwin}`.
- L'idea centrale: mantenere una singola **finestra scorrevole `W`** di
  dimensione **variabile**, che cresce automaticamente quando i dati
  sono stazionari e si accorcia bruscamente quando viene rilevato un
  cambiamento.
- L'algoritmo confronta continuamente sotto-finestre della `W` alla
  ricerca di partizioni dove la media della parte vecchia differisce
  in modo statisticamente significativo da quella della parte nuova.
- Vantaggi rispetto ai metodi a finestra fissa:
  - Non richiede di fissare a priori la dimensione della finestra
  - Si adatta automaticamente alla scala temporale del cambiamento
  - Fornisce **garanzie formali** sui tassi di falsi positivi e negativi

**4.2.2 Distinzione fra ADWIN base e ADWIN2**:

- **ADWIN base** (sezione 3.2 del paper): la formulazione concettuale.
  Mantiene tutti i valori in memoria e prova tutti i possibili tagli
  ad ogni update. Costi in memoria e tempo `O(W)` — inutilizzabile su
  stream lunghi.
- **ADWIN2** (sezione 3.3 del paper): versione efficiente basata su
  **istogrammi esponenziali** `\cite{datar2002maintaining}`. Costi in
  memoria `O(M·log(W/M))` e tempo per update `O(log W)`, con le stesse
  garanzie statistiche di ADWIN base.
- Il framework utilizza **ADWIN2** attraverso la libreria river
  `\cite{montiel2021river}`.

**4.2.3 La struttura a istogrammi esponenziali**:

- Ogni **bucket** contiene due informazioni: **capacità** (numero di
  valori rappresentati, sempre potenza di 2) e **contenuto** (somma dei
  valori). Nessun singolo valore è conservato esplicitamente.
- Il parametro **`M`** (default 5 nel paper) è il **numero massimo di
  bucket per ogni dimensione 2^i**.
- Quando arriva un nuovo valore, viene inserito come bucket di dimensione
  1. Se ora ci sono `M+1` bucket di dimensione 1, i due più vecchi
  vengono **fusi** in un bucket di dimensione 2. Il meccanismo si
  propaga a cascata verso l'alto.
- Con `M=5`, su uno stream di 100.000 valori la struttura contiene
  circa 70 bucket totali.

**4.2.4 Il meccanismo del taglio**:

- Ad ogni nuovo valore, ADWIN2 prova i possibili tagli **solo ai
  confini fra bucket consecutivi**, non fra singoli valori.
- Per ciascun taglio candidato, la finestra viene divisa in:
  - `W₀`: parte vecchia (uno o più bucket)
  - `W₁`: parte nuova (uno o più bucket)
- Si calcolano le medie `μ(W₀)` e `μ(W₁)` sommando contenuti e
  capacità dei bucket coinvolti.
- Si confronta la differenza con la soglia adattiva `ε_cut`:
  se `|μ(W₀) − μ(W₁)| > ε_cut`, si scarta `W₀` e si segnala drift.

**4.2.5 La formula `ε_cut` (Equazione 3.1 del paper)**:

$$
\varepsilon_{cut} = \sqrt{\frac{2}{m} \cdot \sigma^2_W \cdot \ln \frac{2}{\delta'}} + \frac{2}{3m} \cdot \ln \frac{2}{\delta'}
$$

dove:
- **δ (delta)**: parametro di confidenza scelto dall'utente
- **δ' = δ / ln(n)**: correzione per il testing multiplo
- **`m` = (n_0 · n_1) / (n_0 + n_1)**: media armonica delle dimensioni
  di W₀ e W₁
- **σ²_W**: varianza osservata nella finestra

**Ruolo di delta**:
- `delta` piccolo (default 0.002) → `ε_cut` più grande → più prudente,
  meno falsi allarmi
- `delta` grande (es. 0.05) → `ε_cut` più piccolo → più sensibile, ma
  più falsi allarmi

**Nota importante**: `delta` non viene mai confrontato direttamente con
la differenza delle medie. Entra dentro la formula come parametro che
modula la soglia adattiva `ε_cut`.

**4.2.6 Garanzie teoriche (Teorema 3.1 del paper)**:

Due garanzie formali:

1. **Limite sui falsi positivi**: se la distribuzione è stazionaria
   all'interno di `W`, la probabilità che ADWIN segnali drift in un dato
   passo è al massimo `delta`.
2. **Limite sui falsi negativi**: se esiste una partizione `W = W₀ · W₁`
   con `|μ(W₀) − μ(W₁)| > 2·ε_cut`, ADWIN riduce `W` a `W₁` (o più corta)
   con probabilità almeno `1 − delta`.

Queste garanzie **non sono asintotiche**: valgono ad ogni passo
temporale. Questa è una proprietà distintiva di ADWIN rispetto a molti
altri detector (DDM, EDDM, Page-Hinkley) che sono giustificati solo
empiricamente.

**4.2.7 Implementazione — `ADWINStrategy`**:

```python
class ADWINStrategy(BaseDriftDetector):
    def __init__(self, delta: float = 0.002):
        super().__init__(detector_name="ADWIN", drift_type="feature")
        self.delta = delta
        self.adwin = ADWIN(delta=delta)
    
    def update(self, value):
        self.adwin.update(value)
    
    def detect(self):
        return DriftResult(
            drift_detected=self.adwin.drift_detected,
            score=self.adwin.estimation,
            metadata={"delta": self.delta},
        )
    
    def reset(self):
        self.adwin = ADWIN(delta=self.delta)
```

**Dettagli chiave**:
- Il wrapper è un **thin adapter** su `river.drift.ADWIN`: non
  reimplementa l'algoritmo ma ne uniforma l'interfaccia al
  `BaseDriftDetector` del framework.
- Motivazione della scelta di usare river: il valore aggiunto della
  tesi è l'**architettura del framework**, non la reimplementazione
  di un algoritmo consolidato. river ha un'implementazione testata,
  ottimizzata e fedele al paper.
- Nel wrapper si espone solo `delta`. Altri parametri di river
  (`clock`, `max_buckets`, `min_window_length`, `grace_period`) restano
  ai default di libreria.

**4.2.8 Limiti e considerazioni**:

- **Sensibilità solo alla media**: ADWIN monitora la media dello stream.
  Cambiamenti di varianza, di asimmetria o della forma della
  distribuzione senza spostamento della media possono sfuggire.
- **Segnale a impulso**: `drift_detected` è `True` solo nell'istante
  esatto del taglio; al successivo update torna `False` (a meno di un
  altro taglio).
- **Non fornisce dettaglio sul cambiamento**: ADWIN segnala che qualcosa
  è cambiato nella media, ma non indica di quanto o come.

### 5.4.3 Sezione 4.3 — Confronto fra le strategie implementate ✅

**Tabella comparativa da inserire**:

| Aspetto | KS | ADWIN |
|---|---|---|
| Numero di finestre | 2 (reference + current) | 1 adattiva |
| Cosa confronta | Forma intera della distribuzione (ECDF) | Solo la media |
| Test statistico | Massima distanza fra ECDF, p-value | Confronto di medie con soglia `ε_cut` |
| Soglia di decisione | `p-value < α` | `|μ(W₀) − μ(W₁)| > ε_cut` |
| Dati continui | Eccellente | Buono |
| Dati binari/discreti | Potere statistico ridotto | Nativo |
| Garanzie formali | Test KS classico | Limiti su FP/FN (Teorema 3.1) |
| Segnale nel tempo | Continuo (True finché ref ≠ curr) | A impulso (True solo al taglio) |
| Latenza di rilevamento | Alta (dipende da window_size) | Bassa (finestra adattiva) |
| Costo di memoria | `O(window_size)` | `O(M·log(W/M))` logaritmico |

Le due strategie sono **complementari** e non alternative in senso
assoluto: la scelta dipende dal tipo di dati e dallo scenario applicativo.

### 5.4.4 Sezione 4.4 — Strategie future ⏳

Brevi cenni alle strategie che verranno implementate nelle fasi successive:

- **Jensen-Shannon Divergence**: per feature drift e prediction drift,
  utile su distribuzioni discrete o continue binned
- **DDM, EDDM, Page-Hinkley**: per concept drift, operanti su stream
  di errori

---

## 5.5 CAPITOLO 5 — Detector e orchestrazione 🟡

**Pagine target**: 12-15

### 5.5.1 Sezione 5.1 — Architettura dei detector per tipo di drift ✅

- Un **detector** è un orchestratore specifico per un tipo di drift
  (feature, prediction, concept). Non implementa direttamente
  l'algoritmo di detection: delega alle strategie sottostanti.
- Il pattern è: **ricezione di input → smistamento a strategie →
  aggregazione dei verdetti**.

### 5.5.2 Sezione 5.2 — `FeatureDriftDetector` ✅

**Struttura**:

```python
class FeatureDriftDetector(BaseDriftDetector):
    def __init__(self, strategy_cls, n_features, k=1,
                 feature_names=None, **strategy_kwargs):
        super().__init__(detector_name="FeatureDriftDetector",
                         drift_type="feature")
        self.n_features = n_features
        self.k = k
        self.feature_names = feature_names
        self.strategies = [strategy_cls(**strategy_kwargs)
                           for _ in range(n_features)]
    
    def update(self, x):
        for i in range(self.n_features):
            self.strategies[i].update(x[i])
    
    def detect(self):
        drifted_indices = [i for i, s in enumerate(self.strategies)
                           if s.detect().drift_detected]
        ...
```

**Punti chiave**:
- **Una strategia indipendente per ogni feature**: la stessa strategia
  (`KSStrategy` o `ADWINStrategy`) viene istanziata `n_features` volte.
- **Smistamento**: ogni componente del vettore `x` va alla strategia
  corrispondente.
- **Regola di aggregazione**: drift globale se **almeno `k` feature**
  sono drittate. Il parametro `k` è configurabile (default 1).
- **Output**: un solo `DriftResult` con `metadata["drifted_features"]`
  che contiene la lista delle feature problematiche.

**Vantaggio architetturale**: cambiare `strategy_cls` da `KSStrategy` a
`ADWINStrategy` (o a qualsiasi altra strategia futura) non richiede
modifiche al `FeatureDriftDetector`.

### 5.5.3 Sezione 5.3 — `PredictionDriftDetector` ⏳

Placeholder. Da progettare e implementare. Punti previsti:
- Riuso delle stesse strategie sullo stream delle predizioni del modello
- Trade-off classificazione: monitorare classi predette (0/1) vs
  probabilità predette (continue in [0,1])
- Per classificazione binaria: KS su probabilità predette funziona bene
- Per classi discrete: preferibile ADWIN

### 5.5.4 Sezione 5.4 — `ConceptDriftDetector` ⏳

Placeholder. Da progettare e implementare. Punti previsti:
- Richiede disponibilità di `y_true`
- Calcolo dello stream di errori: `error = 1{ŷ ≠ y}` per classificazione,
  `error = |ŷ - y|` o `(ŷ - y)²` per regressione
- Gestione di scenari di label delay (`y_true` disponibile dopo tempo)

### 5.5.5 Sezione 5.5 — `DriftMonitoringService` ⏳

Placeholder. Da progettare e implementare. Punti previsti:
- Orchestratore centrale che unisce modello e detector
- Riceve input `x`, opzionalmente `y_true`
- Chiama `model.predict(x)` internamente
- Smista i dati ai detector appropriati
- Aggrega i risultati in un output unificato

---

## 5.6 CAPITOLO 6 — Validazione sperimentale 🟡

**Pagine target**: 18-22

### 5.6.1 Sezione 6.1 — Metodologia di validazione ✅

**Metriche utilizzate**:
- **Latenza di rilevamento**: numero di passi fra il drift reale e la
  prima segnalazione
- **Tasso di falsi positivi**: segnalazioni di drift quando la
  distribuzione è stazionaria
- **Tasso di falsi negativi**: mancate segnalazioni di drift quando la
  distribuzione cambia
- **Numero totale di segnali**: rilevante per valutare stabilità del
  detector

**Approccio**: validazione su dataset sintetici con drift artificialmente
iniettato in punti noti, per poter confrontare la posizione reale del
drift con quella rilevata dal detector.

### 5.6.2 Sezione 6.2 — Generatore di dati sintetici ✅

**Modulo**: `data/synthetic_generator.py`

**Funzione principale**: `bernoulli_with_abrupt_drift(n_samples,
p_before, p_after, drift_point, seed)`.

- Genera uno stream Bernoulli di `n_samples` valori
- I primi `drift_point` valori seguono `Bernoulli(p_before)`
- I restanti `n_samples - drift_point` valori seguono `Bernoulli(p_after)`
- Il `seed` garantisce riproducibilità

**Motivazione della scelta Bernoulli**: replicare fedelmente lo scenario
della **Figura 2 del paper di Bifet & Gavaldà** `\cite{bifet2007adwin}`,
che è il benchmark canonico per la validazione di detector di drift a
cambio brusco.

**Nota metodologica**: il KS test è progettato per distribuzioni
continue; sui dati Bernoulli il suo potere statistico è ridotto. La
scelta è mantenuta per fedeltà al benchmark del paper, accettando la
sub-ottimalità come discussione metodologica valida per la tesi.

### 5.6.3 Sezione 6.3 — Esperimento 1: KS su dati sintetici Bernoulli ✅

**Setup**:
- 3 feature Bernoulli, 2000 campioni ciascuna
- `f0_stable`: `Bernoulli(p=0.2)` costante
- `f1_drift`: `Bernoulli(p=0.8)` per i primi 1000 passi, poi
  `Bernoulli(p=0.4)` (cambio brusco al passo 1000)
- `f2_stable`: `Bernoulli(p=0.5)` costante
- Detector: `FeatureDriftDetector(strategy_cls=KSStrategy,
  n_features=3, k=1, window_size=200, alpha=0.05)`

**Risultati**:
- Primo drift rilevato al passo `t ≈ 1100`
- Latenza: circa **100 passi** dopo il cambio reale (`t=1000`)
- Feature identificate come drittate: **solo `f1_drift`**
- Falsi positivi su `f0_stable` e `f2_stable`: **nessuno**
- Comportamento post-drift: `drift_detected` resta `True` in modo
  continuativo perché reference e current restano statisticamente diverse

**Interpretazione**: il ritardo di ~100 passi è coerente con la teoria:
la current window (dimensione 200) deve popolarsi di valori nuovi
prima che la differenza fra le ECDF sia statisticamente significativa.
Nella pratica, con `n=200` bastano ~100 valori nuovi perché il p-value
scenda sotto la soglia α=0.05.

### 5.6.4 Sezione 6.4 — Esperimento 2: ADWIN su dati sintetici Bernoulli ✅

**Setup**: identico al precedente, ma con `strategy_cls=ADWINStrategy,
delta=0.002`.

**Risultati**:
- Primo drift rilevato al passo `t = 1055`
- Latenza: circa **55 passi** dopo il cambio reale
- Feature identificate come drittate: **solo `f1_drift`**
- Falsi positivi su `f0_stable` e `f2_stable`: **nessuno**
- Numero totale di segnali di drift: **1** (comportamento a impulso)

**Interpretazione**: la latenza dimezzata rispetto al KS è coerente
con la teoria di ADWIN: la finestra adattiva si accorcia non appena
la differenza fra le medie di W₀ e W₁ supera `ε_cut`, senza dover
attendere il riempimento di una current window.

### 5.6.5 Sezione 6.5 — Confronto KS vs ADWIN sullo stesso scenario ✅

**Tabella riassuntiva dei risultati**:

| Metrica | KS | ADWIN |
|---|---|---|
| Drift rilevato su `f1_drift` | Sì | Sì |
| Falsi positivi su `f0_stable`, `f2_stable` | 0 | 0 |
| Primo passo di rilevamento | ~1100 | 1055 |
| Latenza rispetto al drift reale (t=1000) | ~100 | 55 |
| Numero di segnali di drift durante il flusso | Continui post-drift | 1 (impulso) |
| Comportamento del segnale | Persistente | Istantaneo |

**Discussione**:
- ADWIN è ~2× più veloce di KS nel rilevare il drift, come atteso
- Nessuno dei due genera falsi positivi sulle feature stabili
- Il segnale continuo di KS è più adatto a scenari dove serve un
  flag di stato ("il modello è in drift"); il segnale impulsivo di
  ADWIN è più adatto a scenari basati su eventi ("è successo qualcosa")
- Nessuna delle due strategie è superiore in senso assoluto: la
  scelta dipende dal tipo di dati e dalle esigenze applicative

**Conclusione dell'esperimento**: la validazione conferma il corretto
funzionamento dell'architettura Strategy Pattern del `FeatureDriftDetector`,
che gestisce entrambe le strategie in modo intercambiabile senza
modifiche al codice.

### 5.6.6 Sezione 6.6 — Esperimento 3: validazione su dataset reale ELEC2 ⏳

Placeholder. Da svolgere. Punti previsti:

- Descrizione del dataset ELEC2 (Electricity Market): 45.312 istanze
  dal 7 maggio 1996 al 5 dicembre 1998, mercato elettrico australiano
- Feature numeriche: giorno della settimana, timestamp, domanda NSW,
  domanda Vic, trasferimento programmato
- Etichetta binaria: cambio di prezzo rispetto alla media mobile 24h
- Applicazione delle strategie KS e ADWIN
- Confronto con risultati attesi dalla letteratura

### 5.6.7 Sezione 6.7 — Esperimenti futuri ⏳

Placeholder per esperimenti che verranno svolti:
- Prediction drift su modello di classificazione reale
- Concept drift su stream di errori
- Validazione multi-dataset

---

## 5.7 CAPITOLO 7 — Pipeline MLOps e integrazione ⏳

**Pagine target**: 10-15
**Stato**: PLACEHOLDER COMPLETO — da riempire nelle fasi finali della tesi.

**Sotto-sezioni previste**:
- 7.1 Architettura della pipeline end-to-end
- 7.2 Tracking degli esperimenti con MLflow
- 7.3 Orchestrazione con Argo Workflows e Argo Events
- 7.4 Containerizzazione con Docker
- 7.5 Strategie di retraining
- 7.6 Test end-to-end della pipeline

---

## 5.8 CAPITOLO 8 — Conclusioni e sviluppi futuri ⏳

**Pagine target**: 5-7
**Stato**: SCRIVIBILE SOLO A LAVORO COMPLETO.

**Contenuto previsto**:
- 8.1 Riepilogo del lavoro svolto
- 8.2 Risultati principali e contributo
- 8.3 Limiti del lavoro
- 8.4 Sviluppi futuri

**Direzioni di sviluppo futuro già identificate**:
- Ensemble di detector (majority voting, weighted scoring, stacking)
- Drift scoring continuo (probability, severity, confidence interval)
- Monitoring per-feature avanzato con ranking di instabilità
- Integrazione con MLOps reale (Kafka, FastAPI, time-series DB, alerting)
- Retraining automation (trigger, shadow models, rollback)
- Concept drift senza label (uncertainty-based, pseudo-labels)

---

# PARTE 6 — Log degli aggiornamenti

Sezione per tenere traccia degli aggiornamenti fatti al briefing man mano
che il lavoro procede.

| Data | Aggiornamento |
|---|---|
| 2026-06-16 | Creazione iniziale del briefing con Capitoli 1, 3, 4 (KS+ADWIN), 5.2, 6.1-6.5 completi. Capitoli 2, 5.3-5.5, 6.6-6.7, 7, 8 come placeholder. |

**Da aggiornare quando**:
- Si implementa una nuova strategia (aggiungere a 5.4.4 e spostare in 5.4)
- Si implementa un nuovo detector (aggiornare sezione 5.5)
- Si esegue un nuovo esperimento (aggiungere a Capitolo 6)
- Si prende una decisione architetturale importante (aggiornare Capitolo 3
  o Capitolo 4)
- Si aggiunge un nuovo riferimento bibliografico (aggiornare Parte 4)

---

# PARTE 7 — Istruzioni finali per l'LLM

Quando ti viene chiesto di scrivere uno o più capitoli:

1. **Leggi tutto il briefing** prima di iniziare a scrivere.
2. **Rispetta lo stile del Capitolo 1** (in `thesis/chapter1_introduzione.tex`).
3. **Usa solo le key bibtex definite** in Parte 4 per le citazioni.
4. **Rispetta i format** definiti in Parte 2 (LaTeX, IEEE, italiano,
   font Times New Roman equivalente).
5. **Attieniti al contenuto specificato** in Parte 5 per il capitolo
   richiesto. Espandi la prosa a partire dai bullet, ma non aggiungere
   contenuti non presenti nel briefing e non contraddire i fatti/numeri.
6. **Segnala** eventuali sezioni che il briefing marca come ⏳ e che
   quindi al momento non hanno contenuto: producine solo lo scheletro
   (titolo + sottotitoli), lasciando placeholder vuoti da riempire in
   futuro.
7. **Produci file `.tex` separati per capitolo** (es. `chapter2.tex`,
   `chapter3.tex`, ecc.), con `\chapter{}` iniziale e `\label{ch:...}`
   coerente col Capitolo 1.
8. **Non modificare** il Capitolo 1 già scritto.
