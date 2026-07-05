# Briefing per generazione slide — drift_framework

Questo documento è un **briefing completo** del lavoro svolto sul progetto
`drift_framework`, pensato per essere passato a un LLM che genererà
una presentazione di slide. Copre sia gli **aspetti teorici** sia
l'**implementazione**.

---

## Istruzioni per chi genera le slide

- **Lingua**: italiano
- **Tono**: informale, didattico, non da paper scientifico
- **Lunghezza target**: 12-18 slide complessive
- **Una sola idea per slide**, niente sovraccarico di testo
- **Codice essenziale**: snippet brevi e illustrativi, non blocchi enormi
- **Audience**: relatore di tesi + commissione di laurea (livello tecnico
  buono, ma serve chiarezza espositiva)
- **Stile slide**: minimal, niente decorazioni eccessive. Marp markdown è
  un formato accettabile.

---

## 1. Contesto del progetto

Il progetto è la **tesi di laurea** dello studente. L'obiettivo è costruire
un **framework MLOps di drift detection modulare e model-agnostic**, non
limitato al singolo algoritmo ma capace di accogliere più strategie e più
tipi di drift in un'architettura unificata.

Il valore della tesi sta **nell'architettura del framework**, non nella
reimplementazione degli algoritmi.

---

## 2. Il problema del drift

Quando un modello ML è in produzione, i dati che riceve possono cambiare nel
tempo. Quando i dati cambiano:

- Le performance del modello calano (silenziosamente, spesso senza
  che nessuno se ne accorga subito)
- Serve un sistema che **rilevi automaticamente** questi cambiamenti

Si distinguono tre tipi di drift:

| Tipo | Cosa cambia | Stream da monitorare |
|---|---|---|
| **Feature drift** | Distribuzione delle feature in input `P(X)` | Valori delle feature |
| **Prediction drift** | Distribuzione delle predizioni `P(ŷ)` | Output del modello |
| **Concept drift** | Relazione input/output `P(Y\|X)` | Errori del modello (serve `y_true`) |

**Focus del lavoro svolto finora**: feature drift.

---

## 3. L'architettura del framework

Il framework adotta il **Strategy Pattern** (Gamma et al., GoF) su tre livelli.

### Livello 1 — Strategie (singolo stream)

Sanno fare una sola cosa: dato uno stream di numeri, dire se è cambiato.
Sono intercambiabili tra loro e riutilizzabili per tutti i tipi di drift.

- `KSStrategy` — test di Kolmogorov-Smirnov a due campioni
- `ADWINStrategy` — finestra adattiva di Bifet & Gavaldà

### Livello 2 — Detector per tipo di drift

Orchestratori specifici per tipo di drift. Smistano i dati alle strategie e
aggregano i risultati.

- `FeatureDriftDetector` (implementato)
- `PredictionDriftDetector` (futuro)
- `ConceptDriftDetector` (futuro)

### Livello 3 — Servizio di monitoring (futuro)

Orchestratore centrale `DriftMonitoringService` che lega modello e detector
in un sistema unificato.

### Componenti trasversali

- `BaseDriftDetector` — interfaccia astratta di ogni detector/strategia
- `BaseModel` — interfaccia astratta del modello (model-agnosticità)
- `DriftResult` — dataclass del verdetto restituito da ogni `detect()`

---

## 4. KS Strategy — Teoria

### Cos'è il test KS

Il **test di Kolmogorov-Smirnov a due campioni** è un test statistico
**non-parametrico** che confronta due gruppi di numeri e dice se vengono
dalla stessa distribuzione.

### Come funziona

1. Mantiene due finestre di dimensione fissa:
   - **reference**: i dati "normali" (i primi N campioni, poi congelata)
   - **current**: gli ultimi N campioni (FIFO scorrevole)
2. Quando entrambe sono piene, costruisce le **ECDF** (Empirical Cumulative
   Distribution Functions) di ciascuna
3. Misura la **massima distanza verticale** tra le due ECDF → **statistica D**
4. Da D e dalle dimensioni delle finestre calcola un **p-value**
5. Se `p-value < α` (soglia di significatività, default 0.05) → drift

### Il p-value, in parole semplici

> "Se le due finestre venissero davvero dalla stessa distribuzione, quanto
> sarebbe improbabile vedere una distanza D almeno così grande per puro caso?"

- p-value alto → nessun motivo di sospetto
- p-value basso → la differenza è troppo grande per essere caso → drift

### Limiti del KS

- Eccellente su dati **continui**, **debole sui binari** (l'ECDF di una
  Bernoulli ha solo due gradini, poca informazione)
- Finestre **statiche** (non adattive)
- Dichiara drift in modo **continuo** dopo il cambiamento (non un singolo
  segnale)

---

## 5. KS Strategy — Implementazione

### Codice essenziale

```python
from collections import deque
from scipy.stats import ks_2samp
from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


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
            return DriftResult(drift_detected=False, metadata={"status": "warming_up"})
        statistic, p_value = ks_2samp(list(self.reference), list(self.current))
        return DriftResult(drift_detected=(p_value < self.alpha), score=p_value)
```

### Libreria utilizzata

`scipy.stats.ks_2samp`: implementazione standard del test KS. Restituisce
**sia** la statistica D **sia** il p-value in una chiamata.

### Parametri esposti dal wrapper

| Parametro | Default | Significato |
|---|---|---|
| `window_size` | 100 | Dimensione di reference e current |
| `alpha` | 0.05 | Soglia di significatività |

---

## 6. ADWIN Strategy — Teoria

### Cos'è ADWIN

**ADWIN (ADaptive WINdowing)** è un algoritmo di drift detection introdotto
da Bifet & Gavaldà nel 2006. Idea base: invece di tenere due finestre fisse,
tiene **una sola finestra di dimensione variabile**, che si allunga quando
i dati sono stabili e si accorcia quando si rileva un cambiamento.

### Perché si usa ADWIN2 e non ADWIN base

Il paper presenta due versioni:

- **ADWIN base** (sezione 3.2): mantiene tutti i valori in memoria e prova
  tutti i possibili tagli ad ogni nuovo update. Memoria e tempo `O(W)`,
  inutilizzabile su stream lunghi.
- **ADWIN2** (sezione 3.3): usa una struttura a **istogrammi esponenziali**
  che riduce memoria a `O(M·log(W/M))` e tempo a `O(log W)`, mantenendo le
  stesse garanzie statistiche.

Tutte le implementazioni pratiche, incluso `river.drift.ADWIN`, sono ADWIN2.

### La struttura a bucket

ADWIN2 non memorizza i singoli valori, ma li riassume in **bucket**:

- Ogni bucket ha due numeri: **capacità** (numero di valori, sempre potenza
  di 2) e **contenuto** (somma dei valori)
- Per ogni dimensione 2^i si tengono al massimo **M bucket** (default M=5)
- Quando si arriva a M+1 bucket della stessa dimensione, i due più vecchi
  si fondono in un bucket di dimensione doppia

Esempio: su 100.000 valori la struttura conterrà ~70 bucket totali.

### Il meccanismo del taglio

A ogni update, ADWIN prova solo i tagli **ai confini dei bucket**
(O(log W) tagli invece di O(W)). Per ogni taglio candidato divide la
finestra in:

- `W₀` = porzione vecchia (uno o più bucket)
- `W₁` = porzione nuova (uno o più bucket)

Confronta le medie con la regola:

```
se |μ(W₀) − μ(W₁)| > ε_cut → taglia W₀, segnala drift
```

### La formula ε_cut e il parametro delta

ε_cut è la **soglia adattiva** calcolata dall'Equazione 3.1 del paper:

```
ε_cut = √( (2/m) · σ²_W · ln(2/δ') ) + (2/(3m)) · ln(2/δ')
```

dove:
- `δ` (delta) = parametro di confidenza scelto dall'utente
- `σ²_W` = varianza osservata della finestra
- `m` = media armonica delle dimensioni di W₀ e W₁

**Delta**:
- piccolo (es. 0.002) → prudente, pochi falsi positivi, drift rilevato più tardi
- grande (es. 0.05) → sensibile, drift rilevato prima ma più falsi positivi

### Garanzie teoriche (Teorema 3.1 del paper)

Due garanzie matematiche formali:

1. **Tasso di falsi positivi ≤ delta**: se non c'è drift reale, la
   probabilità di un falso allarme è limitata da delta.
2. **Tasso di falsi negativi ≤ delta**: per drift abbastanza grandi, vengono
   rilevati con probabilità ≥ 1−delta.

Questa è una proprietà unica di ADWIN rispetto a molti altri detector
(es. DDM, EDDM) che sono giustificati solo empiricamente.

### Limiti di ADWIN

- Vede **solo cambiamenti di media** (non varianza o forma della distribuzione)
- Segnale a **impulso**: drift_detected=True solo nell'istante del taglio,
  poi torna False

---

## 7. ADWIN Strategy — Implementazione

### Codice essenziale

```python
from river.drift import ADWIN
from detectors.base_detector import BaseDriftDetector
from results.drift_result import DriftResult


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
        )

    def reset(self):
        self.adwin = ADWIN(delta=self.delta)
```

### Libreria utilizzata

`river.drift.ADWIN`: implementazione canonica di ADWIN2 dalla libreria
`river`, che è lo standard di settore per il machine learning su stream.

### Perché river e non implementazione manuale

- ADWIN2 a mano richiederebbe ~150-200 righe di codice intricato (bucket,
  fusione, formula del taglio)
- Il valore della tesi sta nell'**architettura del framework**, non nella
  reimplementazione di un algoritmo già consolidato
- river ha l'implementazione testata, ottimizzata, fedele al paper

### Parametri esposti dal wrapper

| Parametro | Default | Significato |
|---|---|---|
| `delta` | 0.002 | Parametro di confidenza |

---

## 8. Confronto KS vs ADWIN

| Caratteristica | KS | ADWIN |
|---|---|---|
| Finestra | Due finestre fisse | Una finestra adattiva |
| Cosa misura | Forma della distribuzione (ECDF) | Solo la media |
| Soglia | `p-value < α` | `\|μ_W₀ − μ_W₁\| > ε_cut` |
| Dati continui | Eccellente | Buono |
| Dati binari | Potere statistico ridotto | Nativo |
| Garanzie teoriche | Quelle del test KS classico | Limiti formali (Teorema 3.1) |
| Segnale | Continuo (True finché reference ≠ current) | A impulso (True solo al taglio) |
| Latenza tipica | Più lenta | Più veloce |

Le due strategie sono **complementari**: scegliere quale usare dipende dal
tipo di dati e dallo scenario.

---

## 9. FeatureDriftDetector — Orchestratore multi-feature

### Cosa fa

Una singola strategia (KS o ADWIN) lavora su uno stream di numeri. Per
monitorare un dataset con N feature serve un orchestratore che:

- Crea **N istanze indipendenti** della strategia (una per feature)
- Smista i valori di ogni nuovo campione `x` alla strategia corretta
- Aggrega i verdetti delle N strategie in un unico verdetto globale
- Applica la regola: drift globale se **almeno k feature** sono drittate

### Codice essenziale

```python
class FeatureDriftDetector(BaseDriftDetector):
    def __init__(self, strategy_cls, n_features, k=1, feature_names=None, **strategy_kwargs):
        super().__init__(detector_name="FeatureDriftDetector", drift_type="feature")
        self.n_features = n_features
        self.k = k
        self.feature_names = feature_names
        self.strategies = [strategy_cls(**strategy_kwargs) for _ in range(n_features)]

    def update(self, x):
        for i in range(self.n_features):
            self.strategies[i].update(x[i])

    def detect(self):
        drifted_indices = [i for i, s in enumerate(self.strategies)
                           if s.detect().drift_detected]
        return DriftResult(
            drift_detected=(len(drifted_indices) >= self.k),
            metadata={"drifted_features": drifted_indices, ...}
        )
```

### Parametri

| Parametro | Default | Significato |
|---|---|---|
| `strategy_cls` | obbligatorio | Classe della strategia (`KSStrategy` o `ADWINStrategy`) |
| `n_features` | obbligatorio | Numero di feature da monitorare |
| `k` | 1 | Numero minimo di feature drittate per drift globale |
| `feature_names` | None | Nomi opzionali (per metadata leggibili) |
| `**strategy_kwargs` | — | Parametri passati a ogni istanza di strategia |

### Esempio d'uso

```python
detector = FeatureDriftDetector(
    strategy_cls=KSStrategy,      # o ADWINStrategy
    n_features=3,
    k=1,
    feature_names=["f0", "f1", "f2"],
    window_size=200,              # → passato a KSStrategy
    alpha=0.05,                   # → passato a KSStrategy
)

for x in stream:
    detector.update(x)
    result = detector.detect()
    if result.drift_detected:
        print(result.metadata["drifted_features"])
```

### Nota importante

KS e ADWIN sono **due strategie alternative**, non un ensemble. Si sceglie
una alla volta cambiando `strategy_cls`.

---

## 10. Validazione sperimentale

### Scenario di test

Replica della **Figura 2 del paper Bifet & Gavaldà**: cambio brusco di μ
in distribuzione Bernoulli.

**Dataset sintetico — 3 feature × 2000 campioni**:

| Feature | Comportamento | Atteso |
|---|---|---|
| `f0_stable` | Bernoulli(p=0.2) costante | NO drift |
| `f1_drift` | Bernoulli(p=0.8) → Bernoulli(p=0.4) a t=1000 | DRIFT |
| `f2_stable` | Bernoulli(p=0.5) costante | NO drift |

### Risultati KS

- **Drift rilevato**: solo su `f1_drift` ✓
- **Falsi positivi**: nessuno ✓
- **Primo segnale al passo**: ~1100
- **Latenza**: ~100 passi
- **Comportamento**: segnale continuo dopo il rilevamento

### Risultati ADWIN

- **Drift rilevato**: solo su `f1_drift` ✓
- **Falsi positivi**: nessuno ✓
- **Primo segnale al passo**: ~1055
- **Latenza**: ~55 passi
- **Comportamento**: un solo segnale, a impulso (poi torna False)

### Conclusioni della validazione

- Entrambe le strategie rilevano correttamente il drift
- Entrambe ignorano correttamente le feature stabili
- ADWIN è circa **2× più veloce** del KS nel rilevare il cambio
- Comportamenti del segnale **complementari** (continuo vs impulso)
- L'architettura Strategy Pattern del `FeatureDriftDetector` funziona
  end-to-end con entrambe senza modifiche al codice esistente

---

## 11. Decisioni di design e rationale

### Strategy Pattern
**Decisione**: separare le strategie (algoritmi) dai detector (orchestratori).
**Rationale**: una stessa strategia può servire più tipi di drift (feature,
prediction, concept). Il detector non sa nulla dell'algoritmo dietro.

### Wrapping di librerie esterne
**Decisione**: usare `scipy.stats.ks_2samp` e `river.drift.ADWIN` invece di
reimplementare gli algoritmi.
**Rationale**: il valore della tesi è l'architettura, non la
reimplementazione. Le librerie sono testate, ottimizzate, citabili come
standard di settore.

### KS e ADWIN come strategie alternative (non ensemble)
**Decisione**: le due strategie si scelgono una alla volta.
**Rationale**: maggiore semplicità implementativa. Ensemble è una direzione
futura.

### Reference window auto-fill dallo stream
**Decisione**: la reference window si riempie con i primi N valori dello
stream invece di essere caricata dai dati di training.
**Rationale**: semplicità. Limite accettato consapevolmente, da discutere
nei capitoli finali della tesi come direzione di estensione.

### Parametri di default
**Decisione**: `KSStrategy(window_size=100, alpha=0.05)`,
`ADWINStrategy(delta=0.002)`.
**Rationale**: valori standard del paper e della letteratura.

---

## 12. Prossimi passi (da non includere necessariamente nelle slide)

- Test su dataset reale (ELEC2, citato dal paper di ADWIN)
- Implementazione del `PredictionDriftDetector` (riusa le stesse strategie
  sullo stream delle predizioni)
- Implementazione del `ConceptDriftDetector` (su stream di errori)
- Implementazione del `DriftMonitoringService` (orchestratore centrale)
- Possibili strategie aggiuntive: Jensen-Shannon Divergence, DDM,
  Page-Hinkley

---

## 13. Suggerimenti per la struttura delle slide

Possibile divisione in slide (template suggerito, 14 slide):

1. **Titolo** — Drift Detection MLOps Framework — Tesi
2. **Il problema** — drift in produzione, perché serve rilevarlo
3. **I tre tipi di drift** — feature, prediction, concept
4. **Architettura del framework** — Strategy Pattern + 3 livelli
5. **KS — l'idea** — confronto di due finestre via ECDF
6. **KS — come funziona** — D, p-value, soglia α
7. **KS — implementazione** — codice essenziale
8. **ADWIN — l'idea** — finestra adattiva
9. **ADWIN — come funziona** — bucket, ε_cut, delta
10. **ADWIN — implementazione** — codice essenziale, river
11. **KS vs ADWIN** — tabella di confronto
12. **FeatureDriftDetector** — l'orchestratore
13. **Validazione** — risultati sul benchmark Bernoulli
14. **Conclusioni e prossimi passi**

---

## 14. Riferimenti bibliografici

- Bifet, A., & Gavaldà, R. (2006). *Learning from Time-Changing Data with
  Adaptive Windowing*. Universitat Politècnica de Catalunya.
- Datar, M., Gionis, A., Indyk, P., & Motwani, R. (2002). *Maintaining
  stream statistics over sliding windows*. SIAM Journal on Computing.
- Montiel, J., Halford, M., Mastelini, S. M., et al. (2021). *River:
  machine learning for streaming data in Python*. JMLR.
- Gamma, E., Helm, R., Johnson, R., Vlissides, J. (1994). *Design Patterns:
  Elements of Reusable Object-Oriented Software*. Addison-Wesley.
