# Briefing per generazione diagrammi di flusso — KS e ADWIN

Documento pensato per essere passato a un LLM che genererà **due diagrammi
di flusso dettagliati**, uno per `KSStrategy` e uno per `ADWINStrategy`.
La struttura è: **INPUT → FUNZIONAMENTO → OUTPUT**, con tutti i passaggi
intermedi esplicitati.

---

## Istruzioni per chi genera i diagrammi

- **Lingua**: italiano
- **Formato consigliato**: Mermaid, Graphviz, o ASCII flow chart (a scelta)
- **Livello di dettaglio**: ogni decisione condizionale (if/else) deve
  essere visibile, ogni dato in transito deve essere etichettato
- **Stile**: chiaro e didattico, ottimizzato per essere letto a colpo
  d'occhio in una slide o in un capitolo di tesi
- **Un diagramma per algoritmo**, separati e autosufficienti

---

## Contesto comune ai due diagrammi

Entrambe le strategie estendono la classe astratta `BaseDriftDetector`,
che impone tre metodi:

- `update(value)` — aggiorna lo stato interno ricevendo un nuovo valore
- `detect()` — restituisce un oggetto `DriftResult`
- `reset()` — riporta lo stato iniziale

Vengono **utilizzate dal `FeatureDriftDetector`**, che internamente mantiene
N istanze indipendenti della strategia (una per feature). Quando arriva
un vettore `x = [x₀, x₁, ..., xₙ₋₁]`:

1. Il `FeatureDriftDetector` smista: `strategia_i.update(x[i])` per ogni i.
2. Subito dopo: `strategia_i.detect()` per ogni i.
3. Aggrega i verdetti con la regola `drift se ≥ k feature drittate`.

I diagrammi seguenti descrivono cosa succede **dentro la singola
strategia** quando riceve un valore (non l'orchestrazione esterna, già
descritta sopra).

---

## DIAGRAMMA 1 — `KSStrategy`

### INPUT

- **Tipo**: un singolo numero `float` (un valore della feature monitorata)
- **Chi lo passa**: il `FeatureDriftDetector.update()` chiama
  `strategy.update(value)`

### STATO INTERNO INIZIALE

- `reference`: un `deque(maxlen=window_size)` inizialmente **vuoto**
- `current`: un `deque(maxlen=window_size)` inizialmente **vuoto**
- `window_size`: parametro fisso (default 100)
- `alpha`: parametro fisso (default 0.05)

### FUNZIONAMENTO — passo per passo

#### Fase A — Metodo `update(value)`

1. **Decisione**: la `reference` è ancora piena meno di `window_size`?
   - **SÌ** → vai al passo 2
   - **NO** → vai al passo 3
2. **Riempi la reference**: `self.reference.append(value)`. Fine.
3. **Riempi/aggiorna la current**: `self.current.append(value)`.
   - Se la current era già piena (`window_size` elementi), il `deque`
     scarta automaticamente il valore più vecchio in testa (FIFO).
   - Fine.

#### Fase B — Metodo `detect()`

1. **Decisione**: la `current` ha già `window_size` elementi?
   - **NO (current non piena)** → vai al passo 2 (warming up)
   - **SÌ (current piena)** → vai al passo 3 (test KS)
2. **Warming up**: restituisci subito un `DriftResult` con:
   - `drift_detected = False`
   - `metadata = {"status": "warming_up"}`
   - Nessun confronto eseguito. Fine del detect().
3. **Esegui il test KS**:
   - Chiama `scipy.stats.ks_2samp(list(reference), list(current))`
   - Internamente scipy:
     - Costruisce l'ECDF della reference (funzione cumulativa empirica)
     - Costruisce l'ECDF della current
     - Misura la massima distanza verticale → **statistica D**
     - Calcola il **p-value** dalla statistica D e dalle dimensioni delle
       finestre
   - Riceve in output `(statistic, p_value)`
4. **Confronta il p-value con la soglia alpha**:
   - **`p_value < alpha`** → `drift = True`
   - **`p_value ≥ alpha`** → `drift = False`
5. **Restituisci il DriftResult**:
   - `drift_detected = drift`
   - `score = p_value`
   - `metadata = {"statistic": D, "alpha": alpha}`

### OUTPUT

- **Tipo**: un oggetto `DriftResult` con i seguenti campi:
  - `detector_name = "KS"`
  - `drift_detected`: booleano (True se drift rilevato)
  - `drift_type = "feature"`
  - `score`: il p-value calcolato (o `None` se warming up)
  - `metadata`: dizionario con dettagli (statistica D, alpha, o status)

### NOTE DI COMPORTAMENTO

- **Reference è FISSA** dopo il primo riempimento, non scorre mai.
- **Current scorre come FIFO** ad ogni nuovo update.
- **Il test KS viene rieseguito a ogni `detect()`** una volta che le finestre
  sono piene. Quindi `drift_detected` può oscillare tra True e False nel
  tempo, ma se il drift è netto resterà True a lungo (perché reference e
  current restano statisticamente diverse).

---

## DIAGRAMMA 2 — `ADWINStrategy`

### INPUT

- **Tipo**: un singolo numero `float` (un valore della feature monitorata)
- **Chi lo passa**: il `FeatureDriftDetector.update()` chiama
  `strategy.update(value)`

### STATO INTERNO INIZIALE

- `adwin`: un'istanza di `river.drift.ADWIN(delta=delta)`, inizializzata
  con `delta` (default 0.002)
- Internamente l'istanza di river contiene una **struttura a bucket
  esponenziali**: lista di bucket di varie dimensioni, ognuno con
  capacità (potenza di 2) e contenuto (somma di valori).
- All'inizio la struttura è vuota.

### FUNZIONAMENTO — passo per passo

#### Fase A — Metodo `update(value)`

1. **Inoltra il valore a river**: chiama `self.adwin.update(value)`.
2. **Dentro river**, in sequenza:
   - **Aggiungi nuovo bucket**: viene creato un bucket di capacità 1,
     contenuto = `value`, inserito in fondo (dal lato dei valori più
     recenti).
   - **Compressione (eventuale)**: se ora ci sono **M+1 bucket di
     dimensione 1** (con M=5 di default), i due bucket di dimensione 1
     più vecchi vengono fusi in un singolo bucket di dimensione 2
     (capacità=2, contenuto=somma dei due contenuti precedenti).
   - **Cascata**: se la fusione precedente ha portato a M+1 bucket di
     dimensione 2, anche questi si fondono in uno di dimensione 4. La
     cascata continua a salire finché non si stabilizza.
   - **Tentativi di taglio**: river prova tutti i possibili punti di
     taglio della finestra (intesa come sequenza ordinata dei bucket).
     I punti di taglio sono SOLO i **confini fra bucket consecutivi**,
     non tra singoli valori.
   - Per ciascun punto di taglio candidato:
     - Divide la finestra in `W₀` (parte vecchia: i bucket prima del
       taglio) e `W₁` (parte nuova: i bucket dopo il taglio)
     - Calcola la **media `μ(W₀)`** = (somma contenuti di W₀) / (somma
       capacità di W₀)
     - Calcola la **media `μ(W₁)`** = analogamente per W₁
     - Calcola la **soglia ε_cut** secondo l'Equazione 3.1 del paper:
       `ε_cut = √( (2/m) · σ²_W · ln(2/δ') ) + (2/3m) · ln(2/δ')`
       dove `δ' = δ/ln(n)`, `m` = media armonica di `|W₀|` e `|W₁|`,
       `σ²_W` = varianza osservata
     - Confronto: `|μ(W₀) − μ(W₁)| > ε_cut`?
       - **SÌ** → scarta i bucket di `W₀` (la finestra si accorcia),
         setta `drift_detected = True`, interrompi la ricerca dei tagli
       - **NO** → continua con il prossimo punto di taglio candidato
   - Se nessun taglio supera la soglia: `drift_detected = False`.

#### Fase B — Metodo `detect()`

1. Legge il flag interno `self.adwin.drift_detected`.
2. Legge la stima corrente della media: `self.adwin.estimation`.
3. Restituisce un `DriftResult` con:
   - `drift_detected = self.adwin.drift_detected`
   - `score = self.adwin.estimation` (la media corrente della finestra)
   - `metadata = {"delta": self.delta}`

### OUTPUT

- **Tipo**: un oggetto `DriftResult` con i seguenti campi:
  - `detector_name = "ADWIN"`
  - `drift_detected`: booleano (True solo nell'istante esatto del taglio)
  - `drift_type = "feature"`
  - `score`: la stima corrente della media nella finestra
  - `metadata`: dizionario con `{"delta": ...}`

### NOTE DI COMPORTAMENTO

- **Finestra adattiva**: la struttura a bucket cresce mentre i dati sono
  stabili e si accorcia drasticamente al momento del taglio.
- **Memoria logaritmica**: O(M·log(W/M)) bucket per una finestra di
  lunghezza W.
- **Segnale a impulso**: `drift_detected` è True solo nell'update che
  ha causato un taglio. Subito dopo torna a False (a meno che non ci
  sia un altro taglio).
- **Tempo logaritmico per update**: O(log W) ammortizzato, perché i
  punti di taglio testati sono O(log W) (uno per ogni confine di
  bucket).

---

## COMPONENTI VISIVI SUGGERITI

Per chi disegna i diagrammi:

### Per il KS

- **Nodi di stato**: due rettangoli "reference" e "current", riempiti
  visivamente come barre per indicare quanti elementi contengono
- **Nodi di decisione**: rombi per `len(reference) < window_size?` e
  `len(current) < window_size?` e `p_value < alpha?`
- **Nodi di azione**: rettangoli per "append a reference", "append a
  current (FIFO)", "esegui ks_2samp", "costruisci DriftResult"
- **Frecce etichettate**: con i dati che fluiscono (valore in ingresso,
  D e p_value calcolati, DriftResult finale)

### Per ADWIN

- **Nodo principale**: una "scatola river ADWIN" con dentro la struttura
  a bucket disegnata come catene di rettangoli di dimensione 1, 2, 4,
  8, …, ognuno con sopra `capacità/contenuto`
- **Nodi di azione interni a river**: "crea nuovo bucket dim 1",
  "compressione" (con freccia a cascata che ricorda l'effetto domino),
  "prova taglio ai confini"
- **Nodo di decisione**: rombo per `|μ_W0 − μ_W1| > ε_cut?`
- **Nodo di azione esterno**: "scarta W₀" (con disegno della finestra
  accorciata) e "setta drift_detected"

---

## DIFFERENZE CHIAVE DA EVIDENZIARE NEI DIAGRAMMI

| Aspetto | KS | ADWIN |
|---|---|---|
| Numero finestre | 2 (reference + current) | 1 (adattiva) |
| Riempimento iniziale | 400 passi prima del primo test reale | Inizia da subito |
| Cosa confronta | Forma intera della distribuzione | Media |
| Punti di confronto | Solo le ECDF intere | Tutti i confini fra bucket |
| Dimensione finestra | Fissa (window_size) | Variabile |
| Output del test | p-value (continuo) | True/False (impulso) |

Idealmente i due diagrammi dovrebbero usare **lo stesso stile visivo**
(stessi colori per gli stessi tipi di nodo) in modo che chi li guarda
possa confrontarli a colpo d'occhio.
