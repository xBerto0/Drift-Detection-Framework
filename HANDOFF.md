# Handoff — drift_framework

Documento di passaggio di consegne per riprendere il lavoro su questa tesi
in una nuova chat. Letto questo, una nuova istanza dell'agente sa esattamente
da dove ripartire.

**Data ultimo aggiornamento**: 2026-06-16

---

## 1. Cosa sto facendo (in 3 righe)

Sto sviluppando `drift_framework`, un sistema MLOps modulare per il drift
detection. È il progetto della mia tesi di laurea. L'architettura è basata
sul Strategy Pattern: tre tipi di drift (feature / prediction / concept),
ognuno con un detector che usa strategie intercambiabili (KS, ADWIN, ecc.).

---

## 2. A che punto sono

Settimana 1-2 e 3-6 della roadmap di tesi: **COMPLETATE per il feature drift**.

### Implementato e testato

- **Interfacce base**: `BaseDriftDetector`, `BaseModel`, `DriftResult`
- **Generatore dati sintetici**: `bernoulli_with_abrupt_drift` (fedele a
  Figura 2 di Bifet & Gavaldà 2006)
- **Due strategie SEPARATE e INDIPENDENTI per il feature drift**:
  - `KSStrategy` (via `scipy.stats.ks_2samp`)
  - `ADWINStrategy` (via `river.drift.ADWIN`)
- **Orchestratore**: `FeatureDriftDetector` (parametro `k` per aggregazione
  multi-feature)
- **Due test sintetici Bernoulli funzionanti**:
  - KS rileva drift al passo ~1100 (latenza ~100 passi)
  - ADWIN rileva drift al passo ~1055 (latenza ~55 passi)
  - Entrambi senza falsi positivi sulle feature stabili

### Documentazione tecnica scritta

- `docs/adwin_overview.md` — panoramica completa di ADWIN come materiale di tesi
- `presentations/slides_ks_strategy.md` — slide informali per il relatore

---

## 3. Prossimi passi possibili (scegliere uno)

In ordine di priorità suggerita:

1. **Prediction Drift** — implementare `PredictionDriftDetector` che applica le
   stesse strategie (KS, ADWIN) sullo stream delle predizioni di un modello,
   invece che sulle feature di input.

2. **Test su dataset reale ELEC2** — il dataset citato dal paper di ADWIN,
   per validare anche fuori dal sintetico.

3. **Concept Drift** — `ConceptDriftDetector` su stream di errori (richiede
   `y_true`). Si possono aggiungere strategie DDM e Page-Hinkley.

4. **Documentazione KS** — scrivere `docs/ks_overview.md` analogo a quello di
   ADWIN, per avere il materiale di tesi completo su entrambe le strategie.

5. **Aggiornare gli esperimenti** — ad esempio salvare i risultati in
   `results/` invece che stamparli a video.

---

## 4. Struttura attuale del progetto

```
drift_framework/
├── models/
│   └── base_model.py              # interfaccia astratta BaseModel
├── detectors/
│   ├── base_detector.py           # interfaccia astratta BaseDriftDetector
│   ├── ks_strategy.py             # KSStrategy (scipy.stats.ks_2samp)
│   ├── adwin_strategy.py          # ADWINStrategy (river.drift.ADWIN)
│   └── feature_drift_detector.py  # orchestratore multi-feature
├── results/
│   └── drift_result.py            # dataclass DriftResult
├── monitoring/
│   └── drift_monitoring_service.py  # (vuoto, settimane 7-8)
├── data/
│   └── synthetic_generator.py     # bernoulli_with_abrupt_drift
├── docs/
│   ├── adwin_overview.md          # materiale di tesi su ADWIN
│   └── handoff.md                 # (questo file, se messo qui)
├── presentations/
│   └── slides_ks_strategy.md      # slide informali per il relatore
├── experiment_ks_bernoulli.py     # test KS, da lanciare dal root
├── experiment_adwin_bernoulli.py  # test ADWIN, da lanciare dal root
└── HANDOFF.md                     # questo file
```

---

## 5. Decisioni di design importanti

- **KS e ADWIN sono due strategie SEPARATE, non un ensemble.** Si scelgono
  una alla volta passandole a `FeatureDriftDetector(strategy_cls=...)`.
- **Per ADWIN si usa river**, non si reimplementa l'algoritmo a mano.
  L'argomento difensivo per la tesi: il valore aggiunto della tesi è
  l'architettura del framework, non la reimplementazione di algoritmi
  consolidati.
- **Per il KS si usa scipy.stats.ks_2samp**: stessa motivazione.
- **Parametri esposti nei wrapper, di default standard del paper/libreria**:
  - `KSStrategy(window_size=100, alpha=0.05)`
  - `ADWINStrategy(delta=0.002)`
  - `FeatureDriftDetector(strategy_cls, n_features, k=1, feature_names=None, **strategy_kwargs)`
- **Reference window auto-fill dallo stream** (non da training data), accettato
  come limite consapevole. Per estensione futura: parametro opzionale
  `reference_data=` al costruttore.

---

## 6. Come lanciare i test

Dalla cartella radice `drift_framework/`:

```powershell
# Test KS
python experiment_ks_bernoulli.py

# Test ADWIN (richiede: pip install river)
python experiment_adwin_bernoulli.py
```

---

## 7. Preferenze e stile (anche nelle memorie persistenti)

- **Lingua di lavoro**: italiano.
- **Codice**: semplice e leggibile, niente costrutti "AI-style" (no
  comprehension annidate, no metaclassi, type hints semplici).
- **Slide/deliverable**: minimal e informali, solo l'essenziale.
- **Spiegazioni**: a parole semplici, didattiche, riutilizzabili per spiegare
  ad altri (relatore, esame).

---

## 8. File chiave da leggere per recuperare il contesto tecnico

In ordine di priorità:

1. **`docs/adwin_overview.md`** — la documentazione tecnica completa di ADWIN,
   scritta come pezzo di tesi. Contiene tutto: come funziona, perché si usa
   ADWIN2, le formule, le garanzie teoriche, l'integrazione nel framework, i
   risultati sperimentali.
2. **`detectors/feature_drift_detector.py`** — il core dell'orchestrazione
   multi-feature.
3. **`detectors/ks_strategy.py`** e **`detectors/adwin_strategy.py`** — le due
   strategie concrete.
4. **`experiment_ks_bernoulli.py`** e **`experiment_adwin_bernoulli.py`** —
   gli script di test, mostrano come si usa il framework dall'esterno.

---

## 9. Cosa dire all'agente nella nuova chat (prima riga di apertura)

Esempio:

> "Ciao, ho aperto una nuova chat. Riprendiamo dal drift_framework: ho già
> implementato KS e ADWIN per il feature drift come strategie separate, tutto
> testato su Bernoulli sintetica. Leggi `HANDOFF.md` nella root del progetto
> per il contesto completo, poi voglio [partire con prediction drift /
> testare ELEC2 / scrivere il capitolo di tesi sul KS / ecc.]."

L'agente caricherà automaticamente le memorie persistenti
(profilo utente, architettura, roadmap, preferenze) e leggendo questo file
avrà il quadro tecnico aggiornato.

---

## 10. Riferimenti rapidi

- **Paper di riferimento principale**: Bifet & Gavaldà (2006), *Learning from
  Time-Changing Data with Adaptive Windowing*. Il PDF tradotto è in
  `c:/Users/conti/OneDrive/Desktop/TESI/Papers/`.
- **Librerie**: `scipy.stats.ks_2samp`, `river.drift.ADWIN`, `numpy`.
- **Dataset reale candidato per validazione**: ELEC2 (lo stesso del paper),
  disponibile su OpenML o via `river.datasets.Elec2`.
