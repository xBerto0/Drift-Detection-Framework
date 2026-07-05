# drift_framework

Framework MLOps modulare per la **drift detection** su modelli di Machine
Learning in produzione. Sviluppato come progetto di tesi di laurea magistrale
in Ingegneria Informatica, curriculum Intelligenza Artificiale, Università
degli Studi di Palermo, in collaborazione con Engineering Ingegneria
Informatica S.p.A.

**Autore**: Alberto Conti
**Relatore**: Prof. Marco La Cascia
**Tutor aziendale**: Daniele Fakhoury (Engineering Ingegneria Informatica S.p.A.)

---

## Cosa fa

Rileva **drift** in un modello ML in produzione, cioè cambiamenti nel tempo
delle proprietà statistiche dei dati o del comportamento del modello.
Gestisce tre tipi di drift:

- **Feature drift**: cambiamenti nella distribuzione delle feature in input
- **Prediction drift**: cambiamenti nella distribuzione delle predizioni
- **Concept drift**: cambiamenti nel tasso di errore del modello

Il framework è **modulare** (Strategy Pattern), **model-agnostic** (non fa
assunzioni sulla natura del modello ML) e **estensibile** (aggiungere una
nuova strategia richiede una sola classe).

## Strategie implementate

- **KSStrategy** — test di Kolmogorov-Smirnov a due campioni
  (via `scipy.stats.ks_2samp`)
- **ADWINStrategy** — algoritmo ADWIN a finestra adattiva
  (via `river.drift.ADWIN`)

Entrambe le strategie sono intercambiabili nei detector.

## Struttura del progetto

```
drift_framework/
├── detectors/                Interfacce e strategie di drift detection
├── models/                   Interfaccia astratta del modello
├── results/                  Output degli esperimenti e dataclass DriftResult
├── data/                     Dataset (ELEC2) e generatore sintetico
├── monitoring/               Servizio di monitoring centrale (in sviluppo)
├── experiment_*.py           Script sperimentali
├── docs/                     Documentazione tecnica e materiale di tesi
├── presentations/            Slide per il relatore
└── thesis/                   File LaTeX della tesi
```

## Come lanciare gli esperimenti

Dalla cartella radice del progetto:

```powershell
# Esperimenti su dataset sintetico Bernoulli
python experiment_ks_bernoulli.py
python experiment_adwin_bernoulli.py

# Esperimenti su dataset reale ELEC2
python experiment_ks_elec2.py
python experiment_adwin_elec2.py
```

Gli output vengono salvati in `results/<algoritmo>/<algoritmo>_<tipodrift>_<dataset>.txt`.

## Requisiti

- Python 3.10+
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `river`

Installazione:

```powershell
pip install numpy scipy pandas scikit-learn river
```

## Riferimenti principali

- Bifet, A., Gavaldà, R. (2007). *Learning from Time-Changing Data with
  Adaptive Windowing*. SIAM Int. Conf. on Data Mining.
- Gama, J. et al. (2014). *A Survey on Concept Drift Adaptation*. ACM
  Computing Surveys.
- Montiel, J. et al. (2021). *River: machine learning for streaming data
  in Python*. JMLR.

Bibliografia completa in `docs/thesis_writing_briefing.md`.

## Stato del lavoro

Progetto di tesi in corso. Lo stato aggiornato e le prossime tappe sono
documentate in [HANDOFF.md](HANDOFF.md).
