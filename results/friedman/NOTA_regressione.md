# Friedman — concept drift su modelli di regressione

**Data**: 2026-08-23
**Riproducibile con**: `python -m evaluation.runner_friedman`
**Modello**: `DecisionTreeRegressor(max_depth=8)`, statico, addestrato sui
primi 1.000 campioni e mai aggiornato.
**Stream monitorato**: errore assoluto `|y_vero − y_predetto|`.

---

## 1. Perche' la regressione non e' una ripetizione della classificazione

La proposta di tesi parla di "classification **and** regression", e il caso
della regressione non si ottiene semplicemente sostituendo il modello: cambia
la natura dello stream monitorato, e con essa l'insieme delle strategie
applicabili.

| Strategia | Errore binario (classificazione) | Errore reale non limitato (regressione) |
|---|---|---|
| **DDM** | Elettivo | **Non applicabile** — assume errori bernoulliani |
| **ADWIN** | Nativo | Applicabile ma **fuori specifica** — il Teorema 3.1 vale per variabili in [0,1] |
| **KS** | Sub-ottimo (dati discreti) | **Nel proprio dominio** — l'errore e' continuo |
| **Page-Hinkley** | Applicabile | **Elettivo** — nato per la media di segnali continui |

Il framework rende la distinzione esplicita: `DDMStrategy.update()` solleva
un'eccezione se riceve un valore non binario, invece di produrre numeri privi
di significato. Il comportamento e' verificato in `tests/test_strategie.py`.

L'estensione richiesta e' stata minima: un parametro `error_fn` in
`ConceptDriftDetector`, che sceglie fra errore binario (default, invariato) ed
errore assoluto. Nessuna modifica alle strategie ne' al resto
dell'orchestrazione.

---

## 2. I dataset

Tre varianti di `river.datasets.synth.FriedmanDrift`, generate una volta con
seed 42 e versionate in repository. Le 10 feature sono uniformi in [0,1]; a
cambiare e' la funzione che lega feature e target. E' quindi **concept drift
puro**: `P(X)` costante, `P(y|X)` variabile.

| Variante | Tipo di drift | Punti di drift | MAE del modello statico |
|---|---|---|---|
| `lea` | Locale, brusco, a regioni crescenti | 5.000, 10.000, 15.000 | 2,576 |
| `gra` | Globale, brusco, **ricorrente** | 7.000, 14.000 | 2,943 |
| `gsg` | Globale, **graduale** (transizione 1.000) | 7.000, 14.000 | 3,424 |

---

## 3. Risultati sul concept drift

| Variante | Strategia | Rilevati | Falsi allarmi | Latenza media |
|---|---|---|---|---|
| `lea` | KS | 2/3 | 104 | 87 |
| `lea` | ADWIN | 2/3 | **0** | 659 |
| `lea` | **Page-Hinkley** | **3/3** | 36 | 362 |
| `lea` | Ensemble-MAJORITY | 2/3 | 11 | 659 |
| `gra` | KS | 1/2 | 11 | **28** |
| `gra` | **ADWIN** | **2/2** | **0** | 51 |
| `gra` | Page-Hinkley | 1/2 | 32 | 22 |
| `gra` | **Ensemble-MAJORITY** | **2/2** | 26 | 51 |
| `gsg` | KS | 1/2 | 10 | 49 |
| `gsg` | ADWIN | 1/2 | 2 | 143 |
| `gsg` | **Page-Hinkley** | **2/2** | 38 | 158 |
| `gsg` | **Ensemble-MAJORITY** | **2/2** | 36 | 158 |

**Il compromesso e' netto e coerente su tutte e tre le varianti.**
Page-Hinkley e' l'unico che raggiunge la copertura piena su `lea` e `gsg`, ma
paga con 32-38 falsi allarmi. ADWIN e' l'unico a non produrne mai su `lea` e
`gra`, ma perde drift. Non esiste un vincitore: esiste una scelta, che dipende
dal costo relativo di un allarme mancato e di un allarme inutile.

**Il drift ricorrente (`gra`) e' il caso piu' istruttivo.** L'errore del
modello salta a ~4,3 al primo drift e **ritorna a ~2,1** al secondo, quando il
concetto iniziale si ripresenta: il modello statico, addestrato su quel
concetto, torna improvvisamente ad andare bene. ADWIN cattura entrambe le
transizioni con 0 falsi allarmi ed e' il comportamento migliore osservato in
tutto l'esperimento. E' coerente con quanto misurato sui Bernoulli sintetici,
dove ADWIN era l'unico con tasso di mancate rilevazioni nullo sul ricorrente.

**Nota su Page-Hinkley con `mode='up'`**: rileva solo la crescita dell'errore.
Sul secondo drift di `gra`, che e' un *miglioramento*, e' cieco per
costruzione. Il fatto che risulti 1/2 su quella variante non e' un difetto ma
la conseguenza attesa della configurazione scelta — ed e' l'ennesima conferma
dell'asimmetria direzionale gia' misurata sui sintetici.

---

## 4. Il controllo negativo, e il difetto che ha rivelato

Poiche' in Friedman `P(X)` e' costante per costruzione, **un detector di
feature drift non deve rilevare nulla**. Ogni segnalazione e' per definizione
un falso positivo. E' il controllo negativo che dimostra sperimentalmente la
differenza fra data drift e concept drift.

Il controllo **e' fallito**, e in modo istruttivo:

| Configurazione | Episodi | Tempo in allarme |
|---|---|---|
| *Atteso senza correzione:* `1 − (1−0,05)¹⁰` | — | **40,1%** |
| KS, nessuna correzione | 519 | **33,3%** |
| KS + **Bonferroni** | 69 | **2,9%** |
| KS + **Benjamini-Hochberg** | 71 | **3,1%** |
| ADWIN | **0** | **0,0%** |

Il 33,3% osservato e' vicinissimo al 40,1% previsto dalla teoria del test
multiplo. Non e' un difetto del test di Kolmogorov-Smirnov: e' un difetto
della **regola di aggregazione**, che con `k=1` dichiara drift globale appena
uno qualsiasi degli `n` test indipendenti risulta positivo, senza tener conto
di quanti test sono stati eseguiti.

Applicando la correzione di Bonferroni (soglia `alpha/n`) il tasso scende al
**2,9%**, sotto il livello nominale del 5% — esattamente cio' che la
correzione deve garantire. Benjamini-Hochberg, meno conservativo, arriva al
3,1%.

**ADWIN produce 0 falsi positivi** e supera il controllo senza bisogno di
alcuna correzione: non essendo un test di ipotesi, non soffre del problema.

### Ricaduta sui risultati precedenti

Il difetto era gia' presente negli esperimenti su ELEC2, dove le feature sono
8 e la probabilita' teorica di falso positivo per passo e' `1 − 0,95⁸ = 34%`.
La saturazione permanente del KS osservata su ELEC2 e' quindi almeno in parte
imputabile a questa causa, oltre che agli artefatti di imputazione del
dataset. La correzione e' ora disponibile in `FeatureDriftDetector` tramite i
parametri `correzione` e `alpha_correzione`; il default resta `None`, che
conserva il comportamento storico.

---

## 5. Limite dichiarato

I tre dataset sono file fissi con seed 42. Le metriche di questa nota
provengono quindi da **una singola esecuzione per variante**, su 7 punti di
drift complessivi, e non da una media su piu' ripetizioni come negli
esperimenti Bernoulli. Vanno lette come indicative: le differenze piccole fra
strategie non sono significative, quelle grandi (3/3 contro 1/2, 0 falsi
allarmi contro 38) lo sono.

Il controllo negativo fa eccezione: essendo misurato su 18.000 passi, il
confronto fra 33,3% e 2,9% e' robusto indipendentemente dal seed.
