# Page-Hinkley — test sequenziale per stream continui

**Paper di riferimento**: Page, E. S. (1954). *Continuous Inspection Schemes*.
Biometrika, 41(1/2), pp. 100–115.
**Estensione**: Hinkley, D. V. (1971). *Inference about the change-point from
cumulative sum tests*. Biometrika, 58(3), pp. 509–523.
**Inquadramento nel drift detection**: Gama, J. et al. (2014), *A Survey on
Concept Drift Adaptation*, §3.2.1 (già in `Papers/`).

**Implementazione usata**: `river.drift.PageHinkley` (river 0.22.0)
**Wrapper nel framework**: `detectors/page_hinkley_strategy.py`

---

## 1. Il buco che copre

Guarda cosa avevi prima, e cosa ciascun algoritmo sa fare:

| Strategia | Su cosa lavora | Tipo di dato |
|---|---|---|
| KS | Due finestre di una feature | Continuo, **distribuzioni** |
| ADWIN | Media di uno stream | Numerico **limitato** in [0,1] |
| DDM | Tasso di errore | **Binario** (0/1) |
| **Page-Hinkley** | Media di uno stream | Continuo, **non limitato** |

C'era un caso scoperto, ed è **proprio quello che ti serve per la regressione**.

L'errore di un modello di regressione è `|y_true − y_pred|`: un numero reale
positivo **senza limite superiore**. Su quello:

- **DDM non è applicabile**: assume errori binari, la modellazione Bernoulli non
  ha senso su un reale.
- **ADWIN è applicabile ma fuori specifica**: le garanzie del Teorema 3.1 di
  Bifet & Gavaldà valgono per variabili limitate in [0,1]. Su un errore non
  limitato continua a funzionare, ma perde le garanzie formali che sono la sua
  ragione d'essere.
- **KS sarebbe applicabile**, ma confronta forme di distribuzione, non rileva un
  aumento sistematico del livello dell'errore, che è ciò che ti interessa.

Page-Hinkley è nato esattamente per questo: rilevare un cambiamento nella media
di un segnale continuo.

C'è anche un secondo motivo, più "da tesi": è il rappresentante di una **terza
famiglia teorica**. Nel tuo Cap. 2.4 descrivi test statistici a due campioni
(§2.4.1), misure di divergenza (§2.4.2), metodi error-based (§2.4.3) e metodi
window-based adattivi (§2.4.4). Con KS + ADWIN + DDM ne coprivi tre. Page-Hinkley
appartiene alla famiglia dell'**analisi sequenziale / CUSUM**, ed è quella che
storicamente viene prima di tutte le altre (1954). Averne un rappresentante
implementato rende la tua tassonomia completa invece che dichiarativa.

---

## 2. L'intuizione

Immagina di sommare, passo dopo passo, di quanto ogni valore si discosta dalla
media che hai visto finora.

- Se lo stream è **stazionario**, gli scarti sono a volte positivi e a volte
  negativi: la somma oscilla intorno a zero e non va da nessuna parte.
- Se la media **si sposta verso l'alto**, gli scarti diventano sistematicamente
  positivi: la somma comincia a crescere e non torna più indietro.

Il test non guarda il valore assoluto della somma, ma **quanto si è allontanata
dal minimo che aveva toccato**. Quando quella distanza supera una soglia, dichiara
il cambiamento.

È il principio del **CUSUM** (cumulative sum): un singolo valore anomalo non dice
niente, ma tanti piccoli scostamenti tutti nella stessa direzione si accumulano
fino a diventare evidenza.

---

## 3. La formalizzazione

Ad ogni passo `t` si calcola lo scarto dalla media corrente, ridotto di una
tolleranza `δ`, e lo si accumula:

```
    m_T  =  Σ (da t=1 a T)  ( x_t − media_t − δ )
```

Si tiene traccia del minimo storico di questa somma:

```
    M_T  =  min ( m_t ,  per t = 1 … T )
```

E il test è:

```
    PH_T  =  m_T − M_T  >  λ      ->  CAMBIAMENTO RILEVATO
```

### I due parametri, e cosa fanno davvero

**`δ` (delta, in river `delta`, default 0.005)** — è la **magnitudine di
cambiamento che accetti come rumore** e non vuoi segnalare. Sottraendo `δ` a
ogni scarto, imponi che un aumento debba essere almeno di quell'entità per
accumularsi. Se `δ` è troppo piccolo il test segnala anche derive irrilevanti;
se è troppo grande i cambiamenti piccoli passano inosservati.

**`λ` (lambda, in river `threshold`, default 50.0)** — è la **soglia di
allarme**, e governa il compromesso fondamentale:

| `λ` | Latenza | Falsi allarmi |
|---|---|---|
| basso | bassa (rileva presto) | molti |
| alto | alta (rileva tardi) | pochi |

Questo è **il** trade-off del drift detection, e Page-Hinkley lo espone in un
singolo parametro leggibile. È il motivo per cui è un ottimo candidato per il
grafico *falsi allarmi ↔ latenza* del capitolo di validazione: basta far variare
`λ` e si traccia la curva.

**`α` (alpha, default 0.9999)** — fattore di dimenticanza applicato alla somma
cumulata, che attenua lentamente il contributo del passato. Non è nel test
originale del 1954: è un'aggiunta pratica di river per evitare che la somma
diventi ingestibile su stream molto lunghi.

**`min_instances` (default 30)** — quanti campioni osservare prima di attivare il
test, per non decidere su una media stimata male. È l'analogo del `warm_start`
di DDM.

### Il parametro `mode`

river permette di monitorare aumenti (`'up'`), diminuzioni (`'down'`) o entrambi
(`'both'`, default della libreria).

Il wrapper mantiene `'both'` come default, coerentemente con la scelta adottata
in tutto il framework di non alterare i default delle implementazioni di
riferimento. **Ma negli esperimenti sul concept drift usiamo `mode='up'`**, ed è
impostato così nel `.env`: quando monitori l'errore di un modello ti interessa
solo che **cresca**. Un errore che cala è una buona notizia, non un allarme.

Segnalarlo comunque produrrebbe falsi positivi sistematici — ed è un dettaglio
che vale la pena scrivere in tesi, perché mostra che i parametri sono stati
scelti ragionando sul dominio e non copiati.

---

## 4. Pseudocodice

```
inizializza:  somma = 0,  minimo = 0,  media = 0,  n = 0

per ogni nuovo valore x:
    n <- n + 1
    media <- media + (x − media) / n          # media incrementale

    se n < min_instances:
        continua                              # troppo presto per decidere

    somma <- alpha * somma + (x − media − delta)
    minimo <- min(minimo, somma)

    se (somma − minimo) > threshold:
        segnala DRIFT
        reimposta somma, minimo, media, n      # riparti dal nuovo regime
```

---

## 5. Limiti (da scrivere in tesi)

1. **Rileva solo cambiamenti della media.** Come ADWIN, e a differenza del KS,
   è cieco ai cambiamenti di forma della distribuzione a media costante. Il
   caso `vicprice` di ELEC2 — la feature che il KS vede e ADWIN no — sarebbe
   invisibile anche a Page-Hinkley.

2. **Nessuna garanzia teorica sui falsi positivi.** ADWIN ha il Teorema 3.1 che
   fornisce limiti espliciti su falsi positivi e negativi. Page-Hinkley no: il
   comportamento va caratterizzato **empiricamente**, tarando `δ` e `λ`. È un
   punto a favore di ADWIN, ed è onesto dirlo.

3. **Due parametri da tarare invece di uno.** ADWIN ha il solo `δ`; il KS ha
   `window_size` e `alpha`, ma con un'interpretazione statistica diretta
   (livello di significatività). I parametri di Page-Hinkley non hanno
   un'interpretazione probabilistica immediata: `λ = 50` non significa "5% di
   falsi positivi". Vanno scelti con una ricerca sperimentale.

4. **Sensibile alla scala dei dati.** Poiché `δ` e `λ` sono in unità dello
   stream, gli stessi valori non funzionano su uno stream con errori di ordine
   0.01 e su uno con errori di ordine 1000. Su dati non normalizzati va
   ritarato ogni volta.

5. **Unilaterale per costruzione, se usato bene.** Con `mode='up'` non rileva i
   miglioramenti. È voluto nel nostro caso d'uso, ma va detto.

---

## 6. Come si integra nel framework

Anche Page-Hinkley è una **strategia** di livello 1, intercambiabile con le
altre. Insieme a DDM è la quarta e quinta estensione che il framework assorbe
senza modificare una riga del codice di orchestrazione.

Per la regressione serve però una piccola estensione, fatta contestualmente:
`ConceptDriftDetector` ora accetta un parametro **`error_fn`**, perché l'errore
si calcola in modo diverso a seconda del problema.

```python
from detectors.concept_drift_detector import (
    ConceptDriftDetector, errore_assoluto,
)
from detectors.page_hinkley_strategy import PageHinkleyStrategy

# Concept drift su un modello di REGRESSIONE
detector = ConceptDriftDetector(
    PageHinkleyStrategy,
    error_fn=errore_assoluto,   # |y_true − y_pred| invece di 0/1
    mode="up",
)

detector.update(y_pred, y_true)
detector.detect().drift_detected
```

Il default di `error_fn` resta l'errore binario, quindi gli esperimenti di
classificazione già esistenti non cambiano comportamento.

---

## 7. Primo risultato sperimentale

Test su stream continuo sintetico: errore distribuito come `N(1.0, 0.3)` fino a
t=1000, poi `N(3.0, 0.3)`. Drift vero a **t=1000**.

| Configurazione | Primo rilevamento | Latenza | Falsi allarmi |
|---|---|---|---|
| `PageHinkley(mode='up')`, parametri di default | t=1023 | **23 passi** | **0** |

Lo stesso test passando per `ConceptDriftDetector` con `error_fn=errore_assoluto`
(quindi la catena completa modello → errore → detector) rileva a **t=1027**,
latenza 27 passi, 0 falsi allarmi.

Per confronto, sui rispettivi scenari: ADWIN ha una latenza di ~55 passi sul
Bernoulli, DDM di 27–59 passi. Sono scenari diversi e i numeri **non sono
direttamente confrontabili** — lo diventeranno solo quando faremo girare tutte
le strategie sugli stessi stream con lo stesso protocollo multi-seed. Vale la
pena tenerlo a mente proprio perché è l'errore metodologico che stiamo
sistemando nel resto del lavoro.

---

## 8. Domande probabili in discussione

**"Perché Page-Hinkley e non ADWIN sull'errore di regressione?"**
ADWIN funzionerebbe, ma le sue garanzie teoriche valgono per variabili limitate
in [0,1] e l'errore di regressione non lo è. Useresti l'algoritmo fuori dalle
sue ipotesi, perdendo proprio ciò che lo rende preferibile. Page-Hinkley è
progettato per stream continui non limitati.

**"Qual è la differenza fra CUSUM e Page-Hinkley?"**
Page-Hinkley è una variante del CUSUM. Il CUSUM classico accumula gli scarti da
un valore di riferimento **noto a priori**; Page-Hinkley stima la media
**incrementalmente dai dati**, il che lo rende utilizzabile in streaming dove la
media di riferimento non la conosci.

**"Come avete scelto `δ` e `λ`?"**
Partendo dai default della libreria e caratterizzandone il comportamento
sperimentalmente al variare di `λ`, tracciando la curva latenza ↔ falsi allarmi.
*(Questo è l'esperimento da fare: quando sarà fatto, qui va il numero.)*

**"Il test è del 1954. Non è superato?"**
No, ed è un punto interessante: è ancora usato oggi proprio per la sua
semplicità, il costo di memoria costante e il costo computazionale O(1) per
campione. In un servizio che monitora migliaia di stream in parallelo, questo
conta più della raffinatezza statistica.

---

## 9. Riferimenti

- **Page, E. S. (1954).** *Continuous Inspection Schemes*. Biometrika 41(1/2),
  100–115 — **paper originale del test CUSUM**
- **Hinkley, D. V. (1971).** *Inference about the change-point from cumulative
  sum tests*. Biometrika 58(3), 509–523
- **Gama, J. et al. (2014).** *A Survey on Concept Drift Adaptation*. ACM
  Computing Surveys, §3.2.1 — inquadramento nel drift detection
  (già presente in `Papers/`)
- **Sebastião, R., Gama, J. (2009).** *A Study on Change Detection Methods*.
  EPIA 2009 — confronto sperimentale fra PH e altri metodi
- **Montiel, J. et al. (2021).** *River: machine learning for streaming data in
  Python*. JMLR
- **Ikonomovska, E., Gama, J., Džeroski, S. (2011).** *Learning model trees from
  evolving data streams*. Data Mining and Knowledge Discovery — definisce le
  varianti di drift `lea` / `gra` / `gsg` dei dataset Friedman che useremo per
  la regressione
