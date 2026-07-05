# Analisi dei risultati sperimentali

**Data**: 2026-07-05
**Autore**: Alberto Conti
**Dataset e algoritmi coperti**: KS e ADWIN su Bernoulli sintetica (feature drift) e su ELEC2 (feature, prediction, concept drift)

---

## 1. Riepilogo generale

Sono stati eseguiti 7 esperimenti che coprono tutte le combinazioni implementate
di **algoritmo di detection** (Kolmogorov-Smirnov, ADWIN) × **tipo di drift**
(feature, prediction, concept) × **dataset** (Bernoulli sintetica, ELEC2). Gli
esperimenti sintetici hanno lo scopo di validare che l'implementazione funzioni
correttamente in scenari controllati con drift noto. Gli esperimenti su ELEC2
introducono un dataset reale del mercato elettrico australiano (2.5 anni di
dati, 45.312 istanze, 8 feature numeriche) su cui è addestrato un classificatore
Naive Bayes statico. Il classificatore viene poi valutato in modalità streaming
sui restanti 44.812 campioni, mentre in parallelo i detector monitorano feature
in ingresso, predizioni ed errori.

---

## 2. Risultati per singolo test

### 2.1 KS Feature Drift su Bernoulli sintetica

Il KS ha correttamente identificato la sola feature perturbata (`f1_drift`),
senza falsi positivi sulle feature stabili (`f0_stable`, `f2_stable`). Primo
drift rilevato al passo t≈1100, cioè circa 100 passi dopo il cambio brusco
introdotto in t=1000. Il ritardo è coerente con la teoria: la current window
scorrevole deve popolarsi di valori nuovi prima che il KS test rilevi una
differenza significativa fra ECDF di reference e current. Segnalazione
continuativa nel resto del flusso.

### 2.2 ADWIN Feature Drift su Bernoulli sintetica

ADWIN ha rilevato lo stesso drift al passo t=1055, cioè con una latenza di
soli 55 passi — **circa il doppio della velocità del KS**. Nessun falso
positivo. Una sola segnalazione, coerente con il comportamento a impulso
predetto dal Teorema 3.1 del paper Bifet & Gavaldà: ADWIN tagliaa la
finestra al momento del cambiamento e da quell'istante in poi lavora sulla
distribuzione nuova, senza continuare a segnalare.

### 2.3 KS Feature Drift su ELEC2

44.413 segnalazioni di drift globale su 44.812 passi possibili (99.1% dei
passi utili). Il framework funziona correttamente: la feature `period` (ciclica
giornaliera, stazionaria per costruzione) ha 0 segnalazioni; la feature `date`
(timestamp monotona) ha il 100% di segnalazioni per falso positivo strutturale;
le altre 6 feature oscillano fra il 62% e il 99%. La granularità per-feature
dimostra la discriminazione corretta del test sui singoli stream, ma il numero
aggregato mostra **over-sensitivity del KS con parametri standard**
(`window_size=200`, `alpha=0.05`) su un dataset reale fortemente
non-stazionario.

### 2.4 KS Prediction Drift su ELEC2

44.180 segnalazioni su 44.812 passi (98.6%). Comportamento atteso teoricamente:
il KS è progettato per distribuzioni continue e sui flussi binari (predizioni
Naive Bayes UP/DOWN) perde potere statistico. `scipy.stats.ks_2samp` emette
warning continui sul fallback all'approssimazione asintotica, confermando la
difficoltà operativa. Nessuna informazione utile ricavabile: un solo stream,
nessuna granularità per-sotto-livello, over-sensitivity più inadeguatezza al
dominio.

### 2.5 ADWIN Feature Drift su ELEC2

558 segnalazioni totali, ~80 volte meno del KS. Distribuzione per feature molto
più concentrata:

| Feature | Segnalazioni |
|---|---|
| `date` | 11 |
| `day` | 467 |
| `period` | 0 |
| `nswprice` | 10 |
| `nswdemand` | 45 |
| `vicprice` | 0 |
| `vicdemand` | 72 |
| `transfer` | 69 |

Il caso più interessante è `vicprice`: KS ne rilevava drift al 62% dei passi,
ADWIN 0 volte. Significa che la **forma della distribuzione di `vicprice`
cambia nel tempo, ma la sua media resta stabile**. Il KS vede il cambio di
forma (via ECDF); ADWIN, che monitora solo la media, correttamente non vede
nulla. Dimostrazione empirica diretta del fatto che i due algoritmi
**rispondono a segnali statistici diversi**.

### 2.6 ADWIN Prediction Drift su ELEC2

Solo **2 segnalazioni**, entrambe nella fase iniziale dello streaming
(t=723 e t=1075). Dopo l'assestamento iniziale del modello statico, la
distribuzione delle predizioni UP/DOWN resta stabile per tutti i 45.000
passi successivi. Osservazione critica: la stabilità della distribuzione
delle predizioni non implica che il modello sia corretto — vedi il concept
drift, che invece rileva un forte degrado dell'accuracy. Il modello sta
sbagliando in modo consistente, non erratico.

### 2.7 ADWIN Concept Drift su ELEC2

66 segnalazioni distribuite lungo tutto il flusso. Primo segnale al passo
t=1619, correlato temporalmente con il crollo di accuracy dal 83.4% (passo
1499) al 58.4% (passo 2499). Cluster di segnalazioni in periodi di forte
instabilità del classificatore (es. t=2931-3891, t=5395-6803,
t=27091-27763). L'ultimo cluster (t=44371-44979) coincide con la fase finale
di peggior accuracy dell'esperimento (43-45% ai passi 40499-41499).

Il concept drift è l'unico caso in cui il KS non è stato testato: come
discusso in fase di design, il KS su uno stream binario di errori è
sub-ottimale e non offre informazione discriminativa aggiuntiva rispetto ad
ADWIN, che è nativo per questo caso d'uso.

---

## 3. Confronto diretto KS vs ADWIN su ELEC2

### 3.1 Feature drift

| Metrica | KS | ADWIN | Rapporto |
|---|---|---|---|
| Segnalazioni aggregate | 44.413 | 558 | ~80:1 |
| Falsi positivi su `period` | 0 | 0 | Concordi |
| Segnalazioni su `date` (drift strutturale, non informativo) | 44.413 | 11 | ADWIN meno rumoroso |
| Segnalazioni su `vicprice` | 27.856 | 0 | Interpretazioni divergenti |
| Feature "attive" (>10 segnalazioni ADWIN) | Tutte tranne `period` | `day`, `nswdemand`, `vicdemand`, `transfer` | ADWIN più selettivo |

**Vantaggi del KS**: cattura variazioni fini di forma della distribuzione (es.
`vicprice`); utile in scenari dove interessa lo shape drift, non solo lo
mean drift.
**Vantaggi di ADWIN**: molto meno rumoroso, informazione più concentrata,
adatto a scenari operativi dove il numero di alert deve restare gestibile;
non genera falsi positivi strutturali su feature come `date`.

### 3.2 Prediction drift

| Metrica | KS | ADWIN | Rapporto |
|---|---|---|---|
| Segnalazioni totali | 44.180 | 2 | ~22.000:1 |
| Warning scipy | Sì (continui) | No | ADWIN nativo per binari |
| Utilità operativa | Nessuna (rumore continuo) | Alta (2 segnali all'inizio, poi silenzio informativo) | ADWIN elettivo |

Su stream binari **ADWIN è la scelta corretta senza ambiguità**. Il KS può
essere lasciato come baseline di riferimento per la tesi, ma non offre
informazione utilizzabile.

### 3.3 Concept drift

Testato solo con ADWIN, per scelta di design. 66 segnalazioni, correlate
temporalmente con crolli di accuracy del classificatore. Il KS non è
appropriato per stream binari di errori; l'inclusione del solo ADWIN è
coerente con la letteratura (Bifet & Gavaldà 2006, sezione 5.1: "ADWIN
esterno").

---

## 4. Osservazioni trasversali

### 4.1 Velocità di rilevamento

- Su drift bruschi sintetici (Bernoulli), **ADWIN è ~2× più veloce del KS**
  (55 vs 100 passi di latenza). Confermato dal Teorema 3.1: ADWIN taglia la
  finestra non appena la differenza fra medie supera `ε_cut`.
- Su drift graduali reali (ELEC2), la differenza si amplifica: ADWIN
  identifica pochi punti significativi in modo tempestivo, il KS annega il
  segnale in un mare di segnalazioni continue.

### 4.2 Correttezza della rilevazione

- Entrambi gli algoritmi concordano sulla stazionarietà della feature `period`
  (0 segnalazioni per entrambi). Questo è un check di consistenza importante:
  se avessero disagreed su una feature palesemente stazionaria, il framework
  avrebbe un bug.
- Su `vicprice` divergono: interpretazione teoricamente giustificata dalla
  natura diversa dei due algoritmi (shape drift vs mean drift).
- Sul concept drift, ADWIN mostra correlazione temporale con crolli di
  accuracy del classificatore, evidenza empirica della sua utilità.

### 4.3 Numero di segnali e usabilità operativa

- **KS** produce migliaia o decine di migliaia di segnalazioni: adatto a
  monitoring continuo con soglia di aggregazione (es. "drift è confermato
  solo se persiste per N passi"), non adatto a sistemi di alerting diretto.
- **ADWIN** produce da 0 a poche centinaia di segnalazioni: adatto a sistemi
  di alerting diretto (ogni segnalazione può ragionevolmente diventare un
  evento nel sistema di monitoring MLOps).

### 4.4 Comportamento su dati binari

- **KS su binari**: subottimale, evidenziato sia da warning di scipy che da
  numeri di segnalazioni indistinguibili dal rumore.
- **ADWIN su binari**: nativo, produce risultati puliti e informativi.

### 4.5 Vantaggi e svantaggi sintetici

| Aspetto | KS | ADWIN |
|---|---|---|
| Sensibilità | Alta (rileva shape drift) | Media (solo mean drift) |
| Rumore | Alto su dati reali non-stazionari | Basso |
| Dati continui | Ottimo | Buono |
| Dati binari/discreti | Scarso | Nativo |
| Garanzie teoriche | Test KS classico | Teorema 3.1 (bound su FP/FN) |
| Adattività | No (finestre fisse) | Sì (finestra variabile) |
| Facilità di tuning | Semplice (window_size, alpha) | Molto semplice (solo delta) |

---

## 5. Riepilogo finale e commento generale

L'insieme dei risultati fornisce una prima validazione sperimentale
completa del framework `drift_framework`. Il pattern di comportamento
osservato è coerente su tutti i dataset e per tutti i tipi di drift:

1. **Il KS è un rilevatore fine ma rumoroso**, che restituisce una
   quantità di segnale molto elevata anche in scenari a bassa "vera"
   variazione. Sui dati sintetici Bernoulli funziona come atteso (drift
   corretti, nessun falso positivo su feature stabili). Sui dati reali
   di ELEC2 la sua sensibilità diventa un limite: la per-feature analysis
   resta informativa, ma il segnale aggregato è troppo denso per un uso
   operativo diretto.

2. **ADWIN è un rilevatore parsimonioso e coerente con le sue garanzie
   teoriche**. Produce pochi segnali, ben localizzati temporalmente, e su
   scenari con drift reale (Bernoulli e concept drift ELEC2) i suoi punti
   di segnalazione correlano con eventi statistici concreti. Sui dati
   binari (predizioni ed errori del classificatore) è nettamente
   preferibile al KS.

3. **I due algoritmi non sono in competizione ma complementari**: KS vede
   cambiamenti di forma della distribuzione, ADWIN vede cambiamenti di
   media. Il caso `vicprice` su ELEC2 è la dimostrazione più chiara di
   questa complementarità: il KS lo vede drittato, ADWIN no, ed entrambi
   hanno ragione dal loro punto di vista statistico.

4. **La scelta architetturale del framework — Strategy Pattern con
   strategie intercambiabili — è validata sperimentalmente**: entrambi
   gli algoritmi si integrano nello stesso `FeatureDriftDetector`,
   `PredictionDriftDetector` e `ConceptDriftDetector` senza alcuna
   modifica al codice di orchestrazione. Cambiare `strategy_cls` da
   `KSStrategy` ad `ADWINStrategy` è sufficiente a passare da un
   comportamento all'altro.

5. **Direzione operativa emersa**: in un sistema MLOps di produzione, si
   può ragionevolmente proporre di utilizzare **ADWIN come detector di
   allarme principale** (basso volume di segnali, alta significatività) e
   **KS come strumento di analisi diagnostica offline** (esplorazione
   dettagliata di quale feature specifica sta contribuendo al drift). Le
   due strategie non si escludono e possono coesistere nella stessa
   pipeline.

Nel Capitolo 6 della tesi questi risultati costituiscono l'evidenza
empirica che giustifica sia la scelta architetturale (multi-strategy
framework) sia le decisioni di design specifiche (ADWIN elettivo per
concept drift, entrambi disponibili per feature e prediction drift).
