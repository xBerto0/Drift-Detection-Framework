# Page-Hinkley — fondamenti dalla letteratura

Documento di riferimento scientifico sul test di Page-Hinkley. Ogni affermazione
tecnica riporta la fonte da cui proviene. È il materiale destinato al Capitolo 4
della tesi.

Per la parte pratica — integrazione nel framework, parametri esposti, risultati
sperimentali ottenuti — vedere `docs/page_hinkley_overview.md`, che è
complementare a questo e non lo sostituisce.

---

## Nota preliminare sulle fonti

**Il paper primario non è liberamente accessibile.**

> Page, E. S. (1954). *Continuous Inspection Schemes*. Biometrika, 41(1/2),
> pp. 100–115. Oxford University Press.
> DOI: [10.2307/2333009](https://doi.org/10.2307/2333009)

È distribuito tramite JSTOR e richiede abbonamento. **Va recuperato tramite
l'accesso istituzionale Unipa**, che di norma copre JSTOR.

Va segnalata una particolarità storica del riferimento: il paper del 1954
introduce il **CUSUM**, non il test che oggi si chiama "Page-Hinkley". La
denominazione composta si è consolidata nella letteratura successiva, che
attribuisce a Hinkley l'estensione all'inferenza sul punto di cambiamento:

> Hinkley, D. V. (1971). *Inference about the change-point from cumulative sum
> tests*. Biometrika, 58(3), pp. 509–523.

Nella letteratura sul concept drift entrambe le varianti vengono attribuite a
Page 1954 — così fa esplicitamente [SK17], che cita `(Page, 1954)` sia per
CUSUM sia per PHT. Vale la pena saperlo perché è una domanda che può arrivare.

Quanto segue è ricostruito da **fonti secondarie sottoposte a revisione
paritaria che descrivono formalmente l'algoritmo**, tutte verificabili e due
delle quali già presenti in `Papers/`.

### Fonti effettivamente consultate

| Sigla | Riferimento | Disponibilità |
|---|---|---|
| **[SK17]** | Sethi, T. S., Kantardzic, M. (2017). *On the reliable detection of concept drift from streaming unlabeled data*. Expert Systems with Applications, 82, 77–99. [arXiv:1704.00023](https://arxiv.org/abs/1704.00023) | in `Papers/`, e accesso aperto |
| **[GA14]** | Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation*. ACM Computing Surveys, 46(4), 1–37. | in `Papers/` ⚠️ versione ridotta |
| **[LU19]** | Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G. (2019). *Learning under Concept Drift: A Review*. IEEE TKDE, 31(12). [arXiv:2004.05785](https://arxiv.org/abs/2004.05785) | in `Papers/`, e accesso aperto |

---

## 1. Collocazione tassonomica

[SK17] colloca sia CUSUM sia PHT nella famiglia dell'**analisi sequenziale**,
all'interno dei metodi di rilevamento *espliciti (supervisionati)*:

| Famiglia | Metodi |
|---|---|
| **Analisi sequenziale** | **CUSUM** (Page, 1954), **PHT** (Page, 1954), LFR (Wang e Abraham, 2015) |
| Controllo statistico di processo | DDM, EDDM, STEPD, EWMA |
| Monitoraggio a finestre | ADWIN, DoD, Resampling |

[GA14] fornisce l'inquadramento storico e la relazione fra i due test. Citando
la traduzione presente in `Papers/`:

> Il test CUSUM (somma cumulativa) è una tecnica di analisi sequenziale dovuta a
> [Page 1954] che utilizza i principi dello SPRT. Viene spesso utilizzato per il
> rilevamento dei cambiamenti. Il test produce un allarme quando la media dei
> dati in arrivo si discosta significativamente da zero. **Il test Page-Hinkley
> [Page 1954] (PH) è una variante del CUSUM.** È una tecnica di analisi
> sequenziale tipicamente utilizzata per il rilevamento dei cambiamenti
> nell'elaborazione del segnale.

Due elementi da trattenere: **SPRT** (Sequential Probability Ratio Test, Wald)
come radice teorica, e la provenienza dall'**elaborazione dei segnali**, non
dal machine learning. Page-Hinkley è nato quarant'anni prima che il concept
drift fosse un problema, ed è stato importato nel campo in un secondo momento.

[SK17] conferma l'origine:

> PHT [...] era stato originariamente sviluppato nel dominio dell'elaborazione
> dei segnali per rilevare la **deviazione dalla media di un segnale
> gaussiano**.
> *(parafrasato da [SK17], §2.1.1)*

L'ipotesi di gaussianità del segnale monitorato è dichiarata nel paper e va
riportata: è un'assunzione che l'errore assoluto di un regressore, essendo
positivo e tipicamente asimmetrico, **non soddisfa esattamente**.

---

## 2. La formulazione

[SK17], §2.1.1, riporta CUSUM e PHT come due equazioni distinte e consecutive.
Riportarle entrambe è utile, perché la differenza fra le due è precisamente il
punto che distingue i due test.

### CUSUM — equazione (2) di [SK17]

```
    M_t = max( 0,  M_{t-1} + eps_t - v )

    se M_t > theta   ->   allarme,  e  M_t = 0
```

[SK17] annota che la funzione `max` serve a testare i cambiamenti in **direzione
positiva**, e che per l'effetto contrario — ad esempio per misurare un calo di
accuratezza — si può usare una funzione `min`. Nota inoltre che il test è
**memory-less** e utilizzabile in modo incrementale.

### Page-Hinkley — equazione (3) di [SK17]

```
    M_0    = 0
    M_t    = M_{t-1} + ( eps_t - v )
    M_Ref  = min( V )

    se M_t - M_Ref > theta   ->   allarme,  e  M_t = 0
```

Con, testualmente da [SK17]:

- `M_0` — la metrica iniziale al tempo `t = 0`
- `M_t` — la metrica corrente, accumulazione della metrica fino a quel punto
  (`M_{t-1}`) e della prestazione del campione al tempo `t`, cioè `eps_t`
- `v` — **la deviazione dalla media considerata accettabile**
- `theta` — **la soglia di rilevamento del cambiamento**

### La differenza fra i due, e perché conta

Il CUSUM tronca a zero (`max(0, ...)`): la somma non può scendere sotto lo zero,
e il confronto è con una soglia assoluta. Page-Hinkley **non tronca**: lascia
che la somma vada dove vuole e confronta la sua distanza dal **minimo storico**
`M_Ref`.

La conseguenza è che Page-Hinkley non ha bisogno di conoscere in anticipo il
livello di riferimento: se lo costruisce dai dati, come minimo osservato. È
questa proprietà che lo rende utilizzabile in streaming, dove la media di
riferimento non è nota a priori.

Entrambi, conclude [SK17], sono adatti al rilevamento **univariato** di
cambiamenti in una sequenza di misure di prestazione.

### Corrispondenza con i parametri dell'implementazione

| Notazione di [SK17] | Parametro in `river.drift.PageHinkley` | Significato |
|---|---|---|
| `v` | `delta` | Deviazione accettabile, cioè l'entità di cambiamento che si accetta come rumore |
| `theta` | `threshold` | Soglia di allarme |
| `eps_t` | il valore passato a `update()` | La misura monitorata al passo `t` |
| — | `alpha` | Fattore di dimenticanza. **Non compare in [SK17] né in [GA14]**: è un'aggiunta pratica dell'implementazione, non del test originale |
| — | `min_instances` | Campioni minimi prima di attivare il test. Anch'esso assente dalla formulazione originale |
| — | `mode` | Selettore fra rilevamento in salita, in discesa o bidirezionale. Corrisponde alla scelta fra `max` e `min` annotata da [SK17] per il CUSUM |

Questa tabella è importante per la tesi: **due dei cinque parametri esposti dal
wrapper non appartengono al test del 1954** ma alla sua realizzazione software.
Dichiararlo evita di attribuire al paper originale scelte che non gli
appartengono.

---

## 3. Il compromesso governato da `theta`

La letteratura è concorde nell'identificare in `theta` il parametro che governa
il compromesso fondamentale del rilevamento:

| `theta` | Rilevamento | Falsi allarmi |
|---|---|---|
| basso | rapido | numerosi |
| alto | lento | rari |

[SK17], nella parte sperimentale, riporta inoltre una caratterizzazione
comparativa del PHT: mostra un **ritardo medio di rilevamento minimo**, con
tempi medi inferiori a quelli degli altri metodi confrontati, ma **soffre di un
tasso di falsi allarmi più elevato**.

È esattamente il comportamento osservato negli esperimenti di questo lavoro sui
dataset Friedman: Page-Hinkley è l'unica strategia a raggiungere la copertura
completa sulle varianti `lea` e `gsg`, al prezzo di 36–38 falsi allarmi contro
gli 0–2 di ADWIN. La convergenza fra il dato di letteratura e la misura
sperimentale è un elemento da valorizzare nel capitolo di validazione.

---

## 4. Limiti

### 4.1 Dichiarati dalle fonti

- **Univariato.** [SK17]: CUSUM e PHT sono adatti al rilevamento univariato su
  una sequenza di misure di prestazione. Non gestiscono nativamente il
  monitoraggio congiunto di più flussi.
- **Assunzione di gaussianità.** Il test nasce per rilevare la deviazione dalla
  media di un **segnale gaussiano** ([SK17]). L'errore assoluto di un regressore
  è positivo e in generale asimmetrico: l'assunzione non è soddisfatta
  esattamente, e il test va usato come strumento robusto piuttosto che come test
  esatto.
- **Falsi allarmi.** Vedere §3.

### 4.2 Che discendono dalla formulazione

Conseguenze verificabili delle equazioni del §2, da presentare come tali:

- **Nessuna garanzia formale su falsi positivi e falsi negativi.** A differenza
  di ADWIN, che in [BG07] dispone di limiti dimostrati, per Page-Hinkley il
  comportamento va caratterizzato empiricamente tarando `v` e `theta`.
- **Sensibile alla scala.** `v` e `theta` sono espressi nelle unità del flusso
  monitorato. Gli stessi valori non funzionano su un errore di ordine 0,01 e su
  uno di ordine 1000: su dati non normalizzati vanno ritarati.
- **Nessuna interpretazione probabilistica dei parametri.** `theta = 50` non
  significa "5% di falsi positivi", a differenza dell'`alpha` di un test di
  ipotesi. È una delle ragioni per cui il framework, nell'uscita del servizio di
  monitoring, non associa alcun punteggio a Page-Hinkley.
- **Cieco ai cambiamenti di forma a media costante.** Come ADWIN, e a differenza
  del test di Kolmogorov-Smirnov, osserva la media.
- **Reinizializzazione dopo l'allarme.** Entrambe le equazioni di [SK17]
  pongono `M_t = 0` dopo la segnalazione: il test riparte, e ogni rilevamento è
  indipendente dai precedenti.

---

## 5. Perché è nel framework

L'argomento è di copertura, e si regge su due gambe.

**Gamba tecnica.** Fra le strategie implementate, è l'unica applicabile a un
flusso **continuo e non limitato**, che è il caso dell'errore assoluto di un
modello di regressione. DDM non è applicabile perché presuppone un errore
bernoulliano; ADWIN è applicabile ma le garanzie di [BG07] valgono per variabili
limitate in [0,1], quindi lo si userebbe fuori dalle sue ipotesi.

**Gamba tassonomica.** Nella classificazione di [SK17] le famiglie di metodi
espliciti sono tre. Con il solo KS e ADWIN il framework ne copriva due; con DDM
e Page-Hinkley le copre tutte e tre, e la tassonomia esposta nel capitolo di
stato dell'arte smette di essere puramente descrittiva.

---

## 6. Domande di verifica

**Qual è la differenza fra CUSUM e Page-Hinkley?**
Il CUSUM tronca la somma cumulata a zero e la confronta con una soglia assoluta;
Page-Hinkley non tronca e confronta la distanza dal minimo storico `M_Ref`. Ne
segue che il PH non richiede di conoscere a priori il livello di riferimento, il
che lo rende usabile in streaming. Equazioni (2) e (3) di [SK17].

**A quale famiglia appartiene?**
Analisi sequenziale, insieme al CUSUM e a LFR ([SK17]). La radice teorica è lo
SPRT di Wald ([GA14]).

**Da dove viene?**
Dall'elaborazione dei segnali, per rilevare la deviazione dalla media di un
segnale gaussiano ([SK17]). È stato importato nel concept drift in un secondo
momento.

**Il test è del 1954: non è superato?**
No. Costo di memoria costante, costo computazionale O(1) per campione,
`memory-less` e incrementale ([SK17] lo annota per il CUSUM). In un servizio che
monitora molti flussi in parallelo queste proprietà contano più della
raffinatezza statistica.

**Perché nell'uscita del servizio non c'è un punteggio per Page-Hinkley?**
Perché la statistica interna non ha interpretazione probabilistica: non è né un
p-value né una probabilità di drift. Riportare un numero senza saper dire cosa
significa sarebbe peggio che non riportarlo.

**Come sono stati scelti `v` e `theta`?**
Partendo dai valori predefiniti dell'implementazione e caratterizzandone il
comportamento sperimentalmente. Non esistono valori canonici nel paper
originale: la letteratura indica il compromesso, non i numeri.

---

## 7. Riferimenti

**Paper primari, da recuperare via accesso istituzionale**
- Page, E. S. (1954). *Continuous Inspection Schemes*. Biometrika, 41(1/2),
  100–115. DOI 10.2307/2333009
- Hinkley, D. V. (1971). *Inference about the change-point from cumulative sum
  tests*. Biometrika, 58(3), 509–523.

**Fonti secondarie effettivamente consultate per questo documento**
- Sethi, T. S., Kantardzic, M. (2017). *On the reliable detection of concept
  drift from streaming unlabeled data*. Expert Systems with Applications, 82,
  77–99. arXiv:1704.00023 — equazioni (2) e (3), tassonomia, caratterizzazione
  sperimentale
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A. (2014).
  *A Survey on Concept Drift Adaptation*. ACM Computing Surveys, 46(4), 1–37 —
  inquadramento storico, relazione con CUSUM e SPRT
- Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G. (2019). *Learning under
  Concept Drift: A Review*. IEEE TKDE, 31(12) — inquadramento generale

**Citati dalle fonti sopra, non consultati direttamente**
- Wald, A. (1945). *Sequential Tests of Statistical Hypotheses* — origine dello
  SPRT, base teorica del CUSUM secondo [GA14].
- Wang, H., Abraham, Z. (2015). *Concept drift detection for streaming data*
  (LFR) — citato da [SK17] nella stessa famiglia.
- Bifet, A., Gavaldà, R. (2007). *Learning from Time-Changing Data with Adaptive
  Windowing* — in `Papers/`, usato qui per il confronto sulle garanzie formali.

**Implementazione**
- Montiel, J., Halford, M., Mastelini, S. M., et al. (2021). *River: machine
  learning for streaming data in Python*. JMLR, 22(110), 1–8. — classe
  `river.drift.PageHinkley`, versione 0.22.0
