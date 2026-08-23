# ELEC2 — tre correzioni ai risultati precedenti

**Data**: 2026-08-23
**Sostituisce**: le sezioni su ELEC2 di `results/analisi_risultati.md` e le
sottosezioni corrispondenti del Capitolo 6 della bozza di tesi.
**Riproducibile con**: `python -m evaluation.runner_elec2`

Questa nota documenta tre errori nei risultati riportati finora su ELEC2 e le
conclusioni corrette che li sostituiscono. Sono scritte esplicitamente, invece
di correggere i numeri in silenzio, perche' il percorso che ha portato alla
correzione e' esso stesso materiale di tesi: e' la dimostrazione del perche'
serviva un protocollo di valutazione.

---

## Correzione 1 — Il rapporto "80 a 1" non esisteva

**Quanto affermato prima**: *"Il KS produce 44.413 segnalazioni contro le 558
di ADWIN, circa 80 volte piu' rumoroso."*

**Il problema**: i due numeri misurano cose diverse. ADWIN e' edge-triggered e
il suo conteggio corrisponde a eventi distinti. Il KS e' level-triggered:
rifa' il test a ogni passo su finestre sovrapposte al 99% e resta in stato di
allarme finche' la condizione persiste. Le sue 44.413 non erano 44.413 drift,
ma **passi trascorsi in stato di allarme**.

**Il dato corretto**, contando episodi contigui:

| Tipo di drift | Strategia | Passi in allarme | % del tempo | Episodi | Durata media |
|---|---|---|---|---|---|
| feature | KS | 44.612 | 99,5% | **1** (permanente) | 44.612 |
| feature | ADWIN | 556 | 1,2% | **556** | 1 |
| prediction | KS | 44.403 | 99,1% | **4** | 11.101 |
| prediction | ADWIN | 2 | 0,0% | **2** | 1 |
| concept | ADWIN | 66 | 0,1% | **66** | 1 |
| concept | DDM | 12 | 0,0% | **12** | 1 |
| concept | PageHinkley | 24 | 0,1% | **24** | 1 |
| concept | Ensemble-MAJORITY | 1 | 0,0% | **1** | 1 |

**La conclusione corretta**: sul feature drift il rapporto non e' 80 a 1 a
favore di ADWIN — **e' rovesciato**. Il KS produce **un solo episodio**, ADWIN
ne produce **556**. Per numero di eventi distinti e' ADWIN il piu' prolisso.

Ma nessuno dei due numeri, da solo, descrive il comportamento. Servono
entrambe le misure:

- **quota di tempo in allarme**: KS 99,5%, ADWIN 1,2%
- **numero di episodi distinti**: KS 1, ADWIN 556

Il KS con `k=1` su 8 feature entra in allarme al passo 200 e **non ne esce
piu' fino alla fine dello stream**. Non e' "eccessivamente sensibile": e'
**saturo**. Un allarme sempre acceso non trasporta informazione, ed e'
inutilizzabile come segnale operativo — una conclusione piu' netta di quella
precedente, e per una ragione diversa.

ADWIN, all'opposto, produce 556 eventi istantanei ben localizzati: molti, ma
ciascuno con un istante preciso a cui e' associabile un evento del sistema di
monitoring.

---

## Correzione 2 — Il KS sul prediction drift non e' inutilizzabile

**Quanto affermato prima**: *"44.180 segnalazioni su 44.812 passi (98,6%).
Nessuna informazione utile ricavabile: rumore continuo. Il risultato e'
sostanzialmente inutilizzabile."*

**Il dato corretto**: sono **4 episodi**, di durata media 11.101 passi.

**La conclusione corretta**: quattro episodi distinti su 44.812 campioni sono
un'informazione perfettamente utilizzabile. Il KS sullo stream binario delle
predizioni identifica **quattro regimi successivi** nel comportamento del
classificatore. Resta vero che scipy segnala il ripiego all'approssimazione
asintotica sui dati discreti, e quindi che il test e' fuori dal suo dominio
d'elezione; ma l'affermazione che il risultato fosse "rumore continuo" era
**una conseguenza del modo di contare, non del comportamento del detector**.

---

## Correzione 3 — Il "caso vicprice" e' un artefatto del dataset

Questa e' la correzione piu' importante, perche' riguarda il risultato che la
bozza indicava come il piu' significativo dell'intero capitolo.

**Quanto affermato prima**: *"Su `vicprice` KS rileva drift nel 62% dei passi,
ADWIN mai. Significa che la forma della distribuzione di `vicprice` cambia nel
tempo, ma la sua media resta stabile. Il KS vede il cambio di forma via ECDF;
ADWIN, che monitora solo la media, correttamente non vede nulla. E' la
dimostrazione empirica piu' chiara della complementarita' fra i due algoritmi."*

**Cosa c'e' davvero nei dati**:

| Feature | Costante fino al passo | Quota del dataset | Valore costante | Media dopo | Dev. std dopo |
|---|---|---|---|---|---|
| `vicprice` | 17.424 | 38,5% | 0,003467 | 0,003467 | 0,013018 |
| `vicdemand` | 17.424 | 38,5% | 0,422915 | 0,422915 | 0,154189 |
| `transfer` | 17.424 | 38,5% | 0,414912 | 0,554017 | 0,175439 |

Le tre feature del mercato di Victoria hanno **un solo valore distinto** per i
primi 17.424 campioni, cioe' per il 38,5% del dataset. E' un difetto noto di
ELEC2: i dati di Victoria non venivano raccolti nel periodo iniziale, e i
valori mancanti sono stati riempiti con una costante — pari, come si vede
dalla tabella, alla media del periodo successivo.

**Il meccanismo descritto nella bozza e' corretto, la sua interpretazione no.**

E' vero che il KS vede un cambiamento di forma e ADWIN no, ed e' vero che
entrambi si comportano correttamente. Su `vicprice` la media dopo la
transizione e' `0,003467`, **identica al valore costante fino alla sesta cifra
decimale**: ADWIN, che monitora la media, non ha letteralmente nulla da
rilevare. Il KS invece confronta una delta di Dirac con una distribuzione
reale, e la distanza fra le due ECDF e' massima.

Ma non e' vero che "la forma della distribuzione di `vicprice` cambia nel
tempo" come proprieta' del mercato elettrico. Cambia perche' **si passa da
valori imputati a valori misurati**. Presentare un artefatto di imputazione
dei mancanti come una sottile proprieta' statistica di una variabile di
mercato e' un errore di interpretazione, ed e' esattamente il tipo di
affermazione che verrebbe contestata in sede di discussione.

**La conclusione corretta, che resta un buon risultato**: i detector hanno
individuato un **problema di qualita' del dato**, non un drift del fenomeno.
E' un uso legittimo e interessante del drift detection — il monitoraggio della
distribuzione come strumento di data quality — ma va raccontato per quello che
e'. Come ricaduta pratica ne discende una regola operativa: **prima di
interpretare qualunque segnalazione di drift, occorre verificare che la
finestra di riferimento non sia costruita su dati imputati o degenerati**.

Il controllo e' ora automatico ed e' eseguito da
`controlla_qualita_dataset()` all'avvio di ogni analisi su ELEC2.

Nota collaterale: il classificatore e' addestrato sui primi 500 campioni, dove
**3 feature su 8 sono costanti**. Il `GaussianNB` lavora quindi di fatto su 5
feature informative. Anche questo va dichiarato.

---

## Cosa resta valido dei risultati precedenti

- Il **degrado del classificatore statico** e' confermato e robusto: accuracy
  da 0,850 iniziale a un minimo di 0,429, media 0,583 sull'intero stream.
- La feature `period` resta **stazionaria per entrambi i detector** (0
  segnalazioni). E' il controllo di consistenza che dimostra l'assenza di bug
  sistematici nel framework.
- La feature `date`, timestamp monotono, e' un **falso positivo strutturale**
  confermato: 99,55% del tempo in allarme per il KS.
- L'architettura regge: cinque strategie diverse hanno attraversato gli stessi
  tre detector senza una riga di modifica al codice di orchestrazione.

## Risultati nuovi

- **DDM e' il piu' parsimonioso sul concept drift**: 12 episodi contro i 66 di
  ADWIN e i 24 di Page-Hinkley, sullo stesso stream di errori. Ma i suoi
  episodi si concentrano nella prima parte dello stream e poi si diradano:
  dopo che il modello si e' stabilmente degradato, la baseline `p_min` di DDM
  si adatta al peggioramento e smette di segnalare. E' un limite reale
  dell'algoritmo su un modello statico, da dichiarare.
- **4.599 passi in stato di warning DDM**: e' la base su cui costruire la
  politica di retraining dell'Objective 4.
- **Ensemble MAJORITY produce 1 solo episodio** su tutto lo stream. Coerente
  con quanto misurato sui dati sintetici: la regola a maggioranza abbatte i
  falsi allarmi ma sacrifica pesantemente la copertura.
