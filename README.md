# Portfolio Optimizer – Tre linee di rischio

Web app Streamlit per costruire tre allocazioni long-only:

- Conservative: minimizzazione del CVaR;
- Balanced: massimizzazione dello Sharpe con penalità di concentrazione;
- Aggressive: selezione sulla frontiera rendimento–CVaR;
- minimo di Conservative e massimi di volatilità modificabili;
- volatilità annua prevista crescente tra le tre linee, con distanza minima
  regolabile (0,5 punti percentuali per impostazione iniziale);
- limiti massimi personalizzati per titolo;
- rendimenti attesi robusti con winsorization e shrinkage;
- bootstrap a blocchi e intervalli d'incertezza;
- walk-forward OOS causale con turnover e costi, suddiviso anche in fino a tre
  finestre cronologiche consecutive;
- benchmark diagnostici GMV Ledoit-Wolf, HRP vincolato ed Equal Weight vincolato;
- Sharpe previsto e Sharpe OOS netto affiancati nella tabella principale;
- misure di drawdown, Sortino, Sharpe, CVaR e concentrazione;
- diagnostica della separazione tra linee su volatilità, CVaR, drawdown,
  distanza dei pesi e bootstrap, con avviso se Aggressive ha rendimento atteso
  non superiore a Balanced;
- scenario facoltativo di correlazione di stress, attivabile con un interruttore
  inizialmente spento, che influenza i pesi delle tre linee con intensità
  decrescente (65%, 35%, 10%);
- pulsante Copia accanto a ogni tabella per incollare le celle in Excel.

I benchmark rispettano i limiti per singolo titolo ma non vengono forzati nei
corridoi di volatilità delle tre linee. Servono per verificare fuori campione se
la maggiore complessità del modello produce risultati migliori rispetto ad
allocazioni indipendenti dai rendimenti attesi.

Le linee sono ottimizzate in sequenza. Il minimo di Balanced è la volatilità
prevista calcolata di Conservative più la distanza impostata; il minimo di
Aggressive deriva allo stesso modo da Balanced. La regola si applica anche a
ogni ribilanciamento del walk-forward. Se i massimi o i limiti sui pesi la
rendono impossibile, l'app segnala l'errore senza modificare i vincoli.
Le volatilità realizzate fuori campione possono avere un ordine diverso.
La validazione inizia dopo almeno metà delle osservazioni e presenta fino a
tre finestre non sovrapposte con risultati netti dei costi. Il bootstrap
della separazione usa pesi fissi e non misura l'incertezza della riottimizzazione.

Quando lo stress dinamico è attivo, la correlazione dei periodi peggiori viene
stimata con shrinkage usando solo i dati disponibili a ogni ribilanciamento.
Entra nella scelta dei pesi, mentre i corridoi e l'ordine delle volatilità
continuano a usare la stessa covarianza ordinaria per tutte le linee. Se il
campione non basta per stimare lo scenario, l'app lo segnala invece di
applicarlo parzialmente. Il confronto fuori campione con la versione senza
stress si avvia separatamente dalla scheda Simulation e non viene usato per
selezionare retroattivamente i pesi. Lo stress può migliorare o peggiorare
Sharpe, CVaR e rendimento sui dati caricati.

## Installazione

È consigliato utilizzare un ambiente virtuale Python.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Avvio

```powershell
streamlit run combinazione_ott.py
```

L'app viene normalmente aperta su `http://localhost:8501`.

## Formato dei dati

Il file deve essere un CSV con:

- date nella prima colonna;
- un titolo per ogni colonna successiva;
- prezzi storici positivi;
- frequenza giornaliera, settimanale o mensile;
- valori mancanti indicati anche come `undefined`, se presenti.

La frequenza selezionata nell'app serve per annualizzare le misure e non
ricampiona le osservazioni caricate.

## Test

```powershell
python -m unittest -v test_portfolio_engine.py
```

## File principali

- `combinazione_ott.py`: interfaccia Streamlit;
- `portfolio_engine.py`: motore quantitativo;
- `test_portfolio_engine.py`: test automatici;
- `.streamlit/config.toml`: tema light;
- `requirements.txt`: dipendenze bloccate.

## Avvertenza

I risultati sono stime quantitative basate esclusivamente sui dati storici
caricati. Non costituiscono una previsione certa né una raccomandazione
personalizzata di investimento.
