# Portfolio Optimizer – Tre linee di rischio

Web app Streamlit per costruire tre allocazioni long-only:

- Conservative: minimizzazione del CVaR;
- Balanced: massimizzazione dello Sharpe con penalità di concentrazione;
- Aggressive: selezione sulla frontiera rendimento–CVaR;
- corridoi di volatilità modificabili;
- limiti massimi personalizzati per titolo;
- rendimenti attesi robusti con winsorization e shrinkage;
- bootstrap a blocchi e intervalli d'incertezza;
- walk-forward OOS causale con turnover e costi;
- benchmark diagnostici GMV Ledoit-Wolf, HRP vincolato ed Equal Weight vincolato;
- Sharpe previsto e Sharpe OOS netto affiancati nella tabella principale;
- misure di drawdown, Sortino, Sharpe, CVaR e concentrazione.

I benchmark rispettano i limiti per singolo titolo ma non vengono forzati nei
corridoi di volatilità delle tre linee. Servono per verificare fuori campione se
la maggiore complessità del modello produce risultati migliori rispetto ad
allocazioni indipendenti dai rendimenti attesi.

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
