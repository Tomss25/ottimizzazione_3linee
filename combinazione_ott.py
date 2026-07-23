import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Gestione sicura di Matplotlib per lo styling
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ---------------------------------------------------------
st.set_page_config(
    layout="wide", 
    page_title="Portfolio Optimizer", 
    page_icon="⚙️"
)

# ---------------------------------------------------------
# CSS THEME: AI / CYBERPUNK
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    .stApp { 
        background-color: #F8FAFC; 
        color: #0F172A;
        font-family: 'Fira Sans', sans-serif; 
    }
    
    label, li { color: #0F172A !important; }

    [data-testid="stSidebar"] { 
        background-color: #FFFFFF; 
        border-right: 1px solid #DBEAFE; 
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #334155 !important;
    }
    
    [data-testid="stExpander"] summary p {
        color: #1E3A8A !important;
        font-weight: 700 !important;
    }
    
    h1, h2, h3 { 
        color: #0F172A !important; 
        font-weight: 700; 
        letter-spacing: 0.02em; 
    }
    h1 { font-family: 'Fira Code', monospace; font-size: 2.35rem; }
    
    div[data-testid="metric-container"] { 
        background-color: #FFFFFF; 
        border: 1px solid #DBEAFE; 
        border-left: 4px solid #1E40AF; 
        border-radius: 8px; 
        padding: 16px; 
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06); 
    }
    [data-testid="stMetricLabel"] { font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #475569 !important; }
    [data-testid="stMetricValue"] { font-family: 'Fira Code', monospace; font-size: 1.8rem; color: #1E40AF !important; }

    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button { 
        background: #1E40AF; 
        color: #FFFFFF !important; 
        border: 1px solid #1E40AF; 
        border-radius: 7px; 
        font-family: 'Fira Code', monospace;
        font-weight: 600; 
        text-transform: uppercase;
        letter-spacing: 0.04em;
        min-height: 44px;
        transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { 
        background: #1D4ED8;
        border-color: #1D4ED8; 
        box-shadow: 0 3px 10px rgba(30, 64, 175, 0.2);
        color: #FFFFFF !important;
    }
    .stButton > button p, .stDownloadButton > button p, [data-testid="stFormSubmitButton"] > button p { color: #FFFFFF !important; }

    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible { outline: 3px solid rgba(30, 64, 175, 0.25); outline-offset: 2px; }

    [data-testid="stFileUploader"] { background-color: #FFFFFF; border: 1px solid #DBEAFE; border-radius: 8px; padding: 10px; }
    [data-testid="stFileUploader"] section { background-color: #F8FAFC !important; border: 1px dashed #93C5FD !important; }
    [data-testid="stFileUploader"] section > div > div > span, [data-testid="stFileUploader"] section > div > div > small { color: #475569 !important; }
    [data-testid="stFileUploader"] button { background: #FFFFFF; color: #1E40AF !important; border: 1px solid #93C5FD; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #FFFFFF; border-radius: 7px 7px 0 0; color: #64748B !important; font-family: 'Fira Code', monospace; border: 1px solid #E2E8F0; }
    .stTabs [aria-selected="true"] { background-color: #EFF6FF; color: #1E40AF !important; border: 1px solid #BFDBFE; border-bottom: 3px solid #1E40AF; }
    
    [data-testid="stDataFrame"] { border: 1px solid #DBEAFE; border-radius: 8px; overflow: hidden; }
    
    /* MODIFICA: Intestazione tabellare evidenziata (Testo più grande, maiuscolo e bordo inferiore marcato) */
    thead tr th { background-color: #EFF6FF !important; color: #1E3A8A !important; font-family: 'Fira Code', monospace; font-size: 0.95rem !important; text-transform: uppercase; border-bottom: 2px solid #93C5FD !important; }
    
    /* MODIFICA: Rimosso '!important' da background-color per permettere a Pandas di applicare i colori tenui */
    tbody tr td { color: #0F172A; font-family: 'Fira Sans', sans-serif; background-color: #FFFFFF; }
    
    .stSelectbox > div > div, .stMultiSelect > div > div { background-color: #FFFFFF; color: #0F172A !important; border-color: #CBD5E1; }
    [data-baseweb="tag"] { background-color: #DBEAFE !important; color: #1E3A8A !important; }
    [data-baseweb="tag"] span { color: #1E3A8A !important; }
    [data-baseweb="tag"] svg { fill: #1E40AF !important; }
    div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #FFFFFF !important; border: 1px solid #CBD5E1; }
    div[data-baseweb="popover"] li, div[data-baseweb="menu"] li { background-color: #FFFFFF !important; color: #0F172A !important; }
    div[data-baseweb="popover"] li:hover, div[data-baseweb="menu"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"], div[data-baseweb="menu"] li[aria-selected="true"] { background-color: #EFF6FF !important; color: #1E40AF !important; }
    [data-baseweb="select"] svg { fill: #1E40AF !important; }
    [data-testid="stExpander"], [data-testid="stForm"] { background: #FFFFFF; border-color: #DBEAFE; border-radius: 8px; }
    hr { border-color: #E2E8F0 !important; }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Motore quantitativo verificabile separatamente dall'interfaccia Streamlit.
from portfolio_engine import (
    FREQUENCY_LABELS,
    bootstrap_portfolio_intervals,
    calculate_concentration_metrics,
    calculate_crisis_correlation,
    calculate_historical_risk_metrics,
    calculate_metrics,
    compute_causal_returns,
    compute_core_stats,
    detect_frequency,
    empirical_cvar,
    find_high_correlation_pairs,
    normalize_price_frequency,
    optimize_basket,
    process_data_raw,
    run_walk_forward,
    validate_weight_bounds,
)


def style_chart(fig, title):
    chart_colors = ['#1E40AF', '#D97706', '#0F766E', '#7C3AED', '#DC2626']
    fig.update_layout(
        template="plotly_white", 
        title=dict(text=f"<b>{title}</b>", font=dict(size=18, family="Fira Code, monospace", color="#1E3A8A"), x=0, y=0.96),
        colorway=chart_colors, 
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='#FFFFFF',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Fira Code, monospace", size=11, color="#334155")),
        font=dict(family="Fira Sans, sans-serif", color="#334155"),
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0', gridwidth=1, zerolinecolor='#CBD5E1'),
        yaxis=dict(showgrid=True, gridcolor='#E2E8F0', gridwidth=1, zerolinecolor='#CBD5E1')
    )
    return fig

# ---------------------------------------------------------
# INTERFACCIA
# ---------------------------------------------------------
st.title("📈 Portfolio Optimizer")
st.markdown("<p style='color: #475569; margin-top: -15px; margin-bottom: 30px; font-family: Fira Code'>Constrained Asset Allocation & Risk Analysis System</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ SYSTEM PARAMETERS")
frequency_choice = st.sidebar.selectbox(
    "Frequenza delle serie",
    [
        "Rilevamento automatico",
        "Giornaliera (252)",
        "Settimanale (52)",
        "Mensile (12)",
    ],
    help=(
        "Serve esclusivamente ad annualizzare rendimenti e rischio. "
        "Le osservazioni caricate non vengono ricampionate o aggregate."
    ),
)
mean_shrinkage = st.sidebar.slider(
    "Prudenza sui rendimenti attesi",
    0.0,
    1.0,
    0.70,
    0.05,
    help=(
        "Riduce l'extra-rendimento storico verso zero. Valori alti producono "
        "stime più prudenti. Il metodo usa medie winsorizzate e non viene "
        "presentato come Black–Litterman."
    ),
)
risk_free_rate_pct = st.sidebar.slider(
    "Tasso privo di rischio annuo",
    0.0,
    10.0,
    2.0,
    0.25,
    format="%.2f%%",
)
risk_free_rate = risk_free_rate_pct / 100
transaction_cost_bps = st.sidebar.number_input(
    "Costo per ribilanciamento (bps)",
    min_value=0,
    max_value=200,
    value=10,
    step=5,
    help="Applicato al turnover nella validazione walk-forward.",
)
bootstrap_simulations = st.sidebar.number_input(
    "Simulazioni bootstrap",
    min_value=100,
    max_value=2000,
    value=500,
    step=100,
    help=(
        "Ricampionamento a blocchi delle sole serie caricate per costruire "
        "intervalli d'incertezza."
    ),
)
st.sidebar.markdown("---")
min_w_pct = st.sidebar.slider("Min % per Asset", 0, 20, 0, 1)
max_w_pct = st.sidebar.slider("Max % per Asset", 10, 100, 35, 5)
min_w = min_w_pct / 100
max_w = max_w_pct / 100
st.sidebar.markdown("---")

with st.sidebar.expander("📉 Parametri Volatilità", expanded=True):
    st.caption(
        "I valori mostrati sono i vincoli effettivamente utilizzati. "
        "Qualsiasi modifica sostituisce immediatamente il preset iniziale."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Minima**")
        min_cons = st.number_input("Conservativa min (%)", 0.0, 100.0, 0.0, 1.0)
        min_bal = st.number_input("Balanced min (%)", 0.0, 100.0, 15.0, 1.0)
        min_agg = st.number_input("Aggressive min (%)", 0.0, 100.0, 25.0, 1.0)
    with c2:
        st.markdown("**Massima**")
        max_cons = st.number_input("Conservativa max (%)", 0.0, 100.0, 15.0, 1.0)
        max_bal = st.number_input("Balanced max (%)", 0.0, 100.0, 25.0, 1.0)
        max_agg = st.number_input("Aggressive max (%)", 0.0, 100.0, 40.0, 1.0)

    max_limits = {
        "Conservative": max_cons / 100,
        "Balanced": max_bal / 100,
        "Aggressive": max_agg / 100,
    }
    min_limits = {
        "Conservative": min_cons / 100,
        "Balanced": min_bal / 100,
        "Aggressive": min_agg / 100,
    }

# Main
col_up, col_info = st.columns([1, 2])
with col_up:
    uploaded = st.file_uploader("📂 INGEST DATA (CSV)", type='csv')

if uploaded:
    df_raw, error_msg = process_data_raw(uploaded.getvalue(), uploaded.name)
    
    if df_raw is None:
        st.error(error_msg); st.stop()
        
    interpolated_counts = df_raw.attrs.get("interpolated_counts", {})
    detected_frequency = detect_frequency(df_raw)
    manual_frequency_map = {
        "Giornaliera (252)": 252,
        "Settimanale (52)": 52,
        "Mensile (12)": 12,
    }
    freq = (
        detected_frequency
        if frequency_choice == "Rilevamento automatico"
        else manual_frequency_map[frequency_choice]
    )
    df_raw = normalize_price_frequency(df_raw, freq)
    available_assets = [
        asset for asset in df_raw.columns
        if df_raw[asset].notna().sum() >= 3
    ]
    with col_info:
        st.success("✅ SYSTEM READY: Data Ingested")
        selected_assets = st.multiselect("Active Assets:", options=available_assets, default=available_assets)
        st.caption(
            f"Frequenza rilevata: {FREQUENCY_LABELS[detected_frequency]} · "
            f"Frequenza usata per annualizzare: {FREQUENCY_LABELS[freq]} · "
            "nessun ricampionamento applicato"
        )
        if interpolated_counts:
            total_filled = sum(interpolated_counts.values())
            st.caption(
                f"Ricostruiti {total_filled} piccoli buchi interni con "
                "interpolazione log-lineare; estremi mancanti non modificati."
            )
    
    if len(selected_assets) < 2: st.warning("⚠️ CRITICAL: Select at least 2 assets."); st.stop()

    with st.sidebar.expander("Limiti massimi per singolo titolo", expanded=False):
        capped_assets = st.multiselect(
            "Titoli con limite personalizzato",
            options=selected_assets,
            help="Il limite personalizzato sostituisce il limite massimo generale per il titolo selezionato.",
        )
        custom_max_weights = {}
        for asset in capped_assets:
            custom_max_pct = st.number_input(
                f"Peso massimo (%) · {asset}",
                min_value=0.0,
                max_value=100.0,
                value=float(max_w * 100),
                step=1.0,
                format="%.1f",
                key=f"custom_max_{asset}",
            )
            custom_max_weights[asset] = custom_max_pct / 100

    min_weights = np.full(len(selected_assets), min_w, dtype=float)
    max_weights = np.array(
        [custom_max_weights.get(asset, max_w) for asset in selected_assets],
        dtype=float,
    )

    try:
        invalid_corridors = [
            name for name in min_limits
            if min_limits[name] > max_limits[name]
        ]
        if invalid_corridors:
            raise ValueError(
                "La volatilità minima supera la massima per: "
                + ", ".join(invalid_corridors)
            )
        validate_weight_bounds(min_weights, max_weights)
        selected_prices = df_raw[selected_assets].copy()
        selected_prices.attrs.update(df_raw.attrs)
        returns, mu_views, cov, clean_prices = compute_core_stats(
            selected_prices,
            freq,
            mean_shrinkage,
            risk_free_rate,
        )
        causal_returns, _ = compute_causal_returns(selected_prices)
    except ValueError as e:
        st.error(f"Errore computazionale: {str(e)}"); st.stop()

    with col_info:
        st.caption(
            f"Rendimenti comuni: {clean_prices.index.min().date()} → "
            f"{clean_prices.index.max().date()} · "
            f"{len(returns)} osservazioni · "
            f"{len(causal_returns)} osservazioni causali per il backtest"
        )
        if len(returns) < 2 * freq:
            st.warning(
                "Il campione comune copre meno di due anni equivalenti. "
                "Le stime di rendimento atteso e i portafogli più aggressivi "
                "possono essere instabili."
            )
        
    allocations = {}
    metrics = []
    historical_risk = {}
    concentration_metrics = {}
    strategies_config = [
        ("Conservative", "min_cvar"),
        ("Balanced", "max_sharpe"),
        ("Aggressive", "cvar_frontier"),
    ]
    
    calc_success = True
    
    for name, method in strategies_config:
        mx_vol = max_limits[name]
        mn_vol = min_limits[name]
        
        try:
            w = optimize_basket(
                mu_views.values,
                cov,
                method,
                min_weights,
                max_weights,
                risk_free_rate,
                max_vol=mx_vol,
                min_vol=mn_vol,
                frequency=freq,
                returns_history=returns,
            )
            m = calculate_metrics(
                w,
                mu_views.values,
                cov,
                freq,
                risk_free_rate,
            )
            allocations[name] = w
            metrics.append([name] + list(m))
            historical_risk[name] = calculate_historical_risk_metrics(
                w,
                returns,
                freq,
                risk_free_rate,
            )
            concentration_metrics[name] = calculate_concentration_metrics(
                w,
                returns.corr(),
            )
        except ValueError as e:
            st.error(f"❌ FALLIMENTO STRATEGIA '{name}': {str(e)}")
            calc_success = False
            break

    if not calc_success:
        st.stop() 
    
    res_df = pd.DataFrame(metrics, columns=["Linea", "Rendimento", "Volatilità", "Sharpe", "Diversif. Ratio"]).set_index("Linea")
    risk_df = pd.DataFrame(historical_risk).T
    concentration_df = pd.DataFrame(concentration_metrics).T
    bootstrap_intervals, _ = bootstrap_portfolio_intervals(
        allocations,
        returns,
        freq,
        risk_free_rate,
        simulations=int(bootstrap_simulations),
    )
    high_corr_pairs = find_high_correlation_pairs(returns)
    active_corridors = pd.DataFrame(
        {
            "Volatilità minima": pd.Series(min_limits),
            "Volatilità massima": pd.Series(max_limits),
            "Volatilità calcolata": res_df["Volatilità"],
        }
    )
    active_corridors["Vincolo rispettato"] = (
        (active_corridors["Volatilità calcolata"] >= active_corridors["Volatilità minima"] - 1e-6)
        & (active_corridors["Volatilità calcolata"] <= active_corridors["Volatilità massima"] + 1e-6)
    )
    
    # --- VISUALIZZAZIONE ---
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 OPTIMIZER", "🥧 ALLOCATION", "📈 SIMULATION", "🔗 CORRELATION"])
    
    with tab1:
        st.markdown("### 🧠 Expected Risk & Return")
        best = res_df.loc["Balanced"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exp. Return", f"{best['Rendimento']:.1%}")
        c2.metric("Exp. Volatility", f"{best['Volatilità']:.1%}")
        c3.metric("Sharpe Ratio", f"{best['Sharpe']:.2f}")
        c4.metric("Div. Score", f"{best['Diversif. Ratio']:.2f}")
        
        st.markdown("#### Strategy Matrix")
        st.caption(
            "I rendimenti attesi sono medie winsorizzate con extra-rendimento "
            "ridotto verso zero. Conservative minimizza il CVaR con la "
            "formulazione scenario-based; Aggressive seleziona il punto di "
            "ginocchio della frontiera rendimento–CVaR. Tutte le linee rispettano "
            "i corridoi di volatilità correnti."
        )
        if HAS_MATPLOTLIB:
            st.table(res_df.style.format("{:.1%}", subset=["Rendimento", "Volatilità"]).format("{:.2f}", subset=["Sharpe", "Diversif. Ratio"]).background_gradient(cmap="viridis", subset=["Rendimento", "Sharpe"]))
        else:
            st.table(res_df.style.format("{:.1%}", subset=["Rendimento", "Volatilità"]).format("{:.2f}", subset=["Sharpe", "Diversif. Ratio"]))

        st.markdown("#### Corridoi di volatilità attivi")
        st.caption(
            "Questi sono i valori correnti inseriti nella sidebar, non i preset "
            "originari. Se un corridoio non è fattibile il calcolo si interrompe: "
            "non viene applicato alcun limite alternativo."
        )
        st.dataframe(
            active_corridors.style.format(
                "{:.2%}",
                subset=[
                    "Volatilità minima",
                    "Volatilità massima",
                    "Volatilità calcolata",
                ],
            ),
            width="stretch",
        )
        
        res_export = res_df.copy()
        for col in ["Rendimento", "Volatilità"]: res_export[col] = res_export[col].map('{:.1%}'.format)
        for col in ["Sharpe", "Diversif. Ratio"]: res_export[col] = res_export[col].map('{:.2f}'.format)
        
        st.download_button(label="📥 DOWNLOAD METRICS (CSV)", data=res_export.to_csv().encode('utf-8'), file_name='ai_metrics.csv', mime='text/csv')

        fig_ef = px.scatter(res_df, x="Volatilità", y="Rendimento", color="Sharpe", size=[50]*len(res_df), text=res_df.index, color_continuous_scale="Viridis", title="Risk / Return Comparison")
        fig_ef.update_traces(textposition='top center', textfont=dict(family="Fira Code, monospace", size=12, color="#334155"))
        fig_ef = style_chart(fig_ef, "Risk / Return Comparison")
        fig_ef.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
        st.plotly_chart(fig_ef, width="stretch")

        st.markdown("#### Downside & Stress Metrics")
        percent_columns = [
            "Downside Deviation",
            "VaR 95% (periodo)",
            "CVaR 95% (periodo)",
            "Drawdown medio",
            "Max Drawdown",
            "Stress storico",
            "Peggior periodo",
        ]
        st.dataframe(
            risk_df.style
            .format("{:.2%}", subset=percent_columns)
            .format("{:.2f}", subset=["Sortino"]),
            width="stretch",
        )

        st.markdown("#### Intervalli d'incertezza bootstrap")
        st.caption(
            f"Intervallo 5%–95% ottenuto con {int(bootstrap_simulations)} "
            "ricampionamenti a blocchi. Non utilizza dati esterni."
        )
        bootstrap_display = bootstrap_intervals.copy()
        for column in ["5%", "Mediana", "95%"]:
            bootstrap_display[column] = bootstrap_display.apply(
                lambda row: (
                    f"{row[column]:.2f}"
                    if row["Metrica"] == "Sharpe"
                    else f"{row[column]:.2%}"
                ),
                axis=1,
            )
        st.dataframe(
            bootstrap_display.set_index(["Linea", "Metrica"]),
            width="stretch",
        )

    with tab2:
        st.markdown("### 🧬 Asset DNA (Composition)")
        weights_df = pd.DataFrame(allocations, index=returns.columns)
        w_clean = weights_df.sort_values(by="Balanced", ascending=False)
        
        # MODIFICA: Funzione per applicare colori tenui e differenziati per ogni singola colonna
        def highlight_cols(s):
            if s.name == 'Conservative':
                return ['background-color: rgba(30, 64, 175, 0.08)'] * len(s)
            elif s.name == 'Balanced':
                return ['background-color: rgba(217, 119, 6, 0.08)'] * len(s)
            elif s.name == 'Aggressive':
                return ['background-color: rgba(220, 38, 38, 0.08)'] * len(s)
            return [''] * len(s)
            
        styled_w = w_clean.style.format("{:.1%}").apply(highlight_cols, axis=0)
        st.dataframe(styled_w, height=500, width="stretch")
        
        w_export = w_clean.map(lambda x: '{:.1%}'.format(x))
        st.download_button(label="📥 DOWNLOAD ALLOCATION (CSV)", data=w_export.to_csv().encode('utf-8'), file_name='ai_allocation.csv', mime='text/csv')

        df_melt = w_clean.reset_index().melt(id_vars="index", var_name="Strategia", value_name="Peso")
        fig_bar = px.bar(df_melt, x="Strategia", y="Peso", color="index", text_auto=".0%", title="Allocation Breakdown")
        fig_bar = style_chart(fig_bar, "Strategic Allocation")
        st.plotly_chart(fig_bar, width="stretch")

        st.markdown("#### Concentrazione e sovrapposizione statistica")
        st.caption(
            "La sovrapposizione è inferita esclusivamente dalle correlazioni "
            "storiche: non rappresenta il look-through delle partecipazioni "
            "interne agli ETF."
        )
        st.dataframe(
            concentration_df.style
            .format(
                "{:.2f}",
                subset=[
                    "HHI",
                    "Numero effettivo titoli",
                    "Correlazione ponderata",
                    "Sovrapposizione statistica",
                ],
            )
            .format("{:.1%}", subset=["Peso primi 3"]),
            width="stretch",
        )

    with tab3:
        st.markdown("### 🕰️ Walk-Forward Validation")
        try:
            (
                nav_df,
                test_start,
                walk_summary,
                gross_nav_df,
                rebalance_diagnostics,
            ) = run_walk_forward(
                causal_returns,
                strategies_config,
                min_weights,
                max_weights,
                risk_free_rate,
                max_limits,
                min_limits,
                freq,
                mean_shrinkage,
                transaction_cost_bps,
            )
        except ValueError as e:
            st.warning(f"Validazione fuori campione non disponibile: {str(e)}")
            st.stop()
        st.caption(
            f"Periodo fuori campione dal {nav_df.index[1].date()} · "
            f"Ribilanciamento periodico causale · Costi: "
            f"{transaction_cost_bps} bps sul turnover one-way · "
            f"Benchmark: Equal Weight"
        )
        
        days = (nav_df.index[-1] - nav_df.index[0]).days
        cagr = (nav_df.iloc[-1] / 100) ** (365.25 / days) - 1
        nav_period_ret = nav_df.pct_change(fill_method=None).dropna()
        vol_real = nav_period_ret.std() * np.sqrt(freq)
        annual_mean_return = nav_period_ret.mean() * freq
        sharpe_real = (
            annual_mean_return - risk_free_rate
        ) / vol_real.replace(0, np.nan)
        dd = (nav_df / nav_df.cummax() - 1).min()
        downside = np.minimum(
            nav_period_ret - risk_free_rate / freq,
            0,
        )
        downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(freq)
        sortino_real = (cagr - risk_free_rate) / downside_dev.replace(0, np.nan)
        var_real = -nav_period_ret.quantile(0.05).clip(upper=0)
        cvar_real = pd.Series(
            {
                column: empirical_cvar(series, 0.95)
                for column, series in nav_period_ret.items()
            }
        )

        perf_hist = pd.DataFrame(
            {
                "CAGR": cagr,
                "Volatilità": vol_real,
                "Sharpe": sharpe_real,
                "Sortino": sortino_real,
                "CVaR 95% (periodo)": cvar_real,
                "Max Drawdown": dd,
            }
        )
        
        c_kpi, c_plot = st.columns([1, 2])
        with c_kpi:
            st.markdown("#### Realized Results")
            if HAS_MATPLOTLIB:
                st.table(
                    perf_hist.style
                    .format(
                        "{:.2%}",
                        subset=[
                            "CAGR",
                            "Volatilità",
                            "CVaR 95% (periodo)",
                            "Max Drawdown",
                        ],
                    )
                    .format("{:.2f}", subset=["Sharpe", "Sortino"])
                    .background_gradient(cmap="RdYlGn", subset=["CAGR"])
                )
            else:
                st.table(perf_hist)
            
        with c_plot:
            fig_nav = px.line(nav_df, title="Capital Growth (Base 100)")
            fig_nav = style_chart(fig_nav, "Equity Line Evolution")
            st.plotly_chart(fig_nav, width="stretch")

        st.markdown("#### Rischio previsto e realizzato")
        walk_percent_columns = [
            "Volatilità prevista media",
            "Volatilità realizzata",
            "CVaR realizzato",
            "Costi cumulati",
        ]
        st.dataframe(
            walk_summary.style
            .format("{:.2%}", subset=walk_percent_columns)
            .format(
                "{:.2f}",
                subset=[
                    "Sharpe realizzato",
                    "Turnover cumulato",
                    "Drag costi su NAV",
                ],
            )
            .format("{:.0f}", subset=["Ribilanciamenti"]),
            width="stretch",
        )
        st.caption(
            "Il rispetto del corridoio in questa tabella è valutato sulla "
            "volatilità effettivamente realizzata fuori campione; può differire "
            "dal vincolo ex ante, che resta sempre rispettato dall'ottimizzatore."
        )

        cost_comparison = pd.DataFrame(
            {
                "NAV lorda": gross_nav_df.iloc[-1],
                "NAV netta": nav_df.iloc[-1],
                "Impatto costi": gross_nav_df.iloc[-1] - nav_df.iloc[-1],
            }
        )
        st.markdown("#### Turnover e impatto dei costi")
        st.dataframe(cost_comparison.style.format("{:.2f}"), width="stretch")
        with st.expander("Dettaglio dei ribilanciamenti", expanded=False):
            st.dataframe(
                rebalance_diagnostics.style
                .format(
                    "{:.2%}",
                    subset=[
                        "Volatilità prevista",
                        "Rendimento previsto",
                        "CVaR previsto",
                        "Costo",
                    ],
                )
                .format("{:.2f}", subset=["Turnover"]),
                width="stretch",
            )
            
        drawdown_df = nav_df / nav_df.cummax() - 1
        with st.expander("📉 Deep Drawdown Analysis", expanded=True):
            fig_dd = px.line(drawdown_df, title="Underwater Plot")
            fig_dd = style_chart(fig_dd, "Drawdown Depth")
            fig_dd.update_traces(fill='tozeroy', line=dict(color='#DC2626')) 
            fig_dd.update_yaxes(tickformat=".1%", range=[None, 0.005]) 
            st.plotly_chart(fig_dd, width="stretch")

    with tab4:
        st.markdown("### 🔗 Network Correlation")
        corr_matrix = returns.corr()
        crisis_corr, crisis_average, crisis_observations, crisis_threshold = (
            calculate_crisis_correlation(returns)
        )
        
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_corr.update_layout(title="Correlation Heatmap", template="plotly_white", height=700, font=dict(family="Fira Sans, sans-serif", color="#334155"))
        st.plotly_chart(fig_corr, width="stretch")

        st.markdown("#### Coppie con elevata sovrapposizione statistica")
        if high_corr_pairs.empty:
            st.info(
                "Nessuna coppia presenta correlazione storica pari o superiore "
                "a 0,70 nel campione caricato."
            )
        else:
            st.dataframe(
                high_corr_pairs.style.format(
                    "{:.2f}",
                    subset=["Correlazione"],
                ),
                width="stretch",
            )

        st.markdown("#### Correlazioni nei periodi di stress")
        st.caption(
            f"Peggior 20% delle osservazioni del paniere equal-weight "
            f"({crisis_observations} periodi; soglia {crisis_threshold:.2%}). "
            f"Correlazione media tra titoli: {crisis_average:.2f}."
        )
        fig_crisis = px.imshow(
            crisis_corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig_crisis.update_layout(
            title="Historical Stress Correlation",
            template="plotly_white",
            height=700,
            font=dict(family="Fira Sans, sans-serif", color="#334155"),
        )
        st.plotly_chart(fig_crisis, width="stretch")
        
        with st.expander("📋 View Raw Data"):
            if HAS_MATPLOTLIB:
                st.dataframe(corr_matrix.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"), height=400, width="stretch")
            else:
                st.dataframe(corr_matrix.style.format("{:.2f}"), height=400, width="stretch")

else:
    st.info("👋 SYSTEM STANDBY. Initialize by uploading CSV data.")
