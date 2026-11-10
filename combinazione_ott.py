import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from html.parser import HTMLParser
import json
import re
from uuid import uuid4

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
# Tema visivo chiaro per i componenti Streamlit
# ---------------------------------------------------------
st.markdown(
    f"<style>{(Path(__file__).parent / '.streamlit' / 'light_theme.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

# Motore quantitativo verificabile separatamente dall'interfaccia Streamlit.
from portfolio_engine import (
    FREQUENCY_LABELS,
    STRESS_INTENSITY,
    VOLATILITY_LINE_GAP,
    bootstrap_portfolio_intervals,
    calculate_concentration_metrics,
    calculate_crisis_correlation,
    calculate_historical_risk_metrics,
    calculate_metrics,
    compute_causal_returns,
    compute_core_stats,
    detect_frequency,
    empirical_cvar,
    estimate_stress_covariance,
    find_high_correlation_pairs,
    minimum_feasible_volatility,
    normalize_price_frequency,
    optimize_basket,
    optimize_ordered_lines,
    process_data_raw,
    run_walk_forward,
    summarize_validation_windows,
    validate_weight_bounds,
)

cached_run_walk_forward = st.cache_data(show_spinner=False)(run_walk_forward)


def style_chart(fig, title):
    chart_colors = ['#087652', '#276E9F', '#7352AC', '#94651B', '#52695D', '#A24A64']
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=18, family="Inter, Helvetica Neue, Arial, sans-serif", color="#17251C"), x=0, y=0.96),
        colorway=chart_colors, 
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=11, color="#465C4E")),
        font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color="#465C4E"),
        xaxis=dict(showgrid=True, gridcolor='#CAD8CF', gridwidth=1, zerolinecolor='#CAD8CF'),
        yaxis=dict(showgrid=True, gridcolor='#CAD8CF', gridwidth=1, zerolinecolor='#CAD8CF')
    )
    return fig


def style_signed_metric(value):
    """Colori tenui per gli esiti numerici, senza alterare il dato."""
    if pd.isna(value):
        return ""
    if value < 0:
        return "color: #B83859; background-color: #FBEFF2"
    return "color: #096C4C; background-color: #E0F1E7"


class _StyledCellsParser(HTMLParser):
    """Extract the displayed body values from a pandas Styler table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_body = False
        self.in_cell = False
        self.rows = []
        self.row = []
        self.cell = []

    def handle_starttag(self, tag, attrs):
        if tag == "tbody":
            self.in_body = True
        elif self.in_body and tag == "tr":
            self.row = []
        elif self.in_body and tag == "td":
            self.in_cell = True
            self.cell = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.in_cell:
            self.row.append("".join(self.cell))
            self.in_cell = False
        elif tag == "tr" and self.in_body:
            self.rows.append(self.row)
        elif tag == "tbody":
            self.in_body = False


def _excel_cell(value):
    """Keep pasted columns intact and use Italian decimal separators for Excel."""
    if value is None or (isinstance(value, (float, np.floating)) and pd.isna(value)):
        return ""
    result = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?", result):
        return result.replace(".", ",")
    if result.startswith(("=", "+", "-", "@")):
        return "'" + result
    return result


def _table_tsv(data):
    """Copy all rows, including index labels, in their displayed order."""
    if isinstance(data, pd.io.formats.style.Styler):
        frame = data.data
        parser = _StyledCellsParser()
        parser.feed(data.to_html())
        body = parser.rows
        if len(body) != len(frame) or any(len(row) != len(frame.columns) for row in body):
            raise ValueError("Impossibile preparare la tabella per la copia.")
    else:
        frame = data
        body = frame.values.tolist()

    index_names = [name if name is not None else "" for name in frame.index.names]
    rows = [index_names + list(frame.columns)]
    for index, values in zip(frame.index, body):
        labels = list(index) if isinstance(index, tuple) else [index]
        rows.append(labels + list(values))
    return "\n".join("\t".join(_excel_cell(value) for value in row) for row in rows)


def copyable_table(data, *, static=False, **kwargs):
    """Show a table with a neighboring button that copies Excel-ready cells."""
    table_column, copy_column = st.columns([7, 1], gap="xsmall")
    with table_column:
        if static:
            st.table(data)
        else:
            st.dataframe(data, **kwargs)

    button_id = "copy_table_" + uuid4().hex
    payload = json.dumps(_table_tsv(data), ensure_ascii=False).replace("<", "\\u003c")
    with copy_column:
        st.html(
            f"""<button id="{button_id}" class="copy-table-button" type="button"
                    title="Copia la tabella per Excel" aria-label="Copia la tabella per Excel">Copia</button>
<script>
(() => {{
  const button = document.getElementById("{button_id}");
  const content = {payload};
  button.addEventListener("click", async () => {{
    try {{
      if (navigator.clipboard && window.isSecureContext) {{
        await navigator.clipboard.writeText(content);
      }} else {{
        const field = document.createElement("textarea");
        field.value = content;
        field.style.position = "fixed";
        field.style.opacity = "0";
        document.body.appendChild(field);
        field.select();
        const copied = document.execCommand("copy");
        field.remove();
        if (!copied) throw new Error("Clipboard unavailable");
      }}
      button.textContent = "Copiato";
      window.setTimeout(() => button.textContent = "Copia", 1800);
    }} catch (error) {{
      button.textContent = "Errore";
      button.title = "Il browser ha bloccato l'accesso agli appunti";
      window.setTimeout(() => button.textContent = "Copia", 2500);
    }}
  }});
}})();
</script>""",
            width="stretch",
            unsafe_allow_javascript=True,
        )


# ---------------------------------------------------------
# INTERFACCIA
# ---------------------------------------------------------
st.title("📈 Portfolio Optimizer")
st.markdown("<p style='color: #465c4e; margin-top: -12px; margin-bottom: 32px; font-family: Inter, Helvetica Neue, Arial, sans-serif'>Constrained Asset Allocation & Risk Analysis System</p>", unsafe_allow_html=True)

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
        "Il minimo di Conservative è impostabile. I minimi delle linee "
        "successive derivano dalla volatilità prevista della linea precedente "
        "più la distanza minima scelta qui sotto."
    )
    line_gap_pp = st.number_input(
        "Distanza minima tra linee (punti percentuali)",
        min_value=VOLATILITY_LINE_GAP * 100,
        max_value=10.0,
        value=0.5,
        step=0.1,
        format="%.1f",
        help=(
            "La volatilità minima di Balanced e Aggressive segue quella "
            "calcolata della linea precedente più questo scarto. "
            "Un valore più alto può rendere impossibile costruire tre linee."
        ),
    )
    line_gap = line_gap_pp / 100
    use_stress = st.toggle(
        "Attiva stress dinamico nell'allocazione",
        value=False,
        key="dynamic_stress_enabled",
        help=(
            "Combina la covarianza ordinaria con la correlazione stimata "
            "nei periodi peggiori. L'intensità è 65% per Conservative, "
            "35% per Balanced e 10% per Aggressive. I corridoi e le "
            "volatilità mostrate restano misurati sulla stessa matrice ordinaria."
        ),
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Minima**")
        min_cons = st.number_input("Conservativa min (%)", 0.0, 100.0, 0.0, 1.0)
        derived_minimums = st.empty()
        derived_minimums.caption("Balanced e Aggressive: calcolo automatico dopo il caricamento.")
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
    min_limits = {"Conservative": min_cons / 100}

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
        minimum_vol = minimum_feasible_volatility(
            cov, min_weights, max_weights, freq
        )
        stress_cov, stress_observations = (
            estimate_stress_covariance(returns, cov)
            if use_stress else (None, 0)
        )
        if use_stress and stress_cov is None:
            raise ValueError(
                "Storico insufficiente per stimare la correlazione di stress "
                "con almeno 10 osservazioni e 2 per titolo nel gruppo peggiore. "
                "Carica uno storico più lungo o disattiva lo scenario di stress."
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
        st.caption(
            f"Volatilità annua minima ottenibile con i limiti sui pesi: "
            f"{minimum_vol:.2%}"
        )
        if use_stress:
            st.caption(
                f"Scenario di stress usato nei pesi: {stress_observations} "
                "periodi peggiori del paniere equiponderato; intensità "
                "65% Conservative, 35% Balanced, 10% Aggressive."
            )
        if len(returns) < 2 * freq:
            st.warning(
                "Il campione comune copre meno di due anni equivalenti. "
                "Le stime di rendimento atteso e i portafogli più aggressivi "
                "possono essere instabili."
            )

    infeasible_maximums = [
        name for name, maximum in max_limits.items()
        if minimum_vol > maximum + 1e-6
    ]
    if infeasible_maximums:
        limits = ", ".join(
            f"{name} (massimo {max_limits[name]:.1%})"
            for name in infeasible_maximums
        )
        st.error(
            f"Corridoi non raggiungibili: {limits}. Con i titoli e i limiti "
            f"sui pesi attuali, la volatilità annua minima è "
            f"{minimum_vol:.2%}. Rivedi i massimi, i titoli selezionati "
            "o i limiti sui pesi."
        )
        st.stop()
        
    metrics = []
    historical_risk = {}
    concentration_metrics = {}
    strategies_config = [
        ("Conservative", "min_cvar"),
        ("Balanced", "max_sharpe"),
        ("Aggressive", "cvar_frontier"),
    ]
    benchmark_strategies_config = [
        ("GMV Ledoit-Wolf", "gmv"),
        ("HRP vincolato", "hrp"),
    ]
    
    try:
        allocations, min_limits = optimize_ordered_lines(
            mu_views.values,
            cov,
            strategies_config,
            min_weights,
            max_weights,
            risk_free_rate,
            max_limits,
            min_limits["Conservative"],
            freq,
            returns,
            gap=line_gap,
            stress_cov=stress_cov,
        )
    except ValueError as error:
        st.error(f"Impossibile costruire le tre linee ordinate: {error}")
        st.stop()

    baseline_allocations = None
    baseline_allocation_error = None
    if use_stress:
        try:
            baseline_allocations, _ = optimize_ordered_lines(
                mu_views.values,
                cov,
                strategies_config,
                min_weights,
                max_weights,
                risk_free_rate,
                max_limits,
                min_cons / 100,
                freq,
                returns,
                gap=line_gap,
            )
        except ValueError as error:
            baseline_allocation_error = str(error)

    derived_minimums.caption(
        f"Balanced min: {min_limits['Balanced']:.2%} · "
        f"Aggressive min: {min_limits['Aggressive']:.2%}"
    )

    for name, _ in strategies_config:
        w = allocations[name]
        try:
            m = calculate_metrics(
                w,
                mu_views.values,
                cov,
                freq,
                risk_free_rate,
            )
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
        except ValueError as error:
            st.error(f"Errore nelle metriche della linea {name}: {error}")
            st.stop()
    
    res_df = pd.DataFrame(metrics, columns=["Linea", "Rendimento", "Volatilità", "Sharpe", "Diversif. Ratio"]).set_index("Linea")
    benchmark_allocations = {}
    benchmark_metrics = []
    available_benchmark_strategies = []
    for name, method in benchmark_strategies_config:
        try:
            weights = optimize_basket(
                mu_views.values,
                cov,
                method,
                min_weights,
                max_weights,
                risk_free_rate,
                frequency=freq,
                returns_history=returns,
            )
            benchmark_allocations[name] = weights
            benchmark_metrics.append(
                [name]
                + list(
                    calculate_metrics(
                        weights,
                        mu_views.values,
                        cov,
                        freq,
                        risk_free_rate,
                    )
                )
            )
            available_benchmark_strategies.append((name, method))
        except ValueError as error:
            st.warning(f"Benchmark '{name}' non disponibile: {error}")

    benchmark_df = pd.DataFrame(
        benchmark_metrics,
        columns=[
            "Benchmark",
            "Rendimento",
            "Volatilità",
            "Sharpe",
            "Diversif. Ratio",
        ],
    ).set_index("Benchmark")
    risk_df = pd.DataFrame(historical_risk).T
    concentration_df = pd.DataFrame(concentration_metrics).T
    bootstrap_intervals, bootstrap_raw = bootstrap_portfolio_intervals(
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
    
    walk_result = None
    walk_error = None
    try:
        walk_result = cached_run_walk_forward(
            causal_returns,
            strategies_config + available_benchmark_strategies,
            min_weights,
            max_weights,
            risk_free_rate,
            max_limits,
            min_limits,
            freq,
            mean_shrinkage,
            transaction_cost_bps,
            ordered_strategy_names=tuple(name for name, _ in strategies_config),
            line_gap=line_gap,
            test_start_fraction=0.50,
            use_stress=use_stress,
        )
    except ValueError as error:
        walk_error = str(error)

    strategy_matrix_df = res_df.rename(
        columns={"Sharpe": "Sharpe previsto"}
    ).copy()
    strategy_matrix_df["Sharpe OOS netto"] = np.nan
    if walk_result is not None:
        (
            nav_df,
            test_start,
            walk_summary,
            gross_nav_df,
            rebalance_diagnostics,
        ) = walk_result
        strategy_matrix_df["Sharpe OOS netto"] = walk_summary.reindex(
            strategy_matrix_df.index
        )["Sharpe realizzato"]
    strategy_matrix_df = strategy_matrix_df[
        [
            "Rendimento",
            "Volatilità",
            "Sharpe previsto",
            "Sharpe OOS netto",
            "Diversif. Ratio",
        ]
    ]

    # --- VISUALIZZAZIONE ---
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 OPTIMIZER", "🥧 ALLOCATION", "📈 SIMULATION", "🔗 CORRELATION"])
    
    with tab1:
        st.markdown("### 🧠 Expected Risk & Return")
        best = res_df.loc["Balanced"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exp. Return", f"{best['Rendimento']:.1%}")
        c2.metric("Exp. Volatility", f"{best['Volatilità']:.1%}")
        c3.metric("Sharpe previsto", f"{best['Sharpe']:.2f}")
        c4.metric("Div. Score", f"{best['Diversif. Ratio']:.2f}")
        
        st.markdown("#### Strategy Matrix")
        st.caption(
            "I rendimenti attesi sono medie winsorizzate con extra-rendimento "
            "ridotto verso zero. Conservative parte dal minimo CVaR; "
            "Aggressive seleziona il punto di ginocchio della frontiera "
            "rendimento–CVaR. Tutte le linee rispettano "
            "i corridoi di volatilità correnti. Lo Sharpe OOS netto usa soltanto "
            "dati successivi alla stima dei pesi e include i costi."
        )
        if use_stress:
            st.caption(
                "Con lo stress attivo, Conservative riduce anche la varianza "
                "della matrice mista entro un aumento massimo del 2% circa "
                "del CVaR minimo; Balanced usa la matrice mista nello Sharpe "
                "da ottimizzare; Aggressive ne considera il rischio nella "
                "scelta sulla frontiera."
            )
        if HAS_MATPLOTLIB:
            copyable_table(
                strategy_matrix_df.style
                .format("{:.1%}", subset=["Rendimento", "Volatilità"])
                .format(
                    "{:.2f}",
                    subset=[
                        "Sharpe previsto",
                        "Sharpe OOS netto",
                        "Diversif. Ratio",
                    ],
                )
                .map(style_signed_metric, subset=["Sharpe OOS netto"]),
                static=True,
            )
        else:
            copyable_table(
                strategy_matrix_df.style
                .format("{:.1%}", subset=["Rendimento", "Volatilità"])
                .format(
                    "{:.2f}",
                    subset=[
                        "Sharpe previsto",
                        "Sharpe OOS netto",
                        "Diversif. Ratio",
                    ],
                ),
                static=True,
            )
        if walk_error is not None:
            st.info(f"Sharpe OOS non disponibile: {walk_error}")

        st.markdown("#### Corridoi di volatilità attivi")
        st.caption(
            "Il minimo di Balanced è la volatilità calcolata di Conservative "
            f"più {line_gap_pp:.1f} punti percentuali; quello di Aggressive deriva allo "
            "stesso modo da Balanced. Se un corridoio non è fattibile, il "
            "calcolo si interrompe senza allentare i vincoli."
        )
        copyable_table(
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

        if use_stress:
            st.markdown("#### Effetto delle correlazioni di stress sui pesi")
            st.caption(
                "La correlazione dei periodi peggiori è stimata con shrinkage "
                "e combinata con la matrice ordinaria nella scelta dei pesi. "
                "La quota di stress è maggiore per Conservative e minore "
                "per Aggressive. Le volatilità ordinarie usano per tutte "
                "le linee la stessa matrice."
            )
            if baseline_allocations is None:
                st.warning(
                    "Confronto dei pesi senza stress non disponibile: "
                    f"{baseline_allocation_error}"
                )
            else:
                impact_rows = []
                for name, _ in strategies_config:
                    base_weights = baseline_allocations[name]
                    stressed_weights = allocations[name]
                    impact_rows.append({
                        "Linea": name,
                        "Intensità stress": STRESS_INTENSITY[name],
                        "Distanza pesi": 0.5 * np.abs(
                            stressed_weights - base_weights
                        ).sum(),
                        "Vol. ordinaria senza stress": np.sqrt(
                            base_weights @ cov @ base_weights * freq
                        ),
                        "Vol. ordinaria con stress": np.sqrt(
                            stressed_weights @ cov @ stressed_weights * freq
                        ),
                        "Vol. scenario senza stress": np.sqrt(
                            base_weights @ stress_cov @ base_weights * freq
                        ),
                        "Vol. scenario con stress": np.sqrt(
                            stressed_weights @ stress_cov @ stressed_weights * freq
                        ),
                    })
                impact_df = pd.DataFrame(impact_rows).set_index("Linea")
                copyable_table(impact_df.style.format("{:.2%}"), width="stretch")
                if (impact_df["Distanza pesi"] < 1e-6).all():
                    st.info(
                        "Su questi dati lo scenario è entrato nell'ottimizzazione "
                        "ma non ha cambiato i pesi in misura apprezzabile."
                    )
        
        if not benchmark_df.empty:
            st.markdown("#### Benchmark diagnostici")
            st.caption(
                "GMV e HRP rispettano i limiti minimi e massimi per titolo, "
                "ma non sono forzati nei corridoi delle tre linee. Servono come "
                "controllo indipendente, non come quarta o quinta proposta."
            )
            copyable_table(
                benchmark_df.style
                .format("{:.1%}", subset=["Rendimento", "Volatilità"])
                .format("{:.2f}", subset=["Sharpe", "Diversif. Ratio"]),
                width="stretch",
            )

        res_export = strategy_matrix_df.copy()
        for col in ["Rendimento", "Volatilità"]: res_export[col] = res_export[col].map('{:.1%}'.format)
        for col in ["Sharpe previsto", "Sharpe OOS netto", "Diversif. Ratio"]: res_export[col] = res_export[col].map('{:.2f}'.format)
        
        st.download_button(label="📥 DOWNLOAD METRICS (CSV)", data=res_export.to_csv().encode('utf-8'), file_name='ai_metrics.csv', mime='text/csv')

        fig_ef = px.scatter(res_df, x="Volatilità", y="Rendimento", color="Sharpe", size=[50]*len(res_df), text=res_df.index, color_continuous_scale=[[0, "#A8D7B8"], [0.5, "#39A878"], [1, "#096C4C"]], title="Risk / Return Comparison")
        fig_ef.update_traces(textposition=['top right', 'top center', 'top left'], textfont=dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=12, color="#465C4E"))
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
        copyable_table(
            risk_df.style
            .format("{:.2%}", subset=percent_columns)
            .format("{:.2f}", subset=["Sortino"]),
            width="stretch",
        )

        st.markdown("#### Separazione effettiva dei profili")
        bootstrap_volatility = bootstrap_raw.pivot(
            index="Simulazione", columns="Linea", values="Volatilità"
        )
        profile_rows = []
        for previous, following in [
            ("Conservative", "Balanced"),
            ("Balanced", "Aggressive"),
        ]:
            sampled_gap = (
                bootstrap_volatility[following]
                - bootstrap_volatility[previous]
            )
            profile_rows.append(
                {
                    "Confronto": f"{following} vs {previous}",
                    "Scarto volatilità prevista": (
                        res_df.loc[following, "Volatilità"]
                        - res_df.loc[previous, "Volatilità"]
                    ),
                    "Scarto CVaR storico": (
                        risk_df.loc[following, "CVaR 95% (periodo)"]
                        - risk_df.loc[previous, "CVaR 95% (periodo)"]
                    ),
                    "Scarto Max Drawdown": (
                        risk_df.loc[previous, "Max Drawdown"]
                        - risk_df.loc[following, "Max Drawdown"]
                    ),
                    "Distanza pesi": (
                        0.5
                        * np.abs(allocations[following] - allocations[previous]).sum()
                    ),
                    "Ricampionamenti con ordine vol.": (sampled_gap > 0).mean(),
                    "Scarto vol. 5% bootstrap": sampled_gap.quantile(0.05),
                }
            )
        profile_separation = pd.DataFrame(profile_rows).set_index("Confronto")
        copyable_table(
            profile_separation.style.format("{:.2%}"),
            width="stretch",
        )
        st.caption(
            "Scarti positivi di CVaR e Max Drawdown indicano maggior rischio "
            "nella linea successiva. Distanza pesi = metà della somma delle "
            "differenze assolute. Il bootstrap ricampiona le stesse serie con "
            "pesi fissi: descrive la stabilità nel campione, non garantisce "
            "l'ordine del rischio futuro."
        )
        weak_pairs = profile_separation.index[
            (profile_separation["Scarto CVaR storico"] <= 0)
            | (profile_separation["Scarto Max Drawdown"] <= 0)
            | (profile_separation["Scarto vol. 5% bootstrap"] <= 0)
        ].tolist()
        if weak_pairs:
            st.warning(
                "I profili non risultano chiaramente separati su tutte le "
                "misure di rischio per: " + ", ".join(weak_pairs)
                + ". Aumentare lo scarto di volatilità può aiutare, ma può "
                "rendere i vincoli non fattibili."
            )
        if (
            res_df.loc["Aggressive", "Rendimento"]
            <= res_df.loc["Balanced", "Rendimento"]
        ):
            st.warning(
                "Aggressive ha più volatilità prevista ma rendimento atteso "
                "non superiore a Balanced: sulla coppia rendimento-volatilità "
                "è dominata. Le stime di rendimento restano incerte."
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
        copyable_table(
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
                return ['background-color: rgba(8, 118, 82, 0.08)'] * len(s)
            elif s.name == 'Balanced':
                return ['background-color: rgba(39, 110, 159, 0.08)'] * len(s)
            elif s.name == 'Aggressive':
                return ['background-color: rgba(115, 82, 172, 0.08)'] * len(s)
            return [''] * len(s)
            
        styled_w = w_clean.style.format("{:.1%}").apply(highlight_cols, axis=0)
        copyable_table(styled_w, height=500, width="stretch")

        if benchmark_allocations:
            with st.expander("Pesi dei benchmark diagnostici", expanded=False):
                benchmark_weights_df = pd.DataFrame(
                    benchmark_allocations,
                    index=returns.columns,
                )
                copyable_table(
                    benchmark_weights_df.style.format("{:.1%}"),
                    height=500,
                    width="stretch",
                )
        
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
        copyable_table(
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
        if walk_result is None:
            st.warning(
                f"Validazione fuori campione non disponibile: {walk_error}"
            )
            st.stop()
        st.caption(
            f"Periodo fuori campione dal {nav_df.index[1].date()} · "
            f"Ribilanciamento periodico causale · Costi: "
            f"{transaction_cost_bps} bps sul turnover one-way · "
            "Benchmark: Equal Weight vincolato, GMV Ledoit-Wolf e HRP vincolato"
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
                copyable_table(
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
                    .map(style_signed_metric, subset=["CAGR"]),
                    static=True,
                )
            else:
                copyable_table(perf_hist, static=True)
            
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
        copyable_table(
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
            "volatilità effettivamente realizzata fuori campione. Per Balanced "
            "e Aggressive il campo non è applicabile: il minimo previsto cambia "
            "a ogni ribilanciamento. La volatilità realizzata può non mantenere "
            "l'ordine delle linee. Anche i benchmark non hanno un corridoio "
            "di linea."
        )

        st.markdown("#### Finestre cronologiche fuori campione")
        validation_windows = summarize_validation_windows(
            nav_df, freq, risk_free_rate
        )
        window_count = validation_windows["Finestra"].nunique()
        st.caption(
            f"{window_count} finestre consecutive senza sovrapposizione, "
            "con pesi ricalcolati usando solo le osservazioni precedenti e "
            "risultati netti dei costi. Il test inizia dopo almeno il 50% "
            "dello storico e almeno 12 osservazioni o 2 per titolo. "
            "Le finestre condividono il metodo e l'addestramento progressivo."
        )
        if window_count < 3:
            st.warning(
                "Storico fuori campione insufficiente per tre finestre con "
                "almeno cinque osservazioni ciascuna."
            )
        copyable_table(
            validation_windows.set_index(["Finestra", "Linea"]).style
            .format(
                "{:.2%}",
                subset=[
                    "Rendimento annuo", "Volatilità",
                    "CVaR 95% (periodo)", "Max Drawdown",
                ],
            )
            .format("{:.2f}", subset=["Sharpe"]),
            width="stretch",
        )

        if use_stress:
            st.markdown("#### Stress contro allocazione ordinaria fuori campione")
            compare_baseline = st.toggle(
                "Calcola il confronto fuori campione senza stress",
                value=False,
            )
            baseline_walk_result = None
            baseline_walk_error = None
            if compare_baseline:
                try:
                    with st.spinner("Calcolo il confronto senza stress..."):
                        baseline_walk_result = cached_run_walk_forward(
                            causal_returns,
                            strategies_config + available_benchmark_strategies,
                            min_weights,
                            max_weights,
                            risk_free_rate,
                            max_limits,
                            {"Conservative": min_cons / 100},
                            freq,
                            mean_shrinkage,
                            transaction_cost_bps,
                            ordered_strategy_names=tuple(
                                name for name, _ in strategies_config
                            ),
                            line_gap=line_gap,
                            test_start_fraction=0.50,
                            use_stress=False,
                        )
                except ValueError as error:
                    baseline_walk_error = str(error)
            if compare_baseline and baseline_walk_result is None:
                st.warning(
                    "Confronto fuori campione senza stress non disponibile: "
                    f"{baseline_walk_error}"
                )
            elif baseline_walk_result is not None:
                baseline_nav, _, baseline_summary, _, _ = baseline_walk_result
                comparison_rows = []
                for name, _ in strategies_config:
                    comparison_rows.append({
                        "Linea": name,
                        "Sharpe netto con stress": walk_summary.loc[
                            name, "Sharpe realizzato"
                        ],
                        "Sharpe netto senza stress": baseline_summary.loc[
                            name, "Sharpe realizzato"
                        ],
                        "CVaR con stress": walk_summary.loc[
                            name, "CVaR realizzato"
                        ],
                        "CVaR senza stress": baseline_summary.loc[
                            name, "CVaR realizzato"
                        ],
                        "NAV finale con stress": nav_df[name].iloc[-1],
                        "NAV finale senza stress": baseline_nav[name].iloc[-1],
                    })
                stress_comparison = pd.DataFrame(comparison_rows).set_index("Linea")
                copyable_table(
                    stress_comparison.style
                    .format("{:.2f}", subset=[
                        "Sharpe netto con stress", "Sharpe netto senza stress",
                        "NAV finale con stress", "NAV finale senza stress",
                    ])
                    .format("{:.2%}", subset=[
                        "CVaR con stress", "CVaR senza stress",
                    ]),
                    width="stretch",
                )
                st.caption(
                    "Confronto sulla stessa sequenza fuori campione e con gli "
                    "stessi costi. Sharpe più alto e CVaR più basso sono "
                    "preferibili, ma i risultati storici non dimostrano un "
                    "vantaggio futuro. Il confronto non è usato per "
                    "selezionare i pesi, così non introduce dati futuri "
                    "nell'ottimizzazione."
                )
                weaker = stress_comparison.index[
                    (stress_comparison["Sharpe netto con stress"]
                     <= stress_comparison["Sharpe netto senza stress"])
                    & (stress_comparison["CVaR con stress"]
                       >= stress_comparison["CVaR senza stress"])
                ].tolist()
                if weaker:
                    st.warning(
                        "Nel campione fuori campione lo stress non migliora "
                        "né Sharpe netto né CVaR per: " + ", ".join(weaker)
                    )

        cost_comparison = pd.DataFrame(
            {
                "NAV lorda": gross_nav_df.iloc[-1],
                "NAV netta": nav_df.iloc[-1],
                "Impatto costi": gross_nav_df.iloc[-1] - nav_df.iloc[-1],
            }
        )
        st.markdown("#### Turnover e impatto dei costi")
        copyable_table(cost_comparison.style.format("{:.2f}"), width="stretch")
        with st.expander("Dettaglio dei ribilanciamenti", expanded=False):
            copyable_table(
                rebalance_diagnostics.style
                .format(
                    "{:.2%}",
                    subset=[
                        "Volatilità minima",
                        "Volatilità prevista",
                        "Rendimento previsto",
                        "CVaR previsto",
                        "Costo",
                    ],
                    na_rep="—",
                )
                .format("{:.2f}", subset=["Turnover"]),
                width="stretch",
            )
            
        drawdown_df = nav_df / nav_df.cummax() - 1
        with st.expander("📉 Deep Drawdown Analysis", expanded=True):
            fig_dd = px.line(drawdown_df, title="Underwater Plot")
            fig_dd = style_chart(fig_dd, "Drawdown Depth")
            fig_dd.update_traces(fill='tozeroy', line=dict(color='#B83859'))
            fig_dd.update_yaxes(tickformat=".1%", range=[None, 0.005]) 
            st.plotly_chart(fig_dd, width="stretch")

    with tab4:
        st.markdown("### 🔗 Network Correlation")
        corr_matrix = returns.corr()
        crisis_corr, crisis_average, crisis_observations, crisis_threshold = (
            calculate_crisis_correlation(returns)
        )
        
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_corr.update_layout(title="Correlation Heatmap", template="plotly_white", height=700, paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color="#465C4E"))
        st.plotly_chart(fig_corr, width="stretch")

        st.markdown("#### Coppie con elevata sovrapposizione statistica")
        if high_corr_pairs.empty:
            st.info(
                "Nessuna coppia presenta correlazione storica pari o superiore "
                "a 0,70 nel campione caricato."
            )
        else:
            copyable_table(
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
            f"Correlazione media tra titoli: {crisis_average:.2f}. "
            "Questa matrice non filtrata è descrittiva."
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
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color="#465C4E"),
        )
        st.plotly_chart(fig_crisis, width="stretch")

        if use_stress:
            st.markdown("#### Matrice dinamica usata nell'allocazione")
            st.caption(
                f"Stimata su {stress_observations} periodi negativi con "
                "shrinkage; durante il walk-forward viene ricalcolata "
                "esclusivamente sui dati già osservati. Le volatilità "
                "dei singoli titoli restano quelle della matrice ordinaria."
            )
            base_std = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
            applied_stress_corr = pd.DataFrame(
                stress_cov / np.outer(base_std, base_std),
                index=returns.columns,
                columns=returns.columns,
            )
            fig_applied_stress = px.imshow(
                applied_stress_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
            fig_applied_stress.update_layout(
                title="Stress Correlation Used for Allocation",
                template="plotly_white",
                height=700,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, Helvetica Neue, Arial, sans-serif", color="#465C4E"),
            )
            st.plotly_chart(fig_applied_stress, width="stretch")
            with st.expander("Valori della matrice dinamica", expanded=False):
                copyable_table(
                    applied_stress_corr.style.format("{:.2f}"),
                    height=400,
                    width="stretch",
                )
        
        with st.expander("📋 View Raw Data"):
            if HAS_MATPLOTLIB:
                copyable_table(corr_matrix.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"), height=400, width="stretch")
            else:
                copyable_table(corr_matrix.style.format("{:.2f}"), height=400, width="stretch")

else:
    st.info("👋 SYSTEM STANDBY. Initialize by uploading CSV data.")
