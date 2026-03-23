import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from zipfile import ZipFile
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

# Gestione sicura di Matplotlib per lo styling
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ---------------------------------------------------------
st.set_page_config(
    layout="wide", 
    page_title="AI Portfolio Optimizer", 
    page_icon="🤖"
)

# ---------------------------------------------------------
# CSS THEME: AI / CYBERPUNK
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp { 
        background-color: #1A202C; 
        color: #FFFFFF;
        font-family: 'Inter', sans-serif; 
    }
    
    p, span, div, label, li { color: #FFFFFF !important; }

    [data-testid="stSidebar"] { 
        background-color: #161B22; 
        border-right: 1px solid #30363D; 
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #E0E0E0 !important;
    }
    
    [data-testid="stExpander"] summary p {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    h1, h2, h3 { 
        color: #FFFFFF !important; 
        font-weight: 700; 
        letter-spacing: 0.05em; 
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
    }
    h1 { font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; }
    
    div[data-testid="metric-container"] { 
        background-color: #1C2128; 
        border: 1px solid #30363D; 
        border-left: 4px solid #00FFFF; 
        border-radius: 4px; 
        padding: 16px; 
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5); 
    }
    [data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #CCCCCC !important; }
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; color: #00FFFF !important; text-shadow: 0 0 5px rgba(0, 255, 255, 0.5); }

    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button { 
        background: linear-gradient(90deg, #21262d 0%, #0d1117 100%); 
        color: #00FFFF !important; 
        border: 1px solid #30363D; 
        border-radius: 4px; 
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600; 
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { 
        border-color: #00FFFF; 
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
        color: #FFFFFF !important;
    }

    [data-testid="stFileUploader"] { background-color: #161B22; border-radius: 4px; padding: 10px; }
    [data-testid="stFileUploader"] section { background-color: #161B22 !important; border: 1px dashed #30363D !important; }
    [data-testid="stFileUploader"] section > div > div > span, [data-testid="stFileUploader"] section > div > div > small { color: #E0E0E0 !important; }
    [data-testid="stFileUploader"] button { background: linear-gradient(90deg, #21262d 0%, #0d1117 100%); color: #00FFFF !important; border: 1px solid #30363D; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #161B22; border-radius: 4px; color: #8B949E !important; font-family: 'JetBrains Mono', monospace; border: 1px solid transparent; }
    .stTabs [aria-selected="true"] { background-color: #21262D; color: #00FFFF !important; border: 1px solid #30363D; border-bottom: 2px solid #00FFFF; }
    
    [data-testid="stDataFrame"] { border: 1px solid #30363D; }
    thead tr th { background-color: #161B22 !important; color: #00FFFF !important; font-family: 'JetBrains Mono', monospace; }
    tbody tr td { color: #FFFFFF !important; font-family: 'Inter', sans-serif; background-color: #0E1117 !important; }
    
    .stSelectbox > div > div, .stMultiSelect > div > div { background-color: #0D1117; color: #FFFFFF !important; border-color: #30363D; }
    div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #161B22 !important; border: 1px solid #30363D; }
    div[data-baseweb="popover"] li, div[data-baseweb="menu"] li { background-color: #161B22 !important; color: #FFFFFF !important; }
    div[data-baseweb="popover"] li:hover, div[data-baseweb="menu"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"], div[data-baseweb="menu"] li[aria-selected="true"] { background-color: #21262D !important; color: #00FFFF !important; }
    [data-baseweb="select"] svg { fill: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CORE FUNCTIONS (ENGINEERING & QUANT FIXES)
# ---------------------------------------------------------

def detect_frequency(df):
    if len(df) < 3: return 12 
    days_diff = (df.index[1] - df.index[0]).days
    if days_diff >= 28: return 12 
    elif days_diff >= 5: return 52 
    else: return 252 

@st.cache_data(show_spinner=False)
def process_data_raw(file_bytes, filename):
    separators = [';', ',', '\t']
    for sep in separators:
        try:
            df = pd.read_csv(BytesIO(file_bytes), sep=sep, index_col=0, parse_dates=True, dayfirst=True)
            if not df.index.is_unique: df = df[~df.index.duplicated(keep='first')]
            if df.shape[1] > 0:
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace('€','').str.replace('$','').str.strip()
                        if df[col].str.contains(',').any() and not df[col].str.contains('\.').any():
                             df[col] = df[col].str.replace(',', '.')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                # NON facciamo dropna qui per evitare di distruggere dati prematuramente
                return df, None
        except Exception as e:
            continue
    return None, "Impossibile parsare il file CSV."

@st.cache_data(ttl=86400, show_spinner=False)
def get_ff5():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            z = ZipFile(BytesIO(r.content))
            f = z.open([x for x in z.namelist() if 'Factors' in x][0])
            df = pd.read_csv(f, skiprows=3).rename(columns={'Mkt-RF':'MKT','RF':'RF'}).dropna()
            df['Date'] = pd.to_datetime(df['Unnamed: 0'].astype(str), format='%Y%m', errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date')
            if not df.index.is_unique: df = df[~df.index.duplicated(keep='first')]
            simple_ret = df[['MKT','SMB','HML','RF']].astype(float) / 100
            return np.log(1 + simple_ret)
    except Exception as e:
        st.warning(f"Warning: Impossibile scaricare dati Fama-French ({str(e)}). Verranno usati valori proxy.")
        pass
    dates = pd.date_range('2000-01-01', datetime.today(), freq='ME')
    return pd.DataFrame({'MKT':0.006, 'SMB':0.002, 'HML':0.003, 'RF':0.002}, index=dates)

@st.cache_data(show_spinner=False)
def compute_core_stats(df_subset):
    """Isola i calcoli pesanti. Riceve solo le colonne scelte."""
    df_clean = df_subset.dropna(how='any')
    if df_clean.empty or len(df_clean) < 3:
        raise ValueError("Dati insufficienti dopo l'allineamento degli asset storici.")
    
    returns_log = np.log(df_clean / df_clean.shift(1)).dropna()
    freq = detect_frequency(df_clean)
    
    ff5_log = get_ff5()
    
    # Calcolo Views
    rf_annualized = 0.02 # fallback risk free
    try:
        returns_monthly_proxy = returns_log.resample('ME').sum()
        returns_monthly_proxy.index = pd.to_datetime(returns_monthly_proxy.index)
        ff5_log.index = pd.to_datetime(ff5_log.index)
        common_idx = returns_monthly_proxy.index.intersection(ff5_log.index)
        
        if len(common_idx) >= 12:
            r_aligned = returns_monthly_proxy.loc[common_idx]
            f_aligned = ff5_log.loc[common_idx]
            data = pd.concat([r_aligned, f_aligned], axis=1).iloc[-60:]
            
            X = sm.add_constant(data[['MKT','SMB','HML']])
            factors_mean = data[['MKT','SMB','HML']].mean()
            rf_mean = data['RF'].mean()
            rf_annualized = rf_mean * 12
            
            views_monthly = {}
            for asset in returns_log.columns:
                try:
                    model = sm.OLS(data[asset] - data['RF'], X).fit()
                    views_monthly[asset] = rf_mean + model.params['const'] + (model.params[['MKT','SMB','HML']] * factors_mean).sum()
                except:
                    views_monthly[asset] = data[asset].mean()
            
            views_native = {k: v * (12 / freq) for k, v in views_monthly.items()}
            mu = pd.Series(views_native)
        else:
            raise ValueError("Allineamento Fama-French fallito per storico troppo breve.")
    except Exception as e:
        # Fallback intelligente: Shrinkage verso la media globale invece che media pura (riduce l'error maximization)
        raw_means = returns_log.mean()
        global_mean = raw_means.mean()
        mu = (raw_means * 0.5) + (global_mean * 0.5) 

    cov_log = LedoitWolf().fit(returns_log).covariance_
    
    return returns_log, freq, mu, cov_log, rf_annualized

def calculate_metrics(weights, mu, cov, freq_factor, risk_free_rate):
    ret_log = np.dot(weights, mu) * freq_factor
    var_log = np.dot(weights.T, np.dot(cov, weights))
    vol_log = np.sqrt(var_log) * np.sqrt(freq_factor)
    # Calcolo Sharpe coerente con il Risk Free calcolato dinamicamente
    sharpe = (ret_log - risk_free_rate) / vol_log if vol_log > 0 else 0
    
    asset_vols = np.sqrt(np.diag(cov)) * np.sqrt(freq_factor)
    weighted_vol = np.dot(weights, asset_vols)
    div_ratio = weighted_vol / vol_log if vol_log > 0 else 0
    
    return ret_log, vol_log, sharpe, div_ratio

def optimize_basket(mu, cov, optimization_type, min_w, max_w, risk_free_rate, max_vol=None, min_vol=None, freq_factor=12):
    n = len(mu)
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((min_w, max_w) for _ in range(n))
    init = np.full(n, 1/n) 

    def get_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(freq_factor)
    
    def get_ret(w):
        return np.dot(w, mu) * freq_factor

    # I vincoli vengono forzati rigidamente. Se fallisce, solleva eccezione.
    if max_vol is not None:
        cons.append({'type': 'ineq', 'fun': lambda w, max_v=max_vol: max_v - get_vol(w)})
    if min_vol is not None:
        cons.append({'type': 'ineq', 'fun': lambda w, min_v=min_vol: get_vol(w) - min_v})

    if optimization_type == "min_vol":
        fun = lambda w: np.dot(w.T, np.dot(cov, w))
    elif optimization_type == "max_sharpe":
        fun = lambda w: - ((get_ret(w) - risk_free_rate) / get_vol(w))
    else: 
        fun = lambda w: -get_ret(w)

    res = minimize(fun, init, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-8, options={'maxiter': 2000})
    
    if not res.success:
        raise ValueError(f"Ottimizzatore incagliato. {res.message}")
    return res.x

def run_backtest(returns_log, allocations):
    # CORREZIONE MATEMATICA: Trasformazione log-rendimenti in rendimenti semplici per aggregazione lineare di portafoglio
    simple_returns = np.exp(returns_log) - 1
    
    nav_data = {}
    for name, weights in allocations.items():
        # Rendimento portafoglio è media ponderata dei rendimenti semplici
        port_simple_ret = simple_returns.dot(weights)
        # NAV cumulato moltiplicando 1 + rendimento
        nav = 100 * np.cumprod(1 + port_simple_ret)
        
        try:
            start_date = returns_log.index[0] - timedelta(days=1)
            nav_with_start = pd.concat([pd.Series([100], index=[start_date]), nav])
            nav_data[name] = nav_with_start
        except:
            nav_data[name] = nav
        
    return pd.DataFrame(nav_data).ffill()

def style_chart(fig, title):
    neon_colors = ['#00FFFF', '#BD00FF', '#00FF9D', '#29B5E8', '#FF0055']
    fig.update_layout(
        template="plotly_dark", 
        title=dict(text=f"<b>{title}</b>", font=dict(size=18, family="JetBrains Mono, monospace", color="#00FFFF"), x=0, y=0.96),
        colorway=neon_colors, 
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="JetBrains Mono, monospace", size=11, color="#E0E0E0")),
        font=dict(family="Inter, sans-serif", color="#E0E0E0"),
        xaxis=dict(showgrid=True, gridcolor='#30363D', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='#30363D', gridwidth=1)
    )
    return fig

# ---------------------------------------------------------
# INTERFACCIA
# ---------------------------------------------------------
st.title("🤖 AI Portfolio Optimizer")
st.markdown("<p style='color: #FFFFFF; margin-top: -15px; margin-bottom: 30px; font-family: JetBrains Mono'>Neural-Enhanced Asset Allocation & Risk Analysis System</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ SYSTEM PARAMETERS")
min_w = st.sidebar.slider("Min % per Asset", 0.0, 0.2, 0.0, 0.01)
max_w = st.sidebar.slider("Max % per Asset", 0.1, 1.0, 0.35, 0.05)
st.sidebar.markdown("---")

vol_options = ["Nessun Limite"] + [f"{i}%" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 40, 50]]
def parse_vol_choice(choice):
    if choice == "Nessun Limite": return None
    return float(choice.replace('%', '')) / 100

with st.sidebar.expander("📉 Parametri Volatilità", expanded=True):
    st.caption("Define volatility corridors.")
    with st.form("volatility_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ceiling (Max)**")
            max_cons = st.selectbox("Max Vol Cons.", vol_options, index=0) 
            max_bal = st.selectbox("Max Vol Bal.", vol_options, index=0)    
            max_agg = st.selectbox("Max Vol Agg.", vol_options, index=0)    
        with c2:
            st.markdown("**Floor (Min)**")
            min_cons = st.selectbox("Min Vol Cons.", vol_options, index=0)
            min_bal = st.selectbox("Min Vol Bal.", vol_options, index=0)
            min_agg = st.selectbox("Min Vol Agg.", vol_options, index=0)
        submit_vol = st.form_submit_button("RIBILANCIA")
    
    max_limits = {"Conservative": parse_vol_choice(max_cons), "Balanced": parse_vol_choice(max_bal), "Aggressive": parse_vol_choice(max_agg)}
    min_limits = {"Conservative": parse_vol_choice(min_cons), "Balanced": parse_vol_choice(min_bal), "Aggressive": parse_vol_choice(min_agg)}

# Main
col_up, col_info = st.columns([1, 2])
with col_up:
    uploaded = st.file_uploader("📂 INGEST DATA (CSV)", type='csv')

if uploaded:
    # Parsing sicuro senza distruzione dati
    df_raw, error_msg = process_data_raw(uploaded.getvalue(), uploaded.name)
    
    if df_raw is None:
        st.error(error_msg); st.stop()
        
    available_assets = list(df_raw.columns)
    with col_info:
        st.success("✅ SYSTEM READY: Data Ingested")
        selected_assets = st.multiselect("Active Assets:", options=available_assets, default=available_assets)
    
    if len(selected_assets) < 2: st.warning("⚠️ CRITICAL: Select at least 2 assets."); st.stop()

    # Elaborazione Core Cacheata
    try:
        returns, freq, mu_views, cov, rf_rate = compute_core_stats(df_raw[selected_assets])
    except ValueError as e:
        st.error(f"Errore computazionale: {str(e)}"); st.stop()
        
    allocations = {}
    metrics = []
    strategies_config = [("Conservative", "min_vol"), ("Balanced", "max_sharpe"), ("Aggressive", "max_return")]
    
    calc_success = True
    
    for name, method in strategies_config:
        mx_vol = max_limits[name]
        mn_vol = min_limits[name]
        
        try:
            w = optimize_basket(mu_views.values, cov, method, min_w, max_w, rf_rate, max_vol=mx_vol, min_vol=mn_vol, freq_factor=freq)
            m = calculate_metrics(w, mu_views.values, cov, freq, rf_rate)
            allocations[name] = w
            metrics.append([name] + list(m))
        except ValueError as e:
            st.error(f"❌ FALLIMENTO STRATEGIA '{name}': {str(e)} \nI vincoli di volatilità imposti per questa linea sono matematicamente irrealizzabili. Modifica i parametri nella sidebar e clicca Ribilancia.")
            calc_success = False
            break # Blocca l'esecuzione se un'ottimizzazione fallisce

    if not calc_success:
        st.stop() # Ferma il rendering delle tabelle se mancano dati
    
    res_df = pd.DataFrame(metrics, columns=["Linea", "Rendimento", "Volatilità", "Sharpe", "Diversif. Ratio"]).set_index("Linea")
    
    # --- VISUALIZZAZIONE ---
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 OPTIMIZER", "🥧 ALLOCATION", "📈 SIMULATION", "🔗 CORRELATION"])
    
    with tab1:
        st.markdown("### 🧠 Predictive Performance")
        best = res_df.loc["Balanced"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exp. Return", f"{best['Rendimento']:.1%}")
        c2.metric("Exp. Volatility", f"{best['Volatilità']:.1%}")
        c3.metric("Sharpe Ratio", f"{best['Sharpe']:.2f}")
        c4.metric("Div. Score", f"{best['Diversif. Ratio']:.2f}")
        
        st.markdown("#### Strategy Matrix")
        if HAS_MATPLOTLIB:
            st.table(res_df.style.format("{:.1%}", subset=["Rendimento", "Volatilità"]).format("{:.2f}", subset=["Sharpe", "Diversif. Ratio"]).background_gradient(cmap="viridis", subset=["Rendimento", "Sharpe"]))
        else:
            st.table(res_df.style.format("{:.1%}", subset=["Rendimento", "Volatilità"]).format("{:.2f}", subset=["Sharpe", "Diversif. Ratio"]))
        
        res_export = res_df.copy()
        for col in ["Rendimento", "Volatilità"]: res_export[col] = res_export[col].map('{:.1%}'.format)
        for col in ["Sharpe", "Diversif. Ratio"]: res_export[col] = res_export[col].map('{:.2f}'.format)
        
        st.download_button(label="📥 DOWNLOAD METRICS (CSV)", data=res_export.to_csv().encode('utf-8'), file_name='ai_metrics.csv', mime='text/csv')

        fig_ef = px.scatter(res_df, x="Volatilità", y="Rendimento", color="Sharpe", size=[50]*len(res_df), text=res_df.index, color_continuous_scale="Viridis", title="Efficient Frontier")
        fig_ef.add_trace(go.Scatter(x=res_df["Volatilità"], y=res_df["Rendimento"], mode='lines', line=dict(color='#8B949E', dash='dash', width=1), showlegend=False))
        fig_ef.update_traces(textposition='top center', textfont=dict(family="JetBrains Mono, monospace", size=12))
        fig_ef = style_chart(fig_ef, "Efficient Frontier Analysis")
        fig_ef.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
        st.plotly_chart(fig_ef, use_container_width=True)

    with tab2:
        st.markdown("### 🧬 Asset DNA (Composition)")
        weights_df = pd.DataFrame(allocations, index=returns.columns)
        w_clean = weights_df.sort_values(by="Balanced", ascending=False)
        
        if HAS_MATPLOTLIB:
            st.dataframe(w_clean.style.format("{:.1%}").background_gradient(cmap="winter", axis=None), height=500, use_container_width=True)
        else:
            st.dataframe(w_clean.style.format("{:.1%}"), height=500, use_container_width=True)
        
        w_export = w_clean.applymap(lambda x: '{:.1%}'.format(x))
        st.download_button(label="📥 DOWNLOAD ALLOCATION (CSV)", data=w_export.to_csv().encode('utf-8'), file_name='ai_allocation.csv', mime='text/csv')

        df_melt = w_clean.reset_index().melt(id_vars="index", var_name="Strategia", value_name="Peso")
        fig_bar = px.bar(df_melt, x="Strategia", y="Peso", color="index", text_auto=".0%", title="Allocation Breakdown")
        fig_bar = style_chart(fig_bar, "Strategic Allocation")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.markdown(f"### 🕰️ Backtest Simulation")
        nav_df = run_backtest(returns, allocations)
        
        days = (nav_df.index[-1] - nav_df.index[0]).days
        cagr = (nav_df.iloc[-1] / 100) ** (365.25 / days) - 1
        nav_log_ret = np.log(nav_df / nav_df.shift(1)).dropna()
        vol_real = nav_log_ret.std() * np.sqrt(freq)
        dd = (nav_df / nav_df.cummax() - 1).min()
        
        perf_hist = pd.DataFrame({"CAGR": cagr, "Volatilità": vol_real, "Max Drawdown": dd})
        
        c_kpi, c_plot = st.columns([1, 2])
        with c_kpi:
            st.markdown("#### Realized Results")
            if HAS_MATPLOTLIB:
                st.table(perf_hist.style.format("{:.2%}").background_gradient(cmap="RdYlGn", subset=["CAGR"]))
            else:
                st.table(perf_hist.style.format("{:.2%}"))
            
        with c_plot:
            fig_nav = px.line(nav_df, title="Capital Growth (Base 100)")
            fig_nav = style_chart(fig_nav, "Equity Line Evolution")
            st.plotly_chart(fig_nav, use_container_width=True)
            
        drawdown_df = nav_df / nav_df.cummax() - 1
        with st.expander("📉 Deep Drawdown Analysis", expanded=True):
            fig_dd = px.line(drawdown_df, title="Underwater Plot")
            fig_dd = style_chart(fig_dd, "Drawdown Depth")
            fig_dd.update_traces(fill='tozeroy', line=dict(color='#FF0055')) 
            fig_dd.update_yaxes(tickformat=".1%", range=[None, 0.005]) 
            st.plotly_chart(fig_dd, use_container_width=True)

    with tab4:
        st.markdown("### 🔗 Network Correlation")
        corr_matrix = returns.corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_corr.update_layout(title="Correlation Heatmap", template="plotly_dark", height=700, font=dict(family="JetBrains Mono, monospace"))
        st.plotly_chart(fig_corr, use_container_width=True)
        
        with st.expander("📋 View Raw Data"):
            if HAS_MATPLOTLIB:
                st.dataframe(corr_matrix.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"), height=400, use_container_width=True)
            else:
                st.dataframe(corr_matrix.style.format("{:.2f}"), height=400, use_container_width=True)

else:
    st.info("👋 SYSTEM STANDBY. Initialize by uploading CSV data.")
