from datetime import timedelta
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.covariance import LedoitWolf


FREQUENCY_LABELS = {
    252: "Giornaliera",
    52: "Settimanale",
    12: "Mensile",
}


def detect_frequency(df):
    if len(df) < 3:
        return 12
    dates = pd.DatetimeIndex(df.index).sort_values()
    gaps = dates.to_series().diff().dt.total_seconds().div(86400)
    gaps = gaps[gaps > 0]
    if gaps.empty:
        return 12
    median_days = gaps.median()
    if median_days >= 20:
        return 12
    if median_days >= 3:
        return 52
    return 252


def _interpolate_small_internal_gaps(
    df,
    maximum_internal_missing_share=0.05,
    maximum_consecutive_gap=2,
):
    """Ricostruisce solo piccoli buchi interni, conservando la maschera originale."""
    result = df.copy()
    original_missing = result.isna()
    filled_counts = {}
    for column in result.columns:
        series = result[column].astype(float)
        valid = series.dropna()
        if len(valid) < 2:
            continue
        active = series.loc[valid.index[0] : valid.index[-1]]
        missing_count = int(active.isna().sum())
        if missing_count == 0:
            continue
        if missing_count / len(active) > maximum_internal_missing_share:
            continue
        if (valid <= 0).any():
            interpolated = active.interpolate(
                method="time",
                limit=maximum_consecutive_gap,
                limit_area="inside",
            )
        else:
            interpolated = np.exp(
                np.log(active).interpolate(
                    method="time",
                    limit=maximum_consecutive_gap,
                    limit_area="inside",
                )
            )
        filled = int(active.isna().sum() - interpolated.isna().sum())
        if filled:
            result.loc[active.index, column] = interpolated
            filled_counts[column] = filled
    result.attrs["interpolated_counts"] = filled_counts
    result.attrs["original_missing_positions"] = {
        column: [
            pd.Timestamp(index).isoformat()
            for index in original_missing.index[original_missing[column]]
        ]
        for column in original_missing.columns
        if original_missing[column].any()
    }
    return result


def process_data_raw(file_bytes, filename):
    del filename
    for separator in (";", ",", "\t"):
        try:
            df = pd.read_csv(
                BytesIO(file_bytes),
                sep=separator,
                index_col=0,
                na_values=["undefined", "Undefined", "UNDEFINED"],
            )
            df.index = pd.to_datetime(
                df.index,
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )
            df = df.loc[~df.index.isna()]
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep="last")]
            if df.shape[1] == 0 or len(df) < 3:
                continue
            for column in df.columns:
                values = (
                    df[column]
                    .astype("string")
                    .str.replace("€", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.strip()
                )
                if (
                    values.str.contains(",", na=False).any()
                    and not values.str.contains(r"\.", na=False).any()
                ):
                    values = values.str.replace(",", ".", regex=False)
                df[column] = pd.to_numeric(values, errors="coerce").astype(float)
            df = df.sort_index().dropna(axis=1, how="all").dropna(how="all")
            if df.shape[1] > 0:
                return _interpolate_small_internal_gaps(df), None
        except (ValueError, TypeError, pd.errors.ParserError):
            continue
    return None, "Impossibile interpretare il file CSV."


def normalize_price_frequency(df, frequency):
    """Ordina soltanto i dati: la frequenza serve per annualizzare."""
    del frequency
    result = df.sort_index()
    if not result.index.is_unique:
        result = result[~result.index.duplicated(keep="last")]
    result.attrs.update(df.attrs)
    return result


def _selected_original_missing_mask(prices):
    mask = prices.attrs.get("original_missing_mask")
    if isinstance(mask, pd.DataFrame):
        return mask.reindex(
            index=prices.index,
            columns=prices.columns,
            fill_value=False,
        )
    positions = prices.attrs.get("original_missing_positions", {})
    result = prices.isna()
    for column, dates in positions.items():
        if column not in result.columns:
            continue
        parsed_dates = pd.to_datetime(dates)
        common_dates = result.index.intersection(parsed_dates)
        result.loc[common_dates, column] = True
    return result


def compute_causal_returns(df_subset):
    """Rendimenti utilizzabili nel backtest senza informazione futura.

    Ripristina i missing originali e usa soltanto l'ultimo prezzo già noto.
    Un mercato chiuso produce rendimento zero; il movimento si manifesta alla
    prima quotazione successiva.
    """
    prices = df_subset.astype(float).sort_index()
    original_mask = _selected_original_missing_mask(prices)
    causal_prices = prices.mask(original_mask).ffill()
    returns = causal_prices.pct_change(fill_method=None).dropna(how="any")
    returns = returns.astype(float)
    returns.attrs = {}
    return returns, causal_prices


def _winsorized_means(returns_simple, tail=0.025):
    lower = returns_simple.quantile(tail)
    upper = returns_simple.quantile(1 - tail)
    return returns_simple.clip(lower=lower, upper=upper, axis=1).mean()


def estimate_mu_cov(
    returns_simple,
    prior_strength,
    frequency=12,
    risk_free_rate=0.02,
):
    """Stima robusta: media winsorizzata ridotta verso rendimento risk-free.

    Non viene chiamata Black-Litterman: senza benchmark/capitalizzazioni e
    view esterne sarebbe una denominazione impropria. Il prior nullo
    sull'extra-rendimento riduce l'estrapolazione dei periodi eccezionali.
    """
    values = returns_simple.to_numpy(dtype=float)
    cov = LedoitWolf().fit(values).covariance_
    robust_annual = _winsorized_means(returns_simple) * frequency
    strength = float(np.clip(prior_strength, 0.0, 1.0))
    posterior_annual = (
        risk_free_rate
        + (1 - strength) * (robust_annual - risk_free_rate)
    )
    mu_periodic = (posterior_annual / frequency).astype(float)
    annual_standard_error = (
        returns_simple.std(ddof=1)
        * frequency
        / np.sqrt(max(len(returns_simple), 1))
    )
    diagnostics = {
        "method": "Robust Bayesian shrinkage to zero excess return",
        "robust_sample_annual": robust_annual.to_numpy(dtype=float),
        "posterior_annual": posterior_annual.to_numpy(dtype=float),
        "annual_standard_error": annual_standard_error.to_numpy(dtype=float),
        "prior_strength": strength,
    }
    return mu_periodic, cov, diagnostics


def compute_core_stats(
    df_subset,
    frequency,
    mean_shrinkage,
    risk_free_rate=0.02,
):
    prices = df_subset.astype(float).sort_index()
    if (prices.dropna() <= 0).any().any():
        raise ValueError(
            "Sono presenti prezzi nulli o negativi, incompatibili con il modello."
        )
    individual_returns = prices.pct_change(fill_method=None)
    returns_simple = individual_returns.dropna(how="any").astype(float)
    returns_simple.attrs = {}
    minimum_returns = max(9, len(df_subset.columns) + 1)
    if len(returns_simple) < minimum_returns:
        raise ValueError(
            "Dati insufficienti dopo l'allineamento dei rendimenti: "
            f"servono almeno {minimum_returns} osservazioni comuni, "
            f"ma ne risultano {len(returns_simple)}."
        )
    mu, cov, diagnostics = estimate_mu_cov(
        returns_simple,
        mean_shrinkage,
        frequency,
        risk_free_rate,
    )
    clean_prices = prices.loc[returns_simple.index]
    clean_prices.attrs.update(prices.attrs)
    clean_prices.attrs["mean_diagnostics"] = diagnostics
    return returns_simple, mu, cov, clean_prices


def empirical_cvar(period_returns, confidence=0.95):
    """Expected Shortfall storico con peso frazionario sull'ultima osservazione."""
    values = np.asarray(period_returns, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    losses = np.sort(-values)[::-1]
    tail_mass = (1 - confidence) * len(losses)
    if tail_mass <= 0:
        return float(max(losses[0], 0.0))
    full_count = int(np.floor(tail_mass))
    fractional = tail_mass - full_count
    total = losses[:full_count].sum() if full_count else 0.0
    if fractional > 1e-12 and full_count < len(losses):
        total += fractional * losses[full_count]
    cvar = total / tail_mass
    return float(max(cvar, 0.0))


def calculate_metrics(weights, mu, cov, frequency, risk_free_rate):
    expected_return = float(np.dot(weights, mu) * frequency)
    variance = float(weights @ cov @ weights)
    volatility = np.sqrt(max(variance, 0)) * np.sqrt(frequency)
    sharpe = (
        (expected_return - risk_free_rate) / volatility
        if volatility > 0
        else 0
    )
    asset_volatilities = np.sqrt(np.diag(cov)) * np.sqrt(frequency)
    weighted_volatility = float(np.dot(weights, asset_volatilities))
    diversification_ratio = (
        weighted_volatility / volatility if volatility > 0 else 0
    )
    return expected_return, volatility, sharpe, diversification_ratio


def calculate_historical_risk_metrics(
    weights,
    returns_simple,
    frequency,
    risk_free_rate,
):
    portfolio_returns = returns_simple.to_numpy(dtype=float).dot(weights)
    rf_period = risk_free_rate / frequency
    downside = np.minimum(portfolio_returns - rf_period, 0)
    downside_deviation = float(
        np.sqrt(np.mean(np.square(downside))) * np.sqrt(frequency)
    )
    annual_return = float(np.mean(portfolio_returns) * frequency)
    sortino = (
        (annual_return - risk_free_rate) / downside_deviation
        if downside_deviation > 1e-12
        else np.nan
    )
    var_95 = float(max(-np.quantile(portfolio_returns, 0.05), 0.0))
    cvar_95 = empirical_cvar(portfolio_returns, 0.95)
    nav = pd.Series(
        np.cumprod(1 + portfolio_returns),
        index=returns_simple.index,
    )
    drawdown = nav / nav.cummax() - 1
    negative_drawdowns = drawdown[drawdown < 0]
    average_drawdown = (
        float(negative_drawdowns.mean())
        if len(negative_drawdowns)
        else 0.0
    )
    stress_window = {252: 21, 52: 4, 12: 3}[frequency]
    rolling = (
        (1 + pd.Series(portfolio_returns))
        .rolling(stress_window)
        .apply(np.prod, raw=True)
        - 1
    )
    worst_stress = (
        float(rolling.min())
        if rolling.notna().any()
        else float(np.min(portfolio_returns))
    )
    return {
        "Downside Deviation": downside_deviation,
        "Sortino": float(sortino),
        "VaR 95% (periodo)": var_95,
        "CVaR 95% (periodo)": cvar_95,
        "Drawdown medio": average_drawdown,
        "Max Drawdown": float(drawdown.min()),
        "Stress storico": worst_stress,
        "Peggior periodo": float(np.min(portfolio_returns)),
    }


def calculate_concentration_metrics(weights, correlation_matrix):
    weights = np.asarray(weights, dtype=float)
    hhi = float(np.square(weights).sum())
    effective_assets = 1 / hhi if hhi > 0 else 0.0
    top_three = float(np.sort(weights)[-3:].sum())
    corr = np.asarray(correlation_matrix, dtype=float)
    pair_weight = np.outer(weights, weights)
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)
    denominator = pair_weight[mask].sum()
    weighted_correlation = (
        float((pair_weight[mask] * corr[mask]).sum() / denominator)
        if denominator > 0
        else np.nan
    )
    overlap_score = float(
        (pair_weight[mask] * np.clip(corr[mask], 0, 1)).sum()
    )
    return {
        "HHI": hhi,
        "Numero effettivo titoli": effective_assets,
        "Peso primi 3": top_three,
        "Correlazione ponderata": weighted_correlation,
        "Sovrapposizione statistica": overlap_score,
    }


def find_high_correlation_pairs(returns_simple, threshold=0.70):
    corr = returns_simple.corr()
    rows = []
    for i, left in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            value = float(corr.iloc[i, j])
            if value >= threshold:
                rows.append((left, corr.columns[j], value))
    return pd.DataFrame(
        rows,
        columns=["Strumento A", "Strumento B", "Correlazione"],
    ).sort_values("Correlazione", ascending=False, ignore_index=True)


def calculate_crisis_correlation(returns_simple, tail_share=0.20):
    market_proxy = returns_simple.mean(axis=1)
    threshold = market_proxy.quantile(tail_share)
    crisis_returns = returns_simple.loc[market_proxy <= threshold]
    matrix = crisis_returns.corr()
    values = matrix.to_numpy(dtype=float)
    mask = ~np.eye(len(matrix), dtype=bool)
    average = float(np.nanmean(values[mask])) if mask.any() else np.nan
    return matrix, average, len(crisis_returns), float(threshold)


def validate_weight_bounds(min_weights, max_weights):
    lower = np.asarray(min_weights, dtype=float)
    upper = np.asarray(max_weights, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("I limiti minimi e massimi non hanno la stessa dimensione.")
    if np.any(lower < 0) or np.any(upper > 1) or np.any(lower > upper):
        raise ValueError("Uno o più limiti per asset non sono validi.")
    if lower.sum() > 1 + 1e-9:
        raise ValueError(
            f"La somma dei pesi minimi è {lower.sum():.1%}, superiore al 100%."
        )
    if upper.sum() < 1 - 1e-9:
        raise ValueError(
            f"La somma dei pesi massimi è {upper.sum():.1%}, inferiore al 100%."
        )
    return lower, upper


def _feasible_start(lower, upper, preferred_order=None):
    weights = lower.copy()
    remaining = 1 - weights.sum()
    if preferred_order is None:
        capacity = upper - lower
        if capacity.sum() > 0:
            weights += remaining * capacity / capacity.sum()
        return weights
    for index in preferred_order:
        addition = min(remaining, upper[index] - weights[index])
        weights[index] += addition
        remaining -= addition
        if remaining <= 1e-12:
            break
    return weights


def _portfolio_volatility(weights, cov, frequency):
    return float(np.sqrt(max(weights @ cov @ weights, 0)) * np.sqrt(frequency))


def _solve_min_cvar_lp(
    returns_history,
    lower,
    upper,
    expected_periodic=None,
    minimum_expected_periodic=None,
    confidence=0.95,
):
    returns = np.asarray(returns_history, dtype=float)
    observation_count, asset_count = returns.shape
    tail_probability = 1 - confidence
    objective = np.r_[
        np.zeros(asset_count),
        1.0,
        np.full(observation_count, 1 / (tail_probability * observation_count)),
    ]
    inequality = np.zeros(
        (observation_count, asset_count + 1 + observation_count)
    )
    inequality[:, :asset_count] = -returns
    inequality[:, asset_count] = -1
    inequality[:, asset_count + 1 :] = -np.eye(observation_count)
    inequality_rhs = np.zeros(observation_count)
    if minimum_expected_periodic is not None:
        expected_row = np.zeros(asset_count + 1 + observation_count)
        expected_row[:asset_count] = -np.asarray(expected_periodic, dtype=float)
        inequality = np.vstack([inequality, expected_row])
        inequality_rhs = np.r_[inequality_rhs, -minimum_expected_periodic]
    equality = np.zeros((1, asset_count + 1 + observation_count))
    equality[0, :asset_count] = 1
    bounds = (
        list(zip(lower, upper))
        + [(None, None)]
        + [(0, None)] * observation_count
    )
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=[1],
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None
    return result.x[:asset_count]


def _project_to_volatility_corridor(
    target,
    lower,
    upper,
    cov,
    frequency,
    min_vol,
    max_vol,
):
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if max_vol is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: max_vol
                - _portfolio_volatility(w, cov, frequency),
            }
        )
    if min_vol is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: _portfolio_volatility(w, cov, frequency)
                - min_vol,
            }
        )
    starts = [
        target,
        _feasible_start(lower, upper),
        _feasible_start(lower, upper, np.argsort(np.diag(cov))),
        _feasible_start(lower, upper, np.argsort(np.diag(cov))[::-1]),
    ]
    successful = []
    for start in starts:
        result = minimize(
            lambda w: np.square(w - target).sum() + 0.02 * np.square(w).sum(),
            start,
            method="SLSQP",
            bounds=tuple(zip(lower, upper)),
            constraints=constraints,
            tol=1e-10,
            options={"maxiter": 3000},
        )
        if result.success:
            successful.append(result)
    if not successful:
        return None
    return min(successful, key=lambda result: result.fun).x


def _within_corridor(weights, cov, frequency, min_vol, max_vol, tolerance=1e-6):
    volatility = _portfolio_volatility(weights, cov, frequency)
    if min_vol is not None and volatility < min_vol - tolerance:
        return False
    if max_vol is not None and volatility > max_vol + tolerance:
        return False
    return True


def _aggressive_frontier_weights(
    mu,
    cov,
    returns_history,
    lower,
    upper,
    frequency,
    min_vol,
    max_vol,
):
    mu = np.asarray(mu, dtype=float)
    minimum_return = float(mu @ _feasible_start(lower, upper, np.argsort(mu)))
    maximum_return = float(mu @ _feasible_start(lower, upper, np.argsort(mu)[::-1]))
    targets = np.linspace(minimum_return, maximum_return, 18)
    candidates = []
    for target in targets:
        weights = _solve_min_cvar_lp(
            returns_history,
            lower,
            upper,
            expected_periodic=mu,
            minimum_expected_periodic=target,
        )
        if weights is None:
            continue
        if not _within_corridor(
            weights, cov, frequency, min_vol, max_vol
        ):
            weights = _project_to_volatility_corridor(
                weights,
                lower,
                upper,
                cov,
                frequency,
                min_vol,
                max_vol,
            )
        if weights is None or not _within_corridor(
            weights, cov, frequency, min_vol, max_vol
        ):
            continue
        expected_return = float(weights @ mu * frequency)
        cvar = empirical_cvar(
            np.asarray(returns_history, dtype=float) @ weights
        ) * np.sqrt(frequency)
        concentration = float(np.square(weights).sum())
        candidates.append((weights, expected_return, cvar, concentration))
    if not candidates:
        return None
    returns = np.array([item[1] for item in candidates])
    cvars = np.array([item[2] for item in candidates])
    concentrations = np.array([item[3] for item in candidates])
    return_scale = max(np.ptp(returns), 1e-12)
    cvar_scale = max(np.ptp(cvars), 1e-12)
    concentration_scale = max(np.ptp(concentrations), 1e-12)
    # Punto di ginocchio della frontiera: rendimento marginale meno aumento
    # del rischio di coda, con una penalità contenuta per la concentrazione.
    score = (
        (returns - returns.min()) / return_scale
        - (cvars - cvars.min()) / cvar_scale
        - 0.10 * (concentrations - concentrations.min()) / concentration_scale
    )
    return candidates[int(np.argmax(score))][0]


def optimize_basket(
    mu,
    cov,
    optimization_type,
    min_weights,
    max_weights,
    risk_free_rate,
    max_vol=None,
    min_vol=None,
    frequency=12,
    returns_history=None,
):
    lower, upper = validate_weight_bounds(min_weights, max_weights)
    if min_vol is not None and max_vol is not None and min_vol > max_vol:
        raise ValueError("La volatilità minima supera la volatilità massima.")
    returns_array = (
        None
        if returns_history is None
        else np.asarray(returns_history, dtype=float)
    )

    if optimization_type == "min_cvar":
        if returns_array is None:
            raise ValueError("Lo storico è necessario per il CVaR.")
        weights = _solve_min_cvar_lp(returns_array, lower, upper)
        if weights is not None and not _within_corridor(
            weights, cov, frequency, min_vol, max_vol
        ):
            weights = _project_to_volatility_corridor(
                weights,
                lower,
                upper,
                cov,
                frequency,
                min_vol,
                max_vol,
            )
    elif optimization_type == "cvar_frontier":
        if returns_array is None:
            raise ValueError("Lo storico è necessario per la frontiera CVaR.")
        weights = _aggressive_frontier_weights(
            mu,
            cov,
            returns_array,
            lower,
            upper,
            frequency,
            min_vol,
            max_vol,
        )
    else:
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        if max_vol is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: max_vol
                    - _portfolio_volatility(w, cov, frequency),
                }
            )
        if min_vol is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: _portfolio_volatility(
                        w, cov, frequency
                    )
                    - min_vol,
                }
            )

        def expected_return(w):
            return float(w @ mu * frequency)

        def objective(w):
            concentration_penalty = 0.03 * np.square(w).sum()
            if optimization_type == "max_sharpe":
                volatility = _portfolio_volatility(w, cov, frequency)
                return -(
                    (expected_return(w) - risk_free_rate)
                    / max(volatility, 1e-12)
                ) + concentration_penalty
            if optimization_type == "min_vol":
                return float(w @ cov @ w) + concentration_penalty
            if optimization_type == "max_return":
                return -expected_return(w) + concentration_penalty
            raise ValueError(f"Obiettivo sconosciuto: {optimization_type}")

        starts = [
            _feasible_start(lower, upper),
            _feasible_start(lower, upper, np.argsort(mu)[::-1]),
            _feasible_start(lower, upper, np.argsort(np.diag(cov))),
            _feasible_start(lower, upper, np.argsort(np.diag(cov))[::-1]),
        ]
        results = []
        for start in starts:
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=tuple(zip(lower, upper)),
                constraints=constraints,
                tol=1e-9,
                options={"maxiter": 3000},
            )
            if result.success and _within_corridor(
                result.x, cov, frequency, min_vol, max_vol
            ):
                results.append(result)
        weights = (
            min(results, key=lambda result: result.fun).x
            if results
            else None
        )

    if weights is None or not _within_corridor(
        weights, cov, frequency, min_vol, max_vol
    ):
        corridor = (
            f"{min_vol if min_vol is not None else 0:.1%}–"
            f"{max_vol if max_vol is not None else 100:.1%}"
        )
        raise ValueError(
            "Nessun portafoglio soddisfa tutti i vincoli. "
            f"Corridoio richiesto: {corridor}."
        )
    if (
        abs(weights.sum() - 1) > 1e-6
        or np.any(weights < lower - 1e-7)
        or np.any(weights > upper + 1e-7)
    ):
        raise ValueError("La soluzione numerica viola i limiti sui pesi.")
    return weights


def _moving_block_sample(values, block_size, rng):
    observation_count = len(values)
    starts = rng.integers(
        0,
        observation_count - block_size + 1,
        size=int(np.ceil(observation_count / block_size)),
    )
    indices = np.concatenate(
        [np.arange(start, start + block_size) for start in starts]
    )[:observation_count]
    return values[indices]


def bootstrap_portfolio_intervals(
    allocations,
    returns_simple,
    frequency,
    risk_free_rate,
    simulations=500,
    random_seed=20260723,
):
    values = returns_simple.to_numpy(dtype=float)
    block_size = {252: 10, 52: 4, 12: 2}[frequency]
    block_size = min(block_size, max(len(values) // 3, 1))
    rng = np.random.default_rng(random_seed)
    records = []
    for simulation in range(simulations):
        sample = _moving_block_sample(values, block_size, rng)
        for name, weights in allocations.items():
            portfolio_returns = sample @ weights
            annual_return = float(portfolio_returns.mean() * frequency)
            volatility = float(
                portfolio_returns.std(ddof=1) * np.sqrt(frequency)
            )
            sharpe = (
                (annual_return - risk_free_rate) / volatility
                if volatility > 1e-12
                else np.nan
            )
            nav = np.cumprod(1 + portfolio_returns)
            max_drawdown = float(np.min(nav / np.maximum.accumulate(nav) - 1))
            records.append(
                {
                    "Simulazione": simulation,
                    "Linea": name,
                    "Rendimento": annual_return,
                    "Volatilità": volatility,
                    "Sharpe": sharpe,
                    "CVaR 95%": empirical_cvar(portfolio_returns),
                    "Max Drawdown": max_drawdown,
                }
            )
    raw = pd.DataFrame(records)
    rows = []
    for name, group in raw.groupby("Linea", sort=False):
        for metric in [
            "Rendimento",
            "Volatilità",
            "Sharpe",
            "CVaR 95%",
            "Max Drawdown",
        ]:
            low, median, high = group[metric].quantile([0.05, 0.50, 0.95])
            rows.append(
                {
                    "Linea": name,
                    "Metrica": metric,
                    "5%": low,
                    "Mediana": median,
                    "95%": high,
                }
            )
    return pd.DataFrame(rows), raw


def _apply_block_with_drift(
    test_block,
    target_weights,
    starting_weights,
    cost_rate,
):
    weights = target_weights.copy()
    turnover = (
        1.0
        if starting_weights is None
        else 0.5 * float(np.abs(target_weights - starting_weights).sum())
    )
    net_results = []
    gross_results = []
    applied_cost = cost_rate * turnover
    for period_index, asset_returns in enumerate(
        test_block.to_numpy(dtype=float)
    ):
        gross_return = float(weights @ asset_returns)
        net_return = gross_return - (applied_cost if period_index == 0 else 0)
        gross_results.append(gross_return)
        net_results.append(net_return)
        gross_values = weights * (1 + asset_returns)
        gross_total = float(gross_values.sum())
        if gross_total > 0:
            weights = gross_values / gross_total
    return net_results, gross_results, weights, turnover, applied_cost


def run_walk_forward(
    returns_simple,
    strategies_config,
    min_weights,
    max_weights,
    risk_free_rate,
    max_limits,
    min_limits,
    frequency,
    mean_shrinkage,
    transaction_cost_bps,
):
    observation_count, asset_count = returns_simple.shape
    minimum_training = max(12, 2 * asset_count)
    test_start = max(minimum_training, int(observation_count * 0.70))
    if observation_count - test_start < 5:
        raise ValueError(
            "Storico insufficiente per una validazione fuori campione."
        )
    rebalance_step = {252: 21, 52: 4, 12: 1}[frequency]
    strategy_returns = {name: [] for name, _ in strategies_config}
    gross_strategy_returns = {name: [] for name, _ in strategies_config}
    strategy_returns["Equal Weight"] = []
    gross_strategy_returns["Equal Weight"] = []
    current_weights = {name: None for name, _ in strategies_config}
    current_weights["Equal Weight"] = None
    diagnostics_rows = []
    test_dates = []
    cost_rate = transaction_cost_bps / 10000

    for start in range(test_start, observation_count, rebalance_step):
        end = min(start + rebalance_step, observation_count)
        training_returns = returns_simple.iloc[:start]
        test_block = returns_simple.iloc[start:end]
        mu, cov, _ = estimate_mu_cov(
            training_returns,
            mean_shrinkage,
            frequency,
            risk_free_rate,
        )
        test_dates.extend(test_block.index.tolist())

        for name, method in strategies_config:
            target_weights = optimize_basket(
                mu.values,
                cov,
                method,
                min_weights,
                max_weights,
                risk_free_rate,
                max_vol=max_limits[name],
                min_vol=min_limits[name],
                frequency=frequency,
                returns_history=training_returns,
            )
            net, gross, ending_weights, turnover, cost = (
                _apply_block_with_drift(
                    test_block,
                    target_weights,
                    current_weights[name],
                    cost_rate,
                )
            )
            strategy_returns[name].extend(net)
            gross_strategy_returns[name].extend(gross)
            current_weights[name] = ending_weights
            diagnostics_rows.append(
                {
                    "Data": test_block.index[0],
                    "Linea": name,
                    "Volatilità prevista": _portfolio_volatility(
                        target_weights, cov, frequency
                    ),
                    "Rendimento previsto": float(
                        target_weights @ mu.values * frequency
                    ),
                    "CVaR previsto": empirical_cvar(
                        training_returns.to_numpy() @ target_weights
                    ),
                    "Turnover": turnover,
                    "Costo": cost,
                }
            )

        equal_weights = np.full(asset_count, 1 / asset_count)
        net, gross, ending_weights, turnover, cost = _apply_block_with_drift(
            test_block,
            equal_weights,
            current_weights["Equal Weight"],
            cost_rate,
        )
        strategy_returns["Equal Weight"].extend(net)
        gross_strategy_returns["Equal Weight"].extend(gross)
        current_weights["Equal Weight"] = ending_weights
        training_cov = LedoitWolf().fit(
            training_returns.to_numpy(dtype=float)
        ).covariance_
        diagnostics_rows.append(
            {
                "Data": test_block.index[0],
                "Linea": "Equal Weight",
                "Volatilità prevista": _portfolio_volatility(
                    equal_weights, training_cov, frequency
                ),
                "Rendimento previsto": float(
                    training_returns.mean().to_numpy()
                    @ equal_weights
                    * frequency
                ),
                "CVaR previsto": empirical_cvar(
                    training_returns.to_numpy() @ equal_weights
                ),
                "Turnover": turnover,
                "Costo": cost,
            }
        )

    index = pd.DatetimeIndex(test_dates)
    period_returns = pd.DataFrame(strategy_returns, index=index, dtype=float)
    gross_period_returns = pd.DataFrame(
        gross_strategy_returns, index=index, dtype=float
    )
    nav = (1 + period_returns).cumprod() * 100
    gross_nav = (1 + gross_period_returns).cumprod() * 100
    start_date = nav.index[0] - timedelta(days=1)
    start_frame = pd.DataFrame(100.0, index=[start_date], columns=nav.columns)
    nav = pd.concat([start_frame, nav])
    gross_nav = pd.concat([start_frame, gross_nav])

    diagnostics = pd.DataFrame(diagnostics_rows)
    summary_rows = []
    for name in nav.columns:
        realized = period_returns[name]
        realized_volatility = float(realized.std(ddof=1) * np.sqrt(frequency))
        annual_return = float(realized.mean() * frequency)
        realized_sharpe = (
            (annual_return - risk_free_rate) / realized_volatility
            if realized_volatility > 1e-12
            else np.nan
        )
        strategy_diag = diagnostics.loc[diagnostics["Linea"] == name]
        min_limit = min_limits.get(name)
        max_limit = max_limits.get(name)
        summary_rows.append(
            {
                "Linea": name,
                "Volatilità prevista media": strategy_diag[
                    "Volatilità prevista"
                ].mean(),
                "Volatilità realizzata": realized_volatility,
                "Sharpe realizzato": realized_sharpe,
                "CVaR realizzato": empirical_cvar(realized),
                "Corridoio realizzato rispettato": (
                    (min_limit is None or realized_volatility >= min_limit)
                    and (max_limit is None or realized_volatility <= max_limit)
                )
                if name in min_limits
                else np.nan,
                "Turnover cumulato": strategy_diag["Turnover"].sum(),
                "Costi cumulati": strategy_diag["Costo"].sum(),
                "Drag costi su NAV": float(
                    gross_nav[name].iloc[-1] - nav[name].iloc[-1]
                ),
                "Ribilanciamenti": len(strategy_diag),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("Linea")
    return nav, test_start, summary, gross_nav, diagnostics
