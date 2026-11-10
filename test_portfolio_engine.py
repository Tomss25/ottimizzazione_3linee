import unittest

import numpy as np
import pandas as pd

from portfolio_engine import (
    VOLATILITY_LINE_GAP,
    compute_causal_returns,
    empirical_cvar,
    estimate_mu_cov,
    minimum_feasible_volatility,
    optimize_basket,
    optimize_ordered_lines,
    process_data_raw,
    run_walk_forward,
    summarize_validation_windows,
)


class PortfolioEngineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=260)
        common = rng.normal(0.0003, 0.008, size=(260, 1))
        noise = rng.normal(0, 0.006, size=(260, 4))
        self.returns = pd.DataFrame(
            common + noise,
            index=dates,
            columns=["A", "B", "C", "D"],
        )

    def test_csv_dates_keep_iso_order_and_accept_day_first(self):
        csv = (
            "Data;A;B\n"
            "2024-03-27;100,0;200,0\n"
            "01/04/2024;101,0;201,0\n"
            "2024-04-02;102,0;202,0\n"
        )
        prices, error = process_data_raw(csv.encode(), "prices.csv")
        self.assertIsNone(error)
        self.assertEqual(
            list(prices.index),
            list(pd.to_datetime(["2024-03-27", "2024-04-01", "2024-04-02"])),
        )
        self.assertAlmostEqual(prices.loc["2024-04-01", "A"], 101.0)

    def test_infeasible_maximum_reports_attainable_minimum(self):
        cov = np.diag([0.04, 0.09])
        lower = np.zeros(2)
        upper = np.array([0.60, 1.0])
        minimum = minimum_feasible_volatility(cov, lower, upper, 1)
        self.assertAlmostEqual(minimum, np.sqrt(0.0288), places=7)
        with self.assertRaisesRegex(ValueError, "16.97%.*15.0%"):
            optimize_basket(
                np.zeros(2), cov, "gmv", lower, upper, 0.02,
                max_vol=0.15, min_vol=0, frequency=1,
            )

    def test_causal_returns_do_not_use_future_price(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 120.0, 125.0], "B": [100.0, 105.0, 120.0, 123.0]},
            index=dates,
        )
        missing = pd.DataFrame(False, index=dates, columns=prices.columns)
        missing.loc[dates[1], "A"] = True
        prices.attrs["original_missing_mask"] = missing
        returns, _ = compute_causal_returns(prices)
        self.assertAlmostEqual(returns.loc[dates[1], "A"], 0.0)
        self.assertAlmostEqual(returns.loc[dates[2], "A"], 0.20)

    def test_full_prior_strength_removes_historical_excess_return(self):
        mu, _, diagnostics = estimate_mu_cov(
            self.returns,
            prior_strength=1.0,
            frequency=252,
            risk_free_rate=0.02,
        )
        np.testing.assert_allclose(mu.to_numpy() * 252, 0.02, atol=1e-12)
        self.assertEqual(
            diagnostics["method"],
            "Robust Bayesian shrinkage to zero excess return",
        )

    def test_empirical_cvar_uses_worst_tail(self):
        period_returns = np.array([-0.10, -0.05, 0.01, 0.02, 0.03])
        self.assertAlmostEqual(empirical_cvar(period_returns, 0.80), 0.10)

    def test_optimizer_respects_custom_volatility_corridor(self):
        mu, cov, _ = estimate_mu_cov(
            self.returns,
            prior_strength=0.7,
            frequency=252,
            risk_free_rate=0.02,
        )
        lower = np.zeros(4)
        upper = np.full(4, 0.70)
        weights = optimize_basket(
            mu.to_numpy(),
            cov,
            "max_sharpe",
            lower,
            upper,
            0.02,
            min_vol=0.10,
            max_vol=0.30,
            frequency=252,
            returns_history=self.returns,
        )
        volatility = np.sqrt(weights @ cov @ weights) * np.sqrt(252)
        self.assertGreaterEqual(volatility, 0.10 - 1e-6)
        self.assertLessEqual(volatility, 0.30 + 1e-6)

    def test_ordered_lines_follow_calculated_previous_volatility(self):
        mu, cov, _ = estimate_mu_cov(
            self.returns, prior_strength=0.7, frequency=252, risk_free_rate=0.02
        )
        strategies = [
            ("Conservative", "min_cvar"),
            ("Balanced", "max_sharpe"),
        ]
        maximum = {"Conservative": 0.20, "Balanced": 0.30}
        lower = np.zeros(4)
        upper = np.full(4, 0.70)
        allocations, minimum = optimize_ordered_lines(
            mu.to_numpy(), cov, strategies, lower, upper, 0.02,
            maximum, 0.0, 252, self.returns,
        )
        volatility = {
            name: np.sqrt(weights @ cov @ weights) * np.sqrt(252)
            for name, weights in allocations.items()
        }
        self.assertAlmostEqual(
            minimum["Balanced"],
            volatility["Conservative"] + VOLATILITY_LINE_GAP,
        )
        self.assertGreater(volatility["Balanced"], volatility["Conservative"])

        impossible_maximum = maximum.copy()
        impossible_maximum["Balanced"] = (
            volatility["Conservative"] + VOLATILITY_LINE_GAP / 2
        )
        with self.assertRaisesRegex(ValueError, "minimo dinamico"):
            optimize_ordered_lines(
                mu.to_numpy(), cov, strategies, lower, upper, 0.02,
                impossible_maximum, 0.0, 252, self.returns,
            )

    def test_walk_forward_recalculates_ordered_minimums_each_rebalance(self):
        strategies = [
            ("Conservative", "min_cvar"),
            ("Balanced", "max_sharpe"),
            ("Aggressive", "cvar_frontier"),
        ]
        maximum = {"Conservative": 0.20, "Balanced": 0.30, "Aggressive": 0.40}
        _, _, summary, _, diagnostics = run_walk_forward(
            self.returns,
            strategies,
            np.zeros(4),
            np.full(4, 0.70),
            0.02,
            maximum,
            {"Conservative": 0.0},
            252,
            0.70,
            10,
            ordered_strategy_names=tuple(name for name, _ in strategies),
        )
        for _, block in diagnostics.groupby("Data"):
            lines = block.set_index("Linea")
            conservative = lines.loc["Conservative", "Volatilità prevista"]
            balanced = lines.loc["Balanced", "Volatilità prevista"]
            aggressive = lines.loc["Aggressive", "Volatilità prevista"]
            self.assertGreater(balanced, conservative)
            self.assertGreater(aggressive, balanced)
            self.assertAlmostEqual(
                lines.loc["Balanced", "Volatilità minima"],
                conservative + VOLATILITY_LINE_GAP,
            )
            self.assertAlmostEqual(
                lines.loc["Aggressive", "Volatilità minima"],
                balanced + VOLATILITY_LINE_GAP,
            )
        self.assertTrue(
            summary.loc[["Balanced", "Aggressive"], "Corridoio realizzato rispettato"]
            .isna().all()
        )

    def test_hrp_and_gmv_benchmarks_respect_asset_bounds(self):
        mu, cov, _ = estimate_mu_cov(
            self.returns,
            prior_strength=0.7,
            frequency=252,
            risk_free_rate=0.02,
        )
        lower = np.array([0.05, 0.00, 0.00, 0.00])
        upper = np.array([0.30, 0.45, 0.45, 0.45])
        for method in ("hrp", "gmv"):
            with self.subTest(method=method):
                weights = optimize_basket(
                    mu.to_numpy(),
                    cov,
                    method,
                    lower,
                    upper,
                    0.02,
                    frequency=252,
                    returns_history=self.returns,
                )
                self.assertAlmostEqual(weights.sum(), 1.0, places=7)
                self.assertTrue(np.all(weights >= lower - 1e-7))
                self.assertTrue(np.all(weights <= upper + 1e-7))

    def test_walk_forward_reports_costs_and_realized_risk(self):
        lower = np.zeros(4)
        upper = np.full(4, 0.70)
        strategies = [
            ("Conservative", "min_cvar"),
            ("Balanced", "max_sharpe"),
            ("Aggressive", "cvar_frontier"),
        ]
        minimum = {"Conservative": 0.0, "Balanced": 0.08, "Aggressive": 0.10}
        maximum = {"Conservative": 0.20, "Balanced": 0.30, "Aggressive": 0.40}
        nav, _, summary, gross_nav, diagnostics = run_walk_forward(
            self.returns,
            strategies,
            lower,
            upper,
            0.02,
            maximum,
            minimum,
            252,
            0.70,
            10,
        )
        self.assertIn("Volatilità realizzata", summary.columns)
        self.assertIn("Sharpe realizzato", summary.columns)
        self.assertIn("Turnover cumulato", summary.columns)
        self.assertGreater(len(diagnostics), 0)
        self.assertTrue((gross_nav.iloc[-1] >= nav.iloc[-1]).all())

    def test_walk_forward_includes_oos_challenger_benchmarks(self):
        lower = np.zeros(4)
        upper = np.array([0.30, 0.45, 0.45, 0.45])
        strategies = [
            ("Conservative", "min_cvar"),
            ("Balanced", "max_sharpe"),
            ("Aggressive", "cvar_frontier"),
            ("GMV Ledoit-Wolf", "gmv"),
            ("HRP vincolato", "hrp"),
        ]
        minimum = {"Conservative": 0.0, "Balanced": 0.08, "Aggressive": 0.10}
        maximum = {"Conservative": 0.20, "Balanced": 0.30, "Aggressive": 0.40}
        nav, _, summary, _, diagnostics = run_walk_forward(
            self.returns,
            strategies,
            lower,
            upper,
            0.02,
            maximum,
            minimum,
            252,
            0.70,
            10,
        )
        for name in ("GMV Ledoit-Wolf", "HRP vincolato", "Equal Weight"):
            self.assertIn(name, nav.columns)
            self.assertIn(name, summary.index)
        benchmark_corridor = summary.loc[
            ["GMV Ledoit-Wolf", "HRP vincolato", "Equal Weight"],
            "Corridoio realizzato rispettato",
        ]
        self.assertTrue(benchmark_corridor.isna().all())
        self.assertTrue(
            set(diagnostics["Linea"])
            >= {"GMV Ledoit-Wolf", "HRP vincolato", "Equal Weight"}
        )

    def test_validation_windows_are_chronological_and_non_overlapping(self):
        dates = pd.bdate_range("2024-01-01", periods=31)
        returns = np.r_[np.full(10, 0.01), np.full(10, -0.01), np.full(10, 0.02)]
        nav = pd.DataFrame(
            {"Conservative": 100 * np.r_[1.0, np.cumprod(1 + returns)]},
            index=dates,
        )
        windows = summarize_validation_windows(nav, 252, 0.02)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows["Osservazioni"].tolist(), [10, 10, 10])
        self.assertTrue((windows["Dal"].iloc[1:].to_numpy() > windows["Al"].iloc[:-1].to_numpy()).all())
        self.assertLess(windows.loc[1, "Rendimento annuo"], 0)

    def test_walk_forward_rejects_invalid_training_fraction(self):
        with self.assertRaisesRegex(ValueError, "quota iniziale"):
            run_walk_forward(
                self.returns,
                [("Conservative", "min_cvar")],
                np.zeros(4),
                np.full(4, 0.70),
                0.02,
                {"Conservative": 0.20},
                {"Conservative": 0.0},
                252,
                0.70,
                10,
                test_start_fraction=1.0,
            )


if __name__ == "__main__":
    unittest.main()
