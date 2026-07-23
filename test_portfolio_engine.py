import unittest

import numpy as np
import pandas as pd

from portfolio_engine import (
    compute_causal_returns,
    empirical_cvar,
    estimate_mu_cov,
    optimize_basket,
    run_walk_forward,
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
        self.assertIn("Turnover cumulato", summary.columns)
        self.assertGreater(len(diagnostics), 0)
        self.assertTrue((gross_nav.iloc[-1] >= nav.iloc[-1]).all())


if __name__ == "__main__":
    unittest.main()
