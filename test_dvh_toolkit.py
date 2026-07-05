import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from generate_synthetic_data import generate
from dvh_toolkit import (to_4d_array, friedman_by_organ,
                          apply_multiple_comparison_correction,
                          pairwise_wilcoxon, variance_decomposition)

df = generate(n_patients=10, seed=1)


def test_4d_array_shape():
    arr, patients, organs, techniques = to_4d_array(df, "Dmean_Gy")
    assert arr.shape == (len(patients), len(organs), len(techniques))
    assert not np.isnan(arr).any()


def test_friedman_runs_for_every_organ():
    result = friedman_by_organ(df, "Dmean_Gy")
    assert len(result) == df["Organ"].nunique()
    assert (result["p_value"] >= 0).all() and (result["p_value"] <= 1).all()


def test_bonferroni_is_more_conservative_than_bh():
    pvals = [0.001, 0.01, 0.02, 0.04, 0.20]
    bonf = apply_multiple_comparison_correction(pvals, "bonferroni")
    bh = apply_multiple_comparison_correction(pvals, "bh")
    # Bonferroni should never be less conservative (smaller) than BH
    assert (bonf >= bh - 1e-9).all()


def test_pairwise_wilcoxon_correction_applied():
    result = pairwise_wilcoxon(df, "Dmean_Gy", "L_Parotid", correction="bh")
    n_pairs = len(df["Technique"].unique())
    expected_pairs = n_pairs * (n_pairs - 1) // 2
    assert len(result) == expected_pairs
    assert (result["p_bh"] >= result["p_raw"] - 1e-9).all()  # correction only inflates p


def test_variance_decomposition_sums_to_100_pct():
    result = variance_decomposition(df, "Dmean_Gy", "L_Parotid")
    total_pct = result["pct_variance_explained"].sum()
    assert abs(total_pct - 100.0) < 1e-6


if __name__ == "__main__":
    test_4d_array_shape()
    test_friedman_runs_for_every_organ()
    test_bonferroni_is_more_conservative_than_bh()
    test_pairwise_wilcoxon_correction_applied()
    test_variance_decomposition_sums_to_100_pct()
    print("All tests passed.")
