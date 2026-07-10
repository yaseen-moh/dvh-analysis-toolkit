"""
DVH Analysis Toolkit
----------------------
A general-purpose toolkit for comparing radiotherapy planning techniques
across patients using dose-volume-histogram (DVH) metrics -- the same
class of analysis used to compare volumetric arc therapy / hippocampal-
avoidance whole-brain radiation planning approaches: build a tidy
long-format dataset, reshape it into a (patients x organs x metrics x
techniques) array, run Friedman tests for each organ/metric to check for
a technique effect, decompose variance between patient/organ/technique
sources, and apply multiple-comparison corrections (Bonferroni and
Benjamini-Hochberg) to any pairwise post-hoc tests.

Built on base R equivalents originally (friedman.test, pairwise.wilcox.test,
aov) after package-installation issues with rstatix/lme4 in the original
clinical pipeline; this module is the general, portable Python counterpart
used for the open-source/portfolio version of that workflow, running here
entirely on synthetic data (see generate_synthetic_data.py).

Author: Yaseen Mohamed
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


def to_4d_array(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, list, list, list]:
    """
    Reshapes tidy long-format data into a named 3D array:
    patients x organs x techniques, for a single metric.

    Returns (array, patient_ids, organs, techniques)
    """
    patients = sorted(df["PatientID"].unique())
    organs = sorted(df["Organ"].unique())
    techniques = sorted(df["Technique"].unique())

    arr = np.full((len(patients), len(organs), len(techniques)), np.nan)
    p_idx = {p: i for i, p in enumerate(patients)}
    o_idx = {o: i for i, o in enumerate(organs)}
    t_idx = {t: i for i, t in enumerate(techniques)}

    for _, row in df.iterrows():
        arr[p_idx[row["PatientID"]], o_idx[row["Organ"]], t_idx[row["Technique"]]] = row[metric]

    return arr, patients, organs, techniques


def friedman_by_organ(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Runs a Friedman test (non-parametric repeated-measures test across
    techniques, blocked by patient) separately for each organ, for the
    given metric. This is the right test here because the same patients
    are re-measured under every technique (a repeated-measures design)
    and DVH metrics are not reliably normally distributed.
    """
    arr, patients, organs, techniques = to_4d_array(df, metric)
    results = []
    for oi, organ in enumerate(organs):
        organ_data = arr[:, oi, :]  # patients x techniques
        if np.isnan(organ_data).any():
            continue
        stat, p = stats.friedmanchisquare(*[organ_data[:, ti] for ti in range(len(techniques))])
        results.append({"Organ": organ, "Metric": metric,
                         "friedman_stat": stat, "p_value": p, "n_patients": len(patients)})
    return pd.DataFrame(results)


def apply_multiple_comparison_correction(pvals: list[float], method: str = "bonferroni") -> np.ndarray:
    """
    Corrects a list of p-values for multiple comparisons.
    method: 'bonferroni' (family-wise error control, conservative) or
            'bh' (Benjamini-Hochberg, controls false discovery rate,
            typically more powerful when many tests are run).
    """
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if method == "bonferroni":
        return np.clip(pvals * n, 0, 1)
    elif method == "bh":
        order = np.argsort(pvals)
        ranked = pvals[order] * n / (np.arange(n) + 1)
        # enforce monotonicity (BH step-up procedure)
        corrected = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.clip(corrected, 0, 1)
        return out
    else:
        raise ValueError("method must be 'bonferroni' or 'bh'")


def pairwise_wilcoxon(df: pd.DataFrame, metric: str, organ: str,
                       correction: str = "bh") -> pd.DataFrame:
    """
    Post-hoc pairwise Wilcoxon signed-rank tests between every pair of
    techniques for a single organ/metric, with a multiple-comparison
    correction applied across all pairs tested for that organ.
    """
    arr, patients, organs, techniques = to_4d_array(df, metric)
    oi = organs.index(organ)
    organ_data = arr[:, oi, :]

    rows = []
    for i, j in combinations(range(len(techniques)), 2):
        stat, p = stats.wilcoxon(organ_data[:, i], organ_data[:, j])
        rows.append({"Organ": organ, "Metric": metric,
                      "Technique_A": techniques[i], "Technique_B": techniques[j],
                      "wilcoxon_stat": stat, "p_raw": p})

    result = pd.DataFrame(rows)
    result[f"p_{correction}"] = apply_multiple_comparison_correction(result["p_raw"].tolist(), correction)
    result["significant"] = result[f"p_{correction}"] < 0.05
    return result


def variance_decomposition(df: pd.DataFrame, metric: str, organ: str) -> pd.DataFrame:
    """
    Two-way ANOVA-style variance decomposition (patient effect vs.
    technique effect) for a single organ/metric, via a long-format OLS
    with patient and technique as categorical factors. Returns a
    standard ANOVA table (sum of squares, df, F, p) so you can see
    whether more of the variance in dose comes from patient anatomy or
    from the planning technique itself.
    """
    subset = df[df["Organ"] == organ][["PatientID", "Technique", metric]].dropna()
    grand_mean = subset[metric].mean()

    patient_means = subset.groupby("PatientID")[metric].mean()
    technique_means = subset.groupby("Technique")[metric].mean()

    n_patients = subset["PatientID"].nunique()
    n_techniques = subset["Technique"].nunique()

    ss_patient = n_techniques * ((patient_means - grand_mean) ** 2).sum()
    ss_technique = n_patients * ((technique_means - grand_mean) ** 2).sum()
    ss_total = ((subset[metric] - grand_mean) ** 2).sum()
    ss_residual = ss_total - ss_patient - ss_technique

    df_patient = n_patients - 1
    df_technique = n_techniques - 1
    df_residual = (n_patients - 1) * (n_techniques - 1)

    ms_patient = ss_patient / df_patient
    ms_technique = ss_technique / df_technique
    ms_residual = ss_residual / df_residual if df_residual > 0 else np.nan

    f_patient = ms_patient / ms_residual
    f_technique = ms_technique / ms_residual
    p_patient = 1 - stats.f.cdf(f_patient, df_patient, df_residual)
    p_technique = 1 - stats.f.cdf(f_technique, df_technique, df_residual)

    return pd.DataFrame([
        {"Source": "Patient", "SS": ss_patient, "df": df_patient, "MS": ms_patient,
         "F": f_patient, "p_value": p_patient,
         "pct_variance_explained": 100 * ss_patient / ss_total},
        {"Source": "Technique", "SS": ss_technique, "df": df_technique, "MS": ms_technique,
         "F": f_technique, "p_value": p_technique,
         "pct_variance_explained": 100 * ss_technique / ss_total},
        {"Source": "Residual", "SS": ss_residual, "df": df_residual, "MS": ms_residual,
         "F": np.nan, "p_value": np.nan,
         "pct_variance_explained": 100 * ss_residual / ss_total},
    ])


if __name__ == "__main__":
    import os
    if not os.path.exists("synthetic_dvh_data.csv"):
        from generate_synthetic_data import generate
        generate().to_csv("synthetic_dvh_data.csv", index=False)

    df = pd.read_csv("synthetic_dvh_data.csv")

    print("=== Friedman test across techniques, by organ (Dmean_Gy) ===")
    friedman_results = friedman_by_organ(df, "Dmean_Gy")
    print(friedman_results.to_string(index=False))

    print("\n=== Variance decomposition for L_Parotid (Dmean_Gy) ===")
    print(variance_decomposition(df, "Dmean_Gy", "L_Parotid").to_string(index=False))

    print("\n=== Pairwise post-hoc (Wilcoxon + BH correction) for L_Parotid (Dmean_Gy) ===")
    posthoc = pairwise_wilcoxon(df, "Dmean_Gy", "L_Parotid", correction="bh")
    print(posthoc.to_string(index=False))
