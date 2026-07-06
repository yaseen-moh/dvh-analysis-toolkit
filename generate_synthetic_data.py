"""
Generates a synthetic dose-volume-histogram (DVH) dataset shaped exactly
like a real multi-technique radiotherapy planning comparison study: N
patients x M organs-at-risk x several DVH metrics x K planning techniques.

This is included so the toolkit is fully runnable and demonstrable without
any real patient data -- no PHI, no de-identification questions, nothing
that could ever be mistaken for actual clinical data. All values below are
randomly generated and are NOT derived from or representative of any real
patient.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

TECHNIQUES = ["RUMC", "KTGM", "KT", "GM", "AZD"]
ORGANS = ["L_Lens", "R_Lens", "L_Eye", "R_Eye", "L_Lacrimal", "R_Lacrimal",
          "L_Parotid", "R_Parotid", "Body_Max"]
METRICS = ["Dmean_Gy", "Dmax_Gy", "V20_pct"]

# Rough "typical" baseline dose levels per organ so the synthetic data at
# least resembles plausible OAR dose magnitudes (not real clinical values).
ORGAN_BASELINE_GY = {
    "L_Lens": 3.0, "R_Lens": 3.0, "L_Eye": 8.0, "R_Eye": 8.0,
    "L_Lacrimal": 12.0, "R_Lacrimal": 12.0,
    "L_Parotid": 22.0, "R_Parotid": 22.0, "Body_Max": 55.0,
}

# Each technique gets a mild systematic offset + noise, so techniques are
# meaningfully different (as they would be from different optimization
# strategies) but still overlapping -- this is what makes the statistical
# comparison pipeline (Friedman test, pairwise corrections) meaningful.
TECHNIQUE_OFFSET_GY = {"RUMC": 0.0, "KTGM": -0.8, "KT": 0.4, "GM": 1.2, "AZD": -0.3}


def generate(n_patients: int = 10, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_patients + 1):
        # per-patient random effect: some patients just run "hotter" overall
        patient_effect = rng.normal(0, 1.5)
        for organ in ORGANS:
            baseline = ORGAN_BASELINE_GY[organ]
            for tech in TECHNIQUES:
                offset = TECHNIQUE_OFFSET_GY[tech]
                dmean = max(0.0, baseline + offset + patient_effect + rng.normal(0, 1.0))
                dmax = dmean * rng.uniform(1.15, 1.45)
                v20 = float(np.clip(
                    (dmean / max(baseline, 1.0)) * rng.uniform(15, 45), 0, 100
                ))
                rows.append({
                    "PatientID": f"P{pid:02d}",
                    "Organ": organ,
                    "Technique": tech,
                    "Dmean_Gy": round(dmean, 3),
                    "Dmax_Gy": round(dmax, 3),
                    "V20_pct": round(v20, 3),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("synthetic_dvh_data.csv", index=False)
    print(f"Generated {len(df)} rows -> synthetic_dvh_data.csv")
    print(df.head(10))
