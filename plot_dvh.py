"""Generates boxplots and spaghetti (per-patient) plots comparing techniques."""
import pandas as pd
import matplotlib.pyplot as plt
import os
from generate_synthetic_data import generate

if not os.path.exists("synthetic_dvh_data.csv"):
    generate().to_csv("synthetic_dvh_data.csv", index=False)

df = pd.read_csv("synthetic_dvh_data.csv")
organ = "L_Parotid"
metric = "Dmean_Gy"
subset = df[df["Organ"] == organ]
techniques = sorted(subset["Technique"].unique())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot across techniques
data_by_tech = [subset[subset["Technique"] == t][metric].values for t in techniques]
axes[0].boxplot(data_by_tech, labels=techniques)
axes[0].set_title(f"{organ} {metric} by Technique")
axes[0].set_ylabel(metric)

# Spaghetti plot: one line per patient across techniques
pivot = subset.pivot(index="PatientID", columns="Technique", values=metric)[techniques]
for _, row in pivot.iterrows():
    axes[1].plot(techniques, row.values, marker="o", alpha=0.5, linewidth=1)
axes[1].set_title(f"{organ} {metric} per Patient (spaghetti plot)")
axes[1].set_ylabel(metric)

plt.tight_layout()
plt.savefig("dvh_comparison.png", dpi=150)
print("Saved dvh_comparison.png")
