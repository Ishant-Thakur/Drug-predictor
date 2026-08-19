import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load descriptor dataset
df = pd.read_csv("data/delaney_with_descriptors.csv")

# Features we generated using RDKit
features = [
    "Molecular_Weight",
    "LogP",
    "TPSA",
    "H_Bond_Donors",
    "H_Bond_Acceptors",
    "Rotatable_Bonds",
    "Ring_Count",
    "Heavy_Atom_Count",
    "Fraction_CSP3"
]

target = "measured log solubility in mols per litre"

# ---------------------------------------------
# Correlation matrix
# ---------------------------------------------

correlation = df[features + [target]].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("RDKit Molecular Descriptors vs Solubility")
plt.tight_layout()
plt.show()


# ---------------------------------------------
# Correlation with target
# ---------------------------------------------

target_correlation = (
    correlation[target]
    .drop(target)
    .sort_values()
)

print("\n--- Correlation with Solubility ---")
print(target_correlation)


# ---------------------------------------------
# Feature statistics
# ---------------------------------------------

print("\n--- Feature Statistics ---")
print(df[features].describe())