import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/delaney-processed.csv")

# --------------------------------------------------
# 1. Distribution of measured solubility
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(
    df["measured log solubility in mols per litre"],
    bins=30,
    kde=True
)

plt.title("Distribution of Measured Solubility")
plt.xlabel("Measured Log Solubility (logS)")
plt.ylabel("Number of Molecules")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. Molecular Weight vs Solubility
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Molecular Weight",
    y="measured log solubility in mols per litre"
)

plt.title("Molecular Weight vs Solubility")
plt.xlabel("Molecular Weight")
plt.ylabel("Measured Log Solubility")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. H-Bond Donors vs Solubility
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Number of H-Bond Donors",
    y="measured log solubility in mols per litre"
)

plt.title("H-Bond Donors vs Solubility")
plt.xlabel("Number of H-Bond Donors")
plt.ylabel("Measured Log Solubility")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 4. Number of Rings vs Solubility
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Number of Rings",
    y="measured log solubility in mols per litre"
)

plt.title("Number of Rings vs Solubility")
plt.xlabel("Number of Rings")
plt.ylabel("Measured Log Solubility")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. Correlation Matrix
# --------------------------------------------------

numeric_columns = [
    "Minimum Degree",
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Polar Surface Area",
    "measured log solubility in mols per litre"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Molecular Features")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. ESOL prediction vs actual measured solubility
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="ESOL predicted log solubility in mols per litre",
    y="measured log solubility in mols per litre"
)

plt.title("ESOL Prediction vs Measured Solubility")
plt.xlabel("ESOL Predicted LogS")
plt.ylabel("Measured LogS")

plt.tight_layout()
plt.show()