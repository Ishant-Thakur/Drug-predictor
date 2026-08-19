import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/delaney_with_descriptors.csv")


# --------------------------------------------------
# 2. Select molecular features
# --------------------------------------------------

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


X = df[features]
y = df[target]


# --------------------------------------------------
# 3. Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. Linear Regression
# --------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


# --------------------------------------------------
# 5. Random Forest
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)


# --------------------------------------------------
# 6. Evaluation function
# --------------------------------------------------

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = mean_squared_error(
        actual,
        predicted
    ) ** 0.5

    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


# --------------------------------------------------
# 7. Evaluate models
# --------------------------------------------------

evaluate_model(
    "Linear Regression",
    y_test,
    linear_predictions
)

evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)


# --------------------------------------------------
# 8. Random Forest feature importance
# --------------------------------------------------

importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n--- Random Forest Feature Importance ---")

print(importance)


# --------------------------------------------------
# 9. Plot Random Forest predictions
# --------------------------------------------------

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.6
)

# Perfect prediction line
min_value = min(y_test.min(), rf_predictions.min())
max_value = max(y_test.max(), rf_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual LogS")
plt.ylabel("Predicted LogS")
plt.title("Random Forest: Actual vs Predicted Solubility")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. Save Random Forest model
# --------------------------------------------------

joblib.dump(
    rf_model,
    "models/solubility_random_forest.pkl"
)

print("\nModel saved:")
print("models/solubility_random_forest.pkl")