import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import joblib


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/delaney_with_descriptors.csv")


# --------------------------------------------------
# 2. Features and target
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
# 3. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 4. Random Forest
# --------------------------------------------------

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 5. Hyperparameter search space
# --------------------------------------------------

param_grid = {
    "n_estimators": [200, 300, 500, 700],
    "max_depth": [None, 10, 15, 20, 25],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [0.6, 0.8, 1.0]
}


# --------------------------------------------------
# 6. 5-Fold CV
# --------------------------------------------------

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 7. Randomized Search
# --------------------------------------------------

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=30,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


print("\nStarting hyperparameter search...\n")

search.fit(X_train, y_train)


# --------------------------------------------------
# 8. Best parameters
# --------------------------------------------------

print("\n--- Best Parameters ---")

for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")


print("\nBest CV RMSE:")
print(-search.best_score_)


# --------------------------------------------------
# 9. Evaluate best model on untouched test set
# --------------------------------------------------

best_model = search.best_estimator_

predictions = best_model.predict(X_test)


mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n--- Tuned Random Forest ---")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# --------------------------------------------------
# 10. Training performance
# --------------------------------------------------

train_predictions = best_model.predict(X_train)

train_r2 = r2_score(
    y_train,
    train_predictions
)

print("\n--- Overfitting Check ---")

print(f"Training R² : {train_r2:.4f}")
print(f"Testing R²  : {r2:.4f}")
print(f"Difference  : {train_r2 - r2:.4f}")


# --------------------------------------------------
# 11. Feature importance
# --------------------------------------------------

importance = pd.Series(
    best_model.feature_importances_,
    index=features
).sort_values(ascending=False)


print("\n--- Tuned Model Feature Importance ---")

print(importance)


# --------------------------------------------------
# 12. Save tuned model
# --------------------------------------------------

joblib.dump(
    best_model,
    "models/solubility_random_forest_tuned.pkl"
)

print("\nTuned model saved to:")
print("models/solubility_random_forest_tuned.pkl")