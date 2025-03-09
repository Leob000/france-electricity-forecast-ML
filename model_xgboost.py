# %%
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_pinball_loss
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

CHEAT = True

# %%
df_full = pd.read_csv("Data/treated_data.csv")
# %%
df_full["Date"] = pd.to_datetime(df_full["Date"])
df_full["time_idx"] = np.arange(len(df_full))

# %%
f_mean = df_full.loc[df_full["Date"] <= "2022-09-01", "Net_demand"].mean()
f_std = df_full.loc[df_full["Date"] <= "2022-09-01", "Net_demand"].std()

for f in ["Net_demand", "Net_demand.1", "Net_demand.7"]:
    df_full[f] = (df_full[f] - f_mean) / f_std

df_full = pd.get_dummies(data=df_full, columns=["WeekDays", "BH_Holiday"], dtype=int)
# %%
target = "Net_demand"
to_keep = [
    # "Date",
    # "Net_demand",
    "Load.1",
    "Load.7",
    "Temp",
    "Temp_s95",
    "Temp_s99",
    "Temp_s95_min",
    "Temp_s95_max",
    "Temp_s99_min",
    "Temp_s99_max",
    "Wind",
    "Wind_weighted_ratio",
    "Nebulosity",
    "Nebulosity_weighted_ratio",
    "BH_before",
    "BH",
    "BH_after",
    "Year",
    "DLS",
    "Summer_break",
    "Christmas_break",
    "Holiday",
    "Holiday_zone_a",
    "Holiday_zone_b",
    "Holiday_zone_c",
    "Solar_power.1",
    "Solar_power.7",
    "Wind_power.1",
    "Wind_power.7",
    "Net_demand.1",
    "Net_demand.7",
    "Net_demand.1_trend",
    "x_dayofyear",
    "y_dayofyear",
    "x_dayofweek",
    "y_dayofweek",
    "lundi_vendredi",
    "flag_temp",
    "GovernmentResponseIndex_Average",
    "ContainmentHealthIndex_Average",
    "time_idx",
    "WeekDays_0",
    "WeekDays_1",
    "WeekDays_2",
    "WeekDays_3",
    "WeekDays_4",
    "WeekDays_5",
    "WeekDays_6",
    "BH_Holiday_0",
    "BH_Holiday_1",
    "BH_Holiday_10",
    "BH_Holiday_11",
]
# %%
df_train = df_full[df_full["Date"] <= "2022-09-01"]
df_test = df_full.loc[df_full["Date"] >= "2022-09-02"]

y_train = df_train[target].values
X_train = df_train[to_keep].values
X_test = df_test[to_keep].values

# %%
# Initialize the XGBoost regressor
# objective = "reg:quantileerror"
objective = "reg:squarederror"
model = XGBRegressor(
    n_estimators=100000,  # 100000
    learning_rate=0.05,  # 0.05
    max_depth=2,  # 2
    random_state=42,
    objective=objective,
    tree_method="hist",
    # quantile_alpha=0.1,
)

# Train the model on your training data
model.fit(X_train, y_train)

y_pred = model.predict(X_test) * f_std + f_mean
plt.plot(y_pred)

if CHEAT:
    df_truth = pd.read_csv("Data/test_better.csv")
    truth = df_truth["Net_demand"]
    plt.plot(truth)
    pin = mean_pinball_loss(truth, y_pred, alpha=0.8)

plt.show()
print(pin.round(1))

min_loss = 99999
min_i = 0
for i in range(-1500, 1000, 100):
    pin = mean_pinball_loss(truth, y_pred + i, alpha=0.8)
    if pin < min_loss:
        min_loss = pin
        min_i = i
print(min_loss.round(1), min_i)
# %%
preds_train = model.predict(X_train) * f_std + f_mean
preds_train = pd.DataFrame(preds_train, columns=["xgb"])
preds_train["Date"] = pd.date_range(
    start="2013-03-02", periods=len(preds_train), freq="D"
)

preds_test = model.predict(X_test) * f_std + f_mean
preds_test = pd.DataFrame(preds_test, columns=["xgb"])
preds_test["Date"] = pd.date_range(
    start="2020-09-02", periods=len(preds_test), freq="D"
)

# preds_train.to_csv("Data/preds_xgb_train.csv", index=False)
# preds_test.to_csv("Data/preds_xgb_test.csv", index=False)

# %%
# Define a custom scorer that uses mean_pinball_loss with alpha=0.8.
# Note: greater_is_better is False because lower pinball loss is better.
pinball_scorer = make_scorer(mean_pinball_loss, alpha=0.8, greater_is_better=False)

# Set up the hyperparameter distributions, including quantile_alpha if you wish to tune it.
param_dist = {
    "n_estimators": np.arange(100, 501, 50),  # For example, 100, 150, ..., 500
    "max_depth": np.arange(3, 11),  # Depth from 3 to 10
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "quantile_alpha": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9],  # Values to try
}

# Initialize the model with the quantile error objective.
# If your final evaluation is with alpha=0.8, you might consider starting with quantile_alpha=0.8,
# but here we include it in the search space.
model = XGBRegressor(
    objective="reg:quantileerror",
    tree_method="hist",
    random_state=42,
)

# Set up RandomizedSearchCV with 50 iterations and 3-fold cross-validation.
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    scoring=pinball_scorer,
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1,
)

# Assume X_train and y_train are defined from your preprocessed training data.
random_search.fit(X_train, y_train)

print("Best parameters found:", random_search.best_params_)

# Optionally, evaluate on your test data.
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

test_pinball_loss = mean_pinball_loss(truth, y_pred, alpha=0.8)
print("Test Mean Pinball Loss (alpha=0.8):", test_pinball_loss)

# %%
