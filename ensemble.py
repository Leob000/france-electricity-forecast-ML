# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_pinball_loss
import numpy as np
from sklearn.neural_network import MLPRegressor

plt.rcParams["figure.figsize"] = [10, 6]

CHEAT = True  # REM


# %%
# Importation et formattage de toutes les preds, pour test et pour train
df_rf_test = pd.read_csv("Data/preds_rf_test.csv")
df_rf_train = pd.read_csv("Data/preds_rf_train.csv")
df_tft_test = pd.read_csv("Data/preds_tft_new.csv")
df_tft_train = pd.read_csv("Data/preds_tft_train.csv")
df_gam_test = pd.read_csv("Data/preds_gam_test.csv")
df_gam_train = pd.read_csv("Data/preds_gam_train.csv")
df_truth_train = pd.read_csv("Data/treated_data.csv")

df_truth_test = pd.read_csv("Data/test_better.csv")  # REM
# %%

df_rf_test.index = pd.date_range(start="2022-09-02", periods=len(df_rf_test), freq="D")
df_rf_test.rename(columns={"predictions_test": "RF"}, inplace=True)

df_tft_test.index = pd.date_range(
    start="2022-09-02", periods=len(df_tft_test), freq="D"
)
df_tft_test = df_tft_test.iloc[:, 1:]

if CHEAT:
    df_truth_test = df_truth_test.set_index(
        pd.to_datetime(df_truth_test["Date"])
    )  # REM

preds_test = pd.merge(
    left=df_rf_test, right=df_tft_test, right_index=True, left_index=True
)
if CHEAT:
    preds_test = pd.merge(
        left=preds_test,
        right=df_truth_test["Net_demand"],
        right_index=True,
        left_index=True,
    )  # REM

# ATTENTION pour le formattage des pred sur train, on doit supprimer les 10 premiers jours (car par prédis par le modèle TFT)
df_tft_train.index = pd.date_range(
    start="2013-03-12", periods=len(df_tft_train), freq="D"
)
df_tft_train = df_tft_train.iloc[:, 1:]

df_rf_train.rename(columns={"predictions_train": "RF"}, inplace=True)
df_rf_train.index = pd.date_range(
    start="2013-03-02", periods=len(df_rf_train), freq="D"
)
df_rf_train = df_rf_train.iloc[10:]
preds_train = pd.merge(
    left=df_rf_train, right=df_tft_train, left_index=True, right_index=True
)

# %%
df_gam_test.index = pd.date_range(
    start="2022-09-02", periods=len(df_gam_test), freq="D"
)
df_gam_test.rename(columns={"Net_demand": "gam"}, inplace=True)
# %%

preds_test = pd.merge(
    left=preds_test, right=df_gam_test["gam"], right_index=True, left_index=True
)
# %%
df_gam_train.index = pd.date_range(
    start="2013-03-02", periods=len(df_gam_train), freq="D"
)
df_gam_train.rename(columns={"Net_demand": "gam"}, inplace=True)
# %%

df_gam_train = df_gam_train.iloc[10:]
preds_train = pd.merge(
    left=preds_train, right=df_gam_train["gam"], right_index=True, left_index=True
)

# %%
df_truth_train.index = pd.date_range(
    start="2013-03-02", periods=len(df_truth_train), freq="D"
)
df_truth_train = df_truth_train.iloc[10:]
preds_train = pd.merge(
    left=preds_train,
    right=df_truth_train["Net_demand"],
    right_index=True,
    left_index=True,
)
# %%
preds_train.rename(columns={"Net_demand": "target"}, inplace=True)
preds_test.rename(columns={"Net_demand": "target"}, inplace=True)
# %%
features = preds_train.columns.to_list()
features.remove("target")

if CHEAT:
    for df_name, df in [("Train", preds_train), ("Test", preds_test)]:  # REM
        for f in features:
            err = mean_pinball_loss(df["target"], df[f], alpha=0.8)
            print(f"{df_name}\t{f}\tPinball: {err}")
else:
    for f in features:
        err = mean_pinball_loss(preds_train["target"], preds_train[f], alpha=0.8)
        print(f"Train\t{f}\tPinball: {err}")

# %%
# Plot des preds test
if CHEAT:
    plt.plot(preds_test["target"])
for f in ["q0", "gam", "RF"]:
    plt.plot(preds_test[f])
plt.show()
# %%
# Plot des preds train
plt.plot(preds_train.loc["2021", "target"])
for f in ["q0", "gam", "RF"]:
    plt.plot(preds_train.loc["2021", f])
# %%
for f in ["q0", "q3", "gam", "RF"]:
    plt.hist(preds_train[f] - preds_train["target"], bins=100)
    plt.title(f"Histogram of residuals, mode {f}")
    plt.show()
# %%
aggregate = ["q0", "gam"]
preds_train["aggregate"] = 0
preds_train
for f in aggregate:
    preds_train["aggregate"] += preds_train[f]
preds_train["aggregate"] /= len(aggregate)
mean_pinball_loss(preds_train["target"], preds_train["aggregate"], alpha=0.8)

# %%
# Loss test
if CHEAT:
    aggregate = ["q0", "gam"]
    preds_test["aggregate"] = 0
    for f in aggregate:
        preds_test["aggregate"] += preds_test[f]
    preds_test["aggregate"] /= len(aggregate)
    print(mean_pinball_loss(preds_test["target"], preds_test["aggregate"], alpha=0.8))
# %%
# Test pour voir le poids est le meilleur, 50/50 semble bon
if CHEAT:
    for gam_weight in np.arange(0, 1.1, 0.1):
        preds_test["aggregate"] = (
            gam_weight * preds_test["gam"] + (1 - gam_weight) * preds_test["q0"]
        )
        print(
            gam_weight.round(1),
            mean_pinball_loss(preds_test["target"], preds_test["aggregate"], alpha=0.8),
        )

# %%
if CHEAT:
    gam_weight = 0.5
    for q in ["q0", "q1", "q2", "q3", "q4", "q5", "q6"]:
        print(q)
        for i in range(-2500, 200, 200):
            preds_test["aggregate"] = gam_weight * preds_test["gam"] + (
                1 - gam_weight
            ) * (preds_test[q] + i)
            print(
                i,
                mean_pinball_loss(
                    preds_test["target"], preds_test["aggregate"], alpha=0.8
                ),
            )

# %%
if "aggregate" in preds_train.columns:
    preds_train.drop(columns=["aggregate"], inplace=True)
if "aggregate" in preds_test.columns:
    preds_test.drop(columns=["aggregate"], inplace=True)
# %%
df_truth_train.rename(columns={"Net_demand": "target"}, inplace=True)

df_truth_train["time_idx"] = np.arange(len(df_truth_train))
df_truth_train["Date"] = pd.to_datetime(df_truth_train["Date"])


f_mean = df_truth_train.loc[df_truth_train["Date"] <= "2022-09-01", "target"].mean()
f_std = df_truth_train.loc[df_truth_train["Date"] <= "2022-09-01", "target"].std()

for f in ["target", "Net_demand.1", "Net_demand.7"]:
    df_truth_train[f] = (df_truth_train[f] - f_mean) / f_std

df_truth_train = pd.get_dummies(
    data=df_truth_train, columns=["WeekDays", "BH_Holiday"], dtype=int
)

# %%
df_mlp_train = df_truth_train.loc[df_truth_train.index <= "2022-09-01"]
df_mlp_test = df_truth_train.loc[df_truth_train.index >= "2022-09-02"]

df_mlp_train = pd.merge(
    left=df_mlp_train, right=preds_train, right_index=True, left_index=True
)
df_mlp_train.drop(columns=["target_y"], inplace=True)
df_mlp_train.rename(columns={"target_x": "target"}, inplace=True)

df_mlp_test = pd.merge(
    left=df_mlp_test, right=preds_test, right_index=True, left_index=True
)
df_mlp_test.drop(columns=["target_y"], inplace=True)
df_mlp_test.rename(columns={"target_x": "target"}, inplace=True)
# %%
MODELS_FOR_MLP = ["RF", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "gam"]
for df in [df_mlp_train, df_mlp_test]:
    for f in MODELS_FOR_MLP:
        df[f] = (df[f] - f_mean) / f_std
# %%
to_keep = [
    # "Date",
    # "target",
    # "Load.1",
    # "Load.7",
    # "Temp",
    # "Temp_s95",
    # "Temp_s99",
    # "Temp_s95_min",
    # "Temp_s95_max",
    # "Temp_s99_min",
    # "Temp_s99_max",
    # "Wind",
    # "Wind_weighted_ratio",
    # "Nebulosity",
    # "Nebulosity_weighted_ratio",
    # "BH_before",
    # "BH",
    # "BH_after",
    "Year",
    # "DLS",
    # "Summer_break",
    # "Christmas_break",
    # "Holiday",
    # "Holiday_zone_a",
    # "Holiday_zone_b",
    # "Holiday_zone_c",
    # "Solar_power.1",
    # "Solar_power.7",
    # "Wind_power.1",
    # "Wind_power.7",
    # "Net_demand.1",
    # "Net_demand.7",
    # "Net_demand.1_trend",
    # "x_dayofyear",
    # "y_dayofyear",
    # "x_dayofweek",
    # "y_dayofweek",
    # "lundi_vendredi",
    # "flag_temp",
    # "GovernmentResponseIndex_Average",
    # "ContainmentHealthIndex_Average",
    "time_idx",
    # "WeekDays_0",
    # "WeekDays_1",
    # "WeekDays_2",
    # "WeekDays_3",
    # "WeekDays_4",
    # "WeekDays_5",
    # "WeekDays_6",
    # "BH_Holiday_0",
    # "BH_Holiday_1",
    # "BH_Holiday_10",
    # "BH_Holiday_11",
    # "RF",
    "q0",
    # "q1",
    # "q2",
    # "q3",
    # "q4",
    # "q5",
    # "q6",
    "gam",
]
y_train = df_mlp_train["target"]
X_train = df_mlp_train[to_keep].values
y_test = df_mlp_test["target"]
X_test = df_mlp_test[to_keep].values

model = MLPRegressor(
    hidden_layer_sizes=(395),
    # alpha=0.0008,
    alpha=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    # early_stopping=True,
    learning_rate_init=0.0001,  # 0.0001
    verbose=True,
    activation="relu",
    solver="adam",
    batch_size=128,  # 128
    tol=0.00001,
    n_iter_no_change=10000,
    max_iter=100,  # 100
    random_state=42,
)

model.fit(X_train, y_train)

preds_mlp_train = model.predict(X_train) * f_std + f_mean
preds_train["mlp"] = preds_mlp_train
print(
    "Train Pinball",
    mean_pinball_loss(preds_train["target"], preds_train["mlp"], alpha=0.8),
)

preds_mlp_test = model.predict(X_test) * f_std + f_mean
preds_test["mlp"] = preds_mlp_test
print(
    "Test Pinball",
    mean_pinball_loss(preds_test["target"], preds_test["mlp"], alpha=0.8),
)
# %%
params = model.get_params()
preds_train_saved = preds_train["mlp"]
preds_test_saved = preds_test["mlp"]
# Err 440/326
# %%
preds_test_mlp_df = pd.DataFrame(preds_test["mlp"])
preds_test_mlp_df.rename(columns={"mlp": "Net_demand"}, inplace=True)
preds_test_mlp_df["Id"] = np.arange(len(preds_test_mlp_df)) + 1
preds_test_mlp_df = preds_test_mlp_df[["Id", "Net_demand"]]
preds_test_mlp_df.to_csv("Data/preds_mlp.csv", index=False, header=["Id", "Net_demand"])
# plt.plot(preds_test["mlp"])
