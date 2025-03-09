# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_pinball_loss
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV

plt.rcParams["figure.figsize"] = [10, 6]
TFT_SEQUENCE_SIZE = 366


# %%
# Importation et formattage de toutes les preds, pour test et pour train
df_rf_test = pd.read_csv("Data/preds_rf_test.csv")
df_rf_train = pd.read_csv("Data/preds_rf_train.csv")
df_tft_test = pd.read_csv("Data/preds_tft_new.csv")
df_tft_train = pd.read_csv("Data/preds_tft_train_new.csv")
df_gam_test = pd.read_csv("Data/preds_gam_test_old.csv")
df_gam_train = pd.read_csv("Data/preds_gam_train_old.csv")
df_xgb_test = pd.read_csv("Data/preds_xgb_test.csv")
df_xgb_train = pd.read_csv("Data/preds_xgb_train.csv")
df_truth_train = pd.read_csv("Data/treated_data.csv")

# %%
# Formattage des prédictions sur train et test
df_rf_test.index = pd.date_range(start="2022-09-02", periods=len(df_rf_test), freq="D")
df_rf_test.rename(columns={"predictions_test": "RF"}, inplace=True)

df_tft_test.index = pd.date_range(
    start="2022-09-02", periods=len(df_tft_test), freq="D"
)
df_tft_test = df_tft_test.iloc[:, 1:]

preds_test = pd.merge(
    left=df_rf_test, right=df_tft_test, right_index=True, left_index=True
)

# ATTENTION pour le formattage des pred sur train, on doit supprimer les X premiers jours (car par prédis par le modèle TFT, suivant sa taille de séquence d'apprentissage)
shifted_date = pd.to_datetime("2013-03-02") + pd.DateOffset(days=TFT_SEQUENCE_SIZE)

df_tft_train.index = pd.date_range(
    start=shifted_date, periods=len(df_tft_train), freq="D"
)
df_tft_train = df_tft_train.iloc[:, 1:]
df_rf_train.rename(columns={"predictions_train": "RF"}, inplace=True)
df_rf_train.index = pd.date_range(
    start="2013-03-02", periods=len(df_rf_train), freq="D"
)
df_rf_train = df_rf_train.iloc[TFT_SEQUENCE_SIZE:]
preds_train = pd.merge(
    left=df_rf_train, right=df_tft_train, left_index=True, right_index=True
)

df_gam_test.index = pd.date_range(
    start="2022-09-02", periods=len(df_gam_test), freq="D"
)
df_gam_test.rename(columns={"Net_demand": "gam"}, inplace=True)

preds_test = pd.merge(
    left=preds_test, right=df_gam_test["gam"], right_index=True, left_index=True
)
df_gam_train.index = pd.date_range(
    start="2013-03-02", periods=len(df_gam_train), freq="D"
)
df_gam_train.rename(columns={"Net_demand": "gam"}, inplace=True)

df_gam_train = df_gam_train.iloc[TFT_SEQUENCE_SIZE:]
preds_train = pd.merge(
    left=preds_train, right=df_gam_train["gam"], right_index=True, left_index=True
)

df_truth_train.index = pd.date_range(
    start="2013-03-02", periods=len(df_truth_train), freq="D"
)
df_truth_train = df_truth_train.iloc[TFT_SEQUENCE_SIZE:]
preds_train = pd.merge(
    left=preds_train,
    right=df_truth_train["Net_demand"],
    right_index=True,
    left_index=True,
)
df_xgb_train["Date"] = pd.to_datetime(df_xgb_train["Date"])
df_xgb_train = df_xgb_train.set_index("Date")
df_xgb_train = df_xgb_train.iloc[TFT_SEQUENCE_SIZE:]

df_xgb_test["Date"] = pd.to_datetime(df_xgb_test["Date"])
df_xgb_test = df_xgb_test.set_index("Date")
df_xgb_test

preds_train = pd.merge(preds_train, df_xgb_train, right_index=True, left_index=True)
preds_test = pd.merge(preds_test, df_xgb_test, right_index=True, left_index=True)
# %%
preds_train.rename(columns={"Net_demand": "target"}, inplace=True)
preds_test.rename(columns={"Net_demand": "target"}, inplace=True)
# %%
# Les prédictions q0 à q6 sont les prédictions du modèle TFT, qui prédit 7 quantiles différents
# q3 est la prédiction de la médiane
features = preds_train.columns.to_list()
features.remove("target")

# %%
# En plottant la différence carrée entre les quantiles estimés par TFT et les prédictions du GAM, on se rend compte que le quantile le plus proche du GAM est q0. Ce sera donc celui que nous choisirons de prendre.
# De plus des essais d'upload sur la compétition kaggle nous montrent que c'est bien celui que nous devrions prendre (meilleur public score)... mais bon cette méthode n'est pas très rigoureuse
# En mettant sur histogramme la différence de ces quantiles avec GAM, on se rend d'ailleurs compte que c'est avec ce quantile que la différence est la plus centrée
plt.plot((preds_test["q0"] - preds_test["gam"]) ** 2)
plt.plot((preds_test["q3"] - preds_test["gam"]) ** 2)
plt.plot((preds_test["q6"] - preds_test["gam"]) ** 2)
plt.legend(["(q0-gam)^2", "(q3-gam)^2", "(q6-gam)^2"])
plt.show()
for i in [f"q{j}" for j in range(7)]:
    plt.hist(preds_test[i] - preds_test["gam"], bins=50)
    plt.title(f"Histogram of residuals, {i}")
    plt.show()
# %%
# En plottant la différence carrée de GAM, TFT et XGboost on se rend compte que les modèles diffèrent le plus sur la périodre d'août 2023.
# Cela peut notamment être dû à des facteurs inhabituels arrivant sur ce mois; la canicule, et une augmentation tarifaire de 10% instauré par l'Etat
plt.plot((preds_test["xgb"] - preds_test["gam"]) ** 2)
plt.plot((preds_test["xgb"] - preds_test["q0"]) ** 2)
plt.plot((preds_test["q0"] - preds_test["gam"]) ** 2)
plt.legend(["(xgb-gam)^2", "(xgb-q0)^2", "(q0-gam)^2"])
plt.show()
# %%
# Plot des preds test
for f in ["q0", "gam", "xgb"]:
    plt.plot(preds_test[f])
plt.legend(["q0", "gam", "xgb"])
plt.show()
# %%
# Pinball sur le train set
# RF	Pinball train: 221.9
# q0	Pinball train: 1140.3
# q1	Pinball train: 673.6
# q2	Pinball train: 418.3
# q3	Pinball train: 249.5
# q4	Pinball train: 174.5
# q5	Pinball train: 176.8
# q6	Pinball train: 279.1
# gam	Pinball train: 348.5
# xgb	Pinball train: 310.9
# On note une forte pinball train pour q0 qui décroît pour des quantiles plus haut, mais nous restons confiant sur le choix de ce quantile
# Pour être rigoureux nous aurious pu faire une validation des pinball des quantiles sur la dernière année du train set, mais le package du modèle TFT étant très rigide, nous ne nous sommes pas lancés dans cette aventure
for f in features:
    err = mean_pinball_loss(preds_train["target"], preds_train[f], alpha=0.8)
    print(f"{f}\tPinball train: {err.round(1)}")

# %%
# Comme méthode d'ensemble simple, on peut choisir de faire simplement une moyenne des prédictions des différents modèles
agg = ["q0", "gam", "xgb"]
preds_train["agg"] = 0
preds_train
for f in agg:
    preds_train["agg"] += preds_train[f]
preds_train["agg"] /= len(agg)
mean_pinball_loss(preds_train["target"], preds_train["agg"], alpha=0.8)

# %%
# %%
# On peut par ailleurs prendre une moyenne pondérée qui favorise plus un certain modèle
for gam_weight in np.arange(0, 1.1, 0.1):
    preds_train["agg"] = (
        gam_weight * preds_train["gam"] + (1 - gam_weight) * preds_train["xgb"]
    )
    print(
        mean_pinball_loss(preds_train["target"], preds_train["agg"], alpha=0.8).round(
            1
        ),
        "gam_weight:",
        gam_weight.round(1),
    )

if "agg" in preds_train.columns:
    preds_train.drop(columns=["agg"], inplace=True)
if "agg" in preds_test.columns:
    preds_test.drop(columns=["agg"], inplace=True)
# %%
# On peut aussi faire un MLP qui prend comme input les prédictions des modèles, plus ou moins les différentes variables de départ

# Feature engineering basique
df_truth_train.rename(columns={"Net_demand": "target"}, inplace=True)

df_truth_train["time_idx"] = np.arange(len(df_truth_train))
df_truth_train["Date"] = pd.to_datetime(df_truth_train["Date"])


f_mean = df_truth_train.loc[df_truth_train["Date"] <= "2022-09-01", "target"].mean()
f_std = df_truth_train.loc[df_truth_train["Date"] <= "2022-09-01", "target"].std()

for f in ["target", "Net_demand.1", "Net_demand.7"]:
    df_truth_train[f] = (df_truth_train[f] - f_mean) / f_std

df_truth_train["dayofyear"] = df_truth_train["Date"].dt.dayofyear
df_truth_train["dayofyear"] = (
    df_truth_train["dayofyear"] - df_truth_train["dayofyear"].min()
) / (df_truth_train["dayofyear"].max() - df_truth_train["dayofyear"].min())

df_truth_train = pd.get_dummies(
    data=df_truth_train, columns=["WeekDays", "BH_Holiday"], dtype=int
)

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
# %%
# Normalisation des prédictions
MODELS_FOR_MLP = ["RF", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "gam", "xgb"]
for df in [df_mlp_train, df_mlp_test]:
    for f in MODELS_FOR_MLP:
        df[f] = (df[f] - f_mean) / f_std
# %%
# Variables à garder pour le MLP
# Nous avons testé beaucoup de combinaisons différentes grâce à la validation mise en place plus loin, une sélection parcimonieuse semble la plus efficace
# La plus efficace semble de prendre les prédictions de q0 et gam, mais pas celles du randomForest ni XGboost
to_keep = [
    "dayofyear",
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
    # "xgb",
]
y_train = df_mlp_train["target"]
X_train = df_mlp_train[to_keep].values
y_test = df_mlp_test["target"]
X_test = df_mlp_test[to_keep].values
# %%
# Pour trouver les meilleurs hyperparamètres, on effecture un random search
# On va chercher à minimiser le validation score, qui sera ici obtenu grâce à la MSE
VALIDATION = False
if VALIDATION:
    param_dist = {
        "hidden_layer_sizes": [(395,), (256,), (128,), (64,), (32,)],
        "alpha": [0.0001, 0.001, 0.005, 0.01, 0.1],
        "learning_rate_init": [0.0001, 0.001, 0.01],
        "batch_size": [32, 64, 128, 256],
        "max_iter": [80, 110, 150, 200],
    }

    model = MLPRegressor(
        early_stopping=True,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        verbose=False,
        activation="relu",
        solver="adam",
        tol=0.00001,
        n_iter_no_change=10000,
        random_state=17,
    )

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=3,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=1,
        random_state=17,
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)

    print("Best parameters found: ", random_search.best_params_)
    print("Best validation score: ", -random_search.best_score_)

# %%
# On peut aussi faire une validation plus classique, en prenant comme jeu de validation la dernière année du train set
# Jeux de validation:
y_train = df_mlp_train.loc[df_mlp_train.index <= "2021-09-01", "target"]
X_train = df_mlp_train.loc[df_mlp_train.index <= "2021-09-01", to_keep].values
y_test = df_mlp_train.loc[df_mlp_train.index >= "2021-09-02", "target"]
X_test = df_mlp_train.loc[df_mlp_train.index >= "2021-09-02", to_keep].values

# On peut ensuite entraîner à la main différents modèles et étudier les prédictions
model = MLPRegressor(
    hidden_layer_sizes=(395),
    alpha=0.005,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    learning_rate_init=0.0001,
    verbose=True,
    activation="relu",
    solver="adam",
    batch_size=128,
    tol=0.00001,
    n_iter_no_change=10000,
    max_iter=110,
    random_state=17,
)
# 224.9 de pinball loss sur la validation avec ces hyperparamètres, nous les retenons donc
model.fit(X_train, y_train)

preds_mlp_test_val = model.predict(X_test)
preds_mlp_test_val = preds_mlp_test_val * f_std + f_mean
truth_val = y_test * f_std + f_mean
print(mean_pinball_loss(truth_val, preds_mlp_test_val, alpha=0.8))

# %%
# Les résidus sur le set de validation semblent gaussiens
# De plus, les résidus carrés en fonction du temps ne semblent pas dégager de période vraiment trop mal prédite
# On note cependant des résidus carrés élevés sur la période décembre 2021/janvier 2022
preds_mlp_test_val_df = pd.DataFrame(
    preds_mlp_test_val,
    index=pd.date_range(start="2021-09-02", periods=len(preds_mlp_test_val), freq="D"),
    columns=["preds_mlp_test_val"],
)
df_temp = pd.merge(preds_mlp_test_val_df, truth_val, right_index=True, left_index=True)
plt.hist(df_temp["target"] - df_temp["preds_mlp_test_val"], bins=50)
plt.show()
plt.plot((df_temp["target"] - df_temp["preds_mlp_test_val"]) ** 2)
plt.show()

# %%
# On ré-entraîne le modèle sur toutes les données avec les bons paramètres
y_train = df_mlp_train["target"]
X_train = df_mlp_train[to_keep].values
y_test = df_mlp_test["target"]
X_test = df_mlp_test[to_keep].values

model = MLPRegressor(
    hidden_layer_sizes=(395),
    alpha=0.005,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    learning_rate_init=0.0001,
    verbose=True,
    activation="relu",
    solver="adam",
    batch_size=128,
    tol=0.00001,
    n_iter_no_change=10000,
    max_iter=110,
    random_state=17,
)

model.fit(X_train, y_train)
preds_train["mlp"] = model.predict(X_train) * f_std + f_mean
preds_test["mlp"] = model.predict(X_test) * f_std + f_mean

print(
    "Train Pinball",
    mean_pinball_loss(preds_train["target"], preds_train["mlp"], alpha=0.8),
)

# %%
# On output un csv pour upload sur la compétition
preds_test_mlp_df = pd.DataFrame(preds_test["mlp"])
preds_test_mlp_df.rename(columns={"mlp": "Net_demand"}, inplace=True)
preds_test_mlp_df["Id"] = np.arange(len(preds_test_mlp_df)) + 1
preds_test_mlp_df = preds_test_mlp_df[["Id", "Net_demand"]]
# preds_test_mlp_df.to_csv("Data/preds_mlp.csv", index=False, header=["Id", "Net_demand"])
