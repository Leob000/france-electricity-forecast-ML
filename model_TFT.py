# %%
from sklearn.metrics import mean_pinball_loss  # noqa: F401
import copy  # noqa: F401
from pathlib import Path  # noqa: F401
import warnings  # noqa: F401
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor  # noqa: F401
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch  # noqa: F401
import matplotlib.pyplot as plt
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss  # noqa: F401
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,  # noqa: F401
)

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

FULL_TRAIN = True  # Entrainement complet, False pour validation
plt.rcParams["figure.figsize"] = (10, 6)

# %%
df_full = pd.read_csv("Data/treated_data.csv")
# %%
# Création de "time_idx", feature qui incrémente à chaque timestep
df_full["time_idx"] = np.arange(len(df_full))
df_full["Date"] = pd.to_datetime(df_full["Date"])

# Le package requiert de trier par groupe, on fait un groupe global pour toutes les observations..
df_full["fakegroup"] = "a"
# %%
# Le package requiert de ne pas avoir de "." dans le nom des features
features_to_rename = [
    "Load.1",
    "Load.7",
    "Solar_power.1",
    "Solar_power.7",
    "Wind_power.1",
    "Wind_power.7",
    "Net_demand.1",
    "Net_demand.7",
    "Net_demand.1_trend",
]

df_full.rename(
    columns={feature: feature.replace(".", "_") for feature in features_to_rename},
    inplace=True,
)
# %%
# Normalisation des net_demand laggées, Net_demand sera pris en charge directement pas le modèle
f_mean = df_full.loc[df_full["Date"] <= "2022-09-01", "Net_demand"].mean()
f_std = df_full.loc[df_full["Date"] <= "2022-09-01", "Net_demand"].std()

for f in ["Net_demand_1", "Net_demand_7"]:
    df_full[f] = (df_full[f] - f_mean) / f_std
# %%
df_full.rename(columns={"Date": "date"}, inplace=True)
# Le modèle requiert de n'avoir pas de NaN même pour la target feature, même si les valeurs pour le test set ne seront évidemment pas utilisées
df_full.fillna(40000, inplace=True)
# Passage en feature catégorielle de WeekDays et BH_Holiday
df_full = pd.get_dummies(data=df_full, columns=["WeekDays", "BH_Holiday"], dtype=int)
# %%
# FULL train utilise tout le train set pour apprendre (les 2 jours du test set ne sont pas pris en compte par le modèle, la librairie requiert juste de mettre un minimum de validation set donc on prend ici l'horizon de prediction = 2 jours en plus)
# C'est pourquoi on met "2022-09-04" pour le test set; le 2 et 3 septembre sont dans le train set mais ne seront pas utilisés pour l'apprentissage
# Comme on peut le voir le package pytorch lightning est assez rigide et prend des inputs bien précis qui ne sont pas toujours bien documentés, ce qui rend son utilisation malheureusement peu pratique/intuitive
if FULL_TRAIN:
    df_test = df_full[df_full["date"] >= "2022-09-04"]
    df_train = df_full[df_full["date"] <= "2022-09-03"]
else:  # Validation sur les 2 derniers jours du train set
    df_test = df_full[df_full["date"] >= "2022-09-02"]
    df_train = df_full[df_full["date"] <= "2022-09-01"]

# %%
max_encoder_length = 366  # Séquence qui est prise pour le modèle pour apprendre; résultats légèrements meilleurs quand prend 1 an (366j), mais bcp plus long en calcul qu'une séquence de 2j
max_prediction_length = 2  # Nombre de timesteps futures à prédire, j'aurais bien mis juste prédire le prochain jour (=1), mais le package n'accepte pas cette valeur
# Pour le test set le modèle sera donc glissant, on va prédire chaque jour grâce aux `max_encoder_length` derniers jours le précédent

training_cutoff = df_train["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df_train[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Net_demand",
    group_ids=["fakegroup"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["fakegroup"],
    # static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    static_reals=[],
    # time_varying_known_categoricals=["special_days", "month"],
    time_varying_known_categoricals=[],
    variable_groups={
        # "special_days": special_days
    },  # group of categorical variables can be treated as one variable
    time_varying_known_reals=[
        "time_idx",
        "Net_demand_1",
        "Load_1",
        "Load_7",
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
        "Year",
        "Solar_power_1",
        "Solar_power_7",
        "Wind_power_1",
        "Wind_power_7",
        "Net_demand_1",
        "Net_demand_7",
        "Net_demand_1_trend",
        "x_dayofyear",
        "y_dayofyear",
        "x_dayofweek",
        "y_dayofweek",
        "GovernmentResponseIndex_Average",
        "ContainmentHealthIndex_Average",
        "BH_before",
        "BH",
        "BH_after",
        "DLS",
        "Summer_break",
        "Christmas_break",
        "Holiday",
        "Holiday_zone_a",
        "Holiday_zone_b",
        "Holiday_zone_c",
        # "lundi_vendredi",
        "flag_temp",
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
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        # "Net_demand",
    ],
    target_normalizer=GroupNormalizer(groups=["fakegroup"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(
    training, df_train, predict=True, stop_randomization=True
)

# Create dataloaders for model
batch_size = 32  # set this between 32 to 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)

# %%
# Modèle baseline pour comparaison, la valeur à prédire sera juste la valeur de la veille
if not FULL_TRAIN:
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    MAE()(baseline_predictions.output, baseline_predictions.y)

# %%
pl.seed_everything(42)
# Possbilité d'ajouter un earlystopping, non utilisé ici
# early_stop_callback = EarlyStopping(
#     monitor="val_loss", min_delta=1e-4, patience=20, verbose=False, mode="min"
# )
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    # limit_train_batches=50,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    # callbacks=[lr_logger, early_stop_callback],
    callbacks=[lr_logger],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.008,  # important
    hidden_size=8,  # attention head number, important , 4-16 defaut
    attention_head_size=3,  # 1-3 defaut
    dropout=0.1,
    hidden_continuous_size=8,  # <= hidden size
    loss=QuantileLoss(),
    log_interval=10,
    optimizer="Adam",
)
print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

# %%
# On entraîne le modèle
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
# %%
# On charge le meilleur modèle
# Pas de validation loss si FULL_TRAIN, donc vérifier que c'est bien le modèle de la dernière epoch
best_model_path = trainer.checkpoint_callback.best_model_path
print("BEST MODEL PATH:", best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
# %%
# MAE Loss sur le set de validation
if not FULL_TRAIN:
    predictions = best_tft.predict(
        val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
    )
    print(MAE()(predictions.output, predictions.y))

# Prédictions sur le test de validation
if not FULL_TRAIN:
    raw_predictions = best_tft.predict(
        val_dataloader,
        mode="raw",
        return_x=True,
        trainer_kwargs=dict(accelerator="cpu"),
    )
# %%
# Le modèle prédit les 2 jours suivant, on clone la dernière row du dataset test pour prédire la dernière valeur sans avoir de problèmes
df_full.fillna(0, inplace=True)
last_idx = df_train["time_idx"].max()
df_full.loc[df_full.index.max() + 1] = df_full.loc[df_full.index.max()]
df_full.loc[df_full.index.max(), "time_idx"] += 1
# %%
# Output des prédictions du test set
if FULL_TRAIN:
    preds = pd.DataFrame(columns=[f"q{i}" for i in range(7)])
    adj = 2
    for i in range(395):
        encoder_data = df_full[
            (df_full["time_idx"] >= last_idx + i - adj - max_encoder_length + 1)
            & (df_full["time_idx"] <= last_idx + i - adj)
        ]
        decoder_data = df_full[
            (df_full["time_idx"] >= last_idx - 1 + i + max_prediction_length - adj)
            & (df_full["time_idx"] <= last_idx + i + max_prediction_length - adj)
        ]
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

        new_raw_predictions = best_tft.predict(
            new_prediction_data,
            mode="raw",
            return_x=True,
            trainer_kwargs=dict(accelerator="cpu"),
        )
        for j in range(7):
            preds.loc[i, f"q{j}"] = float(new_raw_predictions[0][0][0][0][j])

# %%
# Output des prédictions du train set
# Dû à l'apprentissage par une séquence, pas de prédictions pour les `max_encoder_length` premières valeurs du train set
PRED_TRAIN = True
if max_encoder_length == 10:
    horizon = 3461
elif max_encoder_length == 366:
    horizon = 3105
else:
    horizon = 3461
    print("Attention risque d'erreur sur l'horizon des prédictions!")
if PRED_TRAIN and FULL_TRAIN:
    preds_train = pd.DataFrame(columns=[f"q{i}" for i in range(7)])
    adj = 2
    for i in range(horizon):
        encoder_data = df_full[
            (df_full["time_idx"] >= i)
            & (df_full["time_idx"] <= max_encoder_length + i - 1)
        ]
        decoder_data = df_full[
            (df_full["time_idx"] > max_encoder_length + i + max_prediction_length - 3)
            & (
                df_full["time_idx"]
                <= max_encoder_length + i + max_prediction_length - 1
            )
        ]
        new_prediction_data_train = pd.concat(
            [encoder_data, decoder_data], ignore_index=True
        )

        new_raw_predictions_train = best_tft.predict(
            new_prediction_data_train,
            mode="raw",
            return_x=True,
            trainer_kwargs=dict(accelerator="cpu"),
        )
        for j in range(7):
            preds_train.loc[i, f"q{j}"] = float(
                new_raw_predictions_train[0][0][0][0][j]
            )
# %%
shifted_date = pd.to_datetime("2013-03-02") + pd.DateOffset(days=max_encoder_length)
preds_train.index = pd.date_range(
    start=shifted_date, periods=len(preds_train), freq="D"
)
# %%
preds_train.to_csv("Data/preds_tft_train_new.csv")
# %%
# Plot des prédiction
preds.index = pd.date_range(start="2022-09-02", periods=len(preds), freq="D")

for i in preds.columns.to_list():
    plt.plot(preds[i], linewidth=1, alpha=1)
plt.show()

# %%
preds.to_csv("Data/preds_tft_new.csv")
