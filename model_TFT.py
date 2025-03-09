# %%
import warnings


warnings.filterwarnings("ignore")  # avoid printing out absolute paths

from sklearn.metrics import mean_pinball_loss
import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
import os
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

FULL_TRAIN = True  # Entrainement complet, False pour validation
plt.rcParams["figure.figsize"] = (10, 6)

# %%
df_full = pd.read_csv("Data/treated_data_tft.csv")
df_full["time_idx"] = np.arange(len(df_full))
df_full["Date"] = pd.to_datetime(df_full["Date"])
df_full["fakegroup"] = "a"
# %%
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
# Normalisation des net_demand
f_mean = df_full.loc[df_full["Date"] <= "2022-09-01", "Net_demand"].mean()
f_std = df_full.loc[df_full["Date"] <= "2022-09-01", "Net_demand"].std()

# TODO implémenter norm pour Net_demand?
for f in ["Net_demand_1", "Net_demand_7"]:
    df_full[f] = (df_full[f] - f_mean) / f_std
# %%
df_full.rename(columns={"Date": "date"}, inplace=True)
df_full.fillna(40000, inplace=True)
df_full = pd.get_dummies(data=df_full, columns=["WeekDays", "BH_Holiday"], dtype=int)
# %%
# FULL train utilise tout le train set pour apprendre (les 2 jours du test set ne sont pas pris en compte par le modèle, la librairie requiert juste de mettre un minimum de validation set donc on prend ici l'horizon de prediction = 2 jours en plus)
if FULL_TRAIN:
    df_test = df_full[df_full["date"] >= "2022-09-04"]
    df_train = df_full[df_full["date"] <= "2022-09-03"]
else:  # Validation sur les 2 derniers jours du train set
    df_test = df_full[df_full["date"] >= "2022-09-02"]
    df_train = df_full[df_full["date"] <= "2022-09-01"]

# TODO Pb dernier jour du val set non train

# %%
max_prediction_length = 2  # TODO (len de val set)
max_encoder_length = 10  # TODO (len de la sequence)
training_cutoff = df_train["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df_train[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Net_demand",
    # group_ids=["agency", "sku"],
    group_ids=["fakegroup"],
    min_encoder_length=max_encoder_length
    // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    # static_categoricals=["agency", "sku"],
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
    target_normalizer=GroupNormalizer(
        groups=["fakegroup"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(
    training, df_train, predict=True, stop_randomization=True
)

# create dataloaders for model
batch_size = 32  # set this between 32 to 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)

# %%
# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
if not FULL_TRAIN:
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    MAE()(baseline_predictions.output, baseline_predictions.y)

# %%
# configure network and trainer
pl.seed_everything(42)
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
    attention_head_size=3,  # 1-2 defaut
    dropout=0.1,
    hidden_continuous_size=8,  # <= hidden size
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    # optimizer="ranger",
    optimizer="Adam",
    # reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

# %%
# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
# %%
# TODO Check hyperparam tuning https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html#Hyperparameter-tuning

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
print("BEST MODEL PATH:", best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
# %%
# calcualte mean absolute error on validation set
if not FULL_TRAIN:
    predictions = best_tft.predict(
        val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
    )
    print(MAE()(predictions.output, predictions.y))
# %%
# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions = best_tft.predict(
    val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu")
)
# %%
# for idx in range(1):  # plot 10 examples
#     best_tft.plot_prediction(
#         raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
#     )
# %%
# calcualte metric by which to display
# predictions = best_tft.predict(
#     val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
# )
# mean_losses = SMAPE(reduction="none").loss(predictions.output, predictions.y[0]).mean(1)
# indices = mean_losses.argsort(descending=True)  # sort losses
# for idx in range(1):  # plot 10 examples
#     best_tft.plot_prediction(
#         raw_predictions.x,
#         raw_predictions.output,
#         idx=indices[idx],
#         add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles),
#     )
# %%
# predictions = best_tft.predict(
#     val_dataloader, return_x=True, trainer_kwargs=dict(accelerator="cpu")
# )
# predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(
#     predictions.x, predictions.output
# )
# best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

# %%
df_full.fillna(0, inplace=True)
last_idx = df_train["time_idx"].max()
df_full.loc[df_full.index.max() + 1] = df_full.loc[df_full.index.max()]
df_full.loc[df_full.index.max(), "time_idx"] += 1
# %%
# Attention, break si change jours d'encoder ou preds
# Dans ce cas vérif encoder_data et decoder_data correspondent bien
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
# Pred sur train data
PRED_TRAIN = True
if PRED_TRAIN:
    preds_train = pd.DataFrame(columns=[f"q{i}" for i in range(7)])
    adj = 2
    for i in range(3461):  # 3461
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
    preds_train.index = pd.date_range(
        start="2013-03-02", periods=len(preds_train), freq="D"
    )
# %%
# preds_train.to_csv("Data/preds_tft_train.csv")
# %%
# Interpret model
interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
best_tft.plot_interpretation(interpretation)
# %%
# TODO enlever truth
truth = pd.read_csv("Data/test_better.csv")
# %%
preds.index = pd.date_range(start="2022-09-02", periods=len(preds), freq="D")

# plt.plot(preds.index, truth["Net_demand"])
for i in preds.columns.to_list():
    err = mean_pinball_loss(truth["Net_demand"], preds[i], alpha=0.8)
    print(i, "Pinball=", err)
    plt.plot(preds[i], linewidth=1, alpha=1)
plt.show()

# %%
# preds.to_csv("Data/preds_tft.csv")

# %%
# Play a sound when training is complete
os.system('say "Training complete"')
