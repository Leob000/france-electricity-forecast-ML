# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_pinball_loss
import numpy as np

# %%
df_rf = pd.read_csv("Data/preds_rf_test.csv")
df_lstm = pd.read_csv("Data/preds_lstm_test.csv")
df_truth = pd.read_csv("Data/test_better.csv")
# %%
preds_rf = df_rf["predictions_test"]
preds_lstm = df_lstm["preds"]
truth = df_truth["Net_demand"]
date = pd.to_datetime(df_lstm["Date"])
# plt.plot(date, truth)
# plt.plot(date, preds_rf)
# plt.plot(date, preds_lstm)
# plt.show()
print("RF:", mean_pinball_loss(truth, preds_rf, alpha=0.8))
print("LSTM:", mean_pinball_loss(truth, preds_lstm, alpha=0.8))

rf_weight = 0.75

for rf_weight in np.arange(0.05, 1.0, 0.05):
    preds_mean = rf_weight * preds_rf + (1 - rf_weight) * preds_lstm
    print(
        "Mean:",
        mean_pinball_loss(truth, preds_mean, alpha=0.8),
        "RF weight:",
        rf_weight.round(2),
    )

# %%
preds = 0.5 * preds_rf + 0.5 * preds_lstm
df_preds = pd.DataFrame({"Net_demand": preds})
df_preds["Id"] = range(1, len(df_preds) + 1)
df_preds = df_preds[["Id", "Net_demand"]]
df_preds.to_csv("Data/predictions_.csv", index=False)
# %%
df_lul = pd.read_csv("Data/submission_mod.csv")
mean_pinball_loss(truth, df_lul["Net_demand"], alpha=0.8)
