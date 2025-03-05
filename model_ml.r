rm(list = objects())
graphics.off()
source("R/score.R")
library(tidyverse)

set.seed(42)

df_full <- read_csv("Data/treated_data.csv")

# On doit remettre en facteur à cause du csv
df_full$WeekDays <- as.factor(df_full$WeekDays) # 0 = Lundi, 6 = Dimanche
df_full$BH_Holiday <- as.factor(df_full$BH_Holiday)

# Normalisation de Net_demand et laggées, en conservant la moyenne et l'écart type pour faire la transformation inverse sur les prédictions
idx_train_val <- which(df_full$Date <= "2022-09-01")
df_train_val <- df_full[idx_train_val, ]
Net_demand_mean <- mean(df_train_val$Net_demand)
Net_demand_sd <- sd(df_train_val$Net_demand)

df_full$Net_demand <- (df_full$Net_demand - Net_demand_mean) / Net_demand_sd
df_full$Net_demand.1 <- (df_full$Net_demand.1 - Net_demand_mean) / Net_demand_sd
df_full$Net_demand.7 <- (df_full$Net_demand.7 - Net_demand_mean) / Net_demand_sd

# On fera une validation sur la dernière année du jeu train
# Création des jeux de données train, val, test; et un jeu qui combine train et val
idx_train <- which(df_full$Date <= "2021-09-01")
idx_val <- which((df_full$Date > "2021-09-01") & (df_full$Date <= "2022-09-01"))
idx_test <- which(df_full$Date > "2022-09-01")
df_train <- df_full[idx_train, ]
df_val <- df_full[idx_val, ]
df_test <- df_full[idx_test, ]

### Models
target_col <- "Net_demand"
features_col <- setdiff(names(df_full), target_col)
features_col <- setdiff(features_col, "Date")
features_col

# Train a basic random forest model
library(randomForest)
rf_model <- randomForest(as.formula(paste(target_col, "~ .")), data = df_train[, c(features_col, target_col)], importance = TRUE)

# Print the model summary
print(rf_model)

# Predict on the validation set
predictions <- (predict(rf_model, df_val[, features_col]) * Net_demand_sd) + Net_demand_mean
ground_truth <- (df_val$Net_demand * Net_demand_sd) + Net_demand_mean

# Calculate and print the RMSE
rmse <- sqrt(mean((ground_truth - predictions)^2))
print(paste("RMSE on validation set:", rmse))

err_pinball <- pinball_loss(ground_truth, predictions, 0.8)
print(paste("Pinball on validation set:", err_pinball))

# Get the importance of variables
importance <- importance(rf_model)
print(importance)

# Plot the importance of variables
varImpPlot(rf_model)
# Plot predictions and ground truth as lines
plot(df_val$Date, ground_truth, type = "l", col = "blue", lwd = 2, ylab = "Net Demand", xlab = "Date", main = "Predictions vs Ground Truth")
lines(df_val$Date, predictions, col = "red", lwd = 2)
legend("topright", legend = c("Ground Truth", "Predictions"), col = c("blue", "red"), lwd = 2)
