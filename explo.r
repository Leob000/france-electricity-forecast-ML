rm(list = objects())
graphics.off()
source("R/score.R")
library(tidyverse)

set.seed(42)
GRAPH <- FALSE

df_train_val <- read_csv("Data/train.csv")
df_test <- read_csv("Data/test.csv")

# Problème avec Nebulosity, non stabilisé pré 2018
plot(df_train$Nebulosity)
# Stabilisation de Nebulosity
years <- 2013:2017
seuil_bas <- 58
seuil_haut <- 96
for (year in years) {
  idx <- which(df_train$Year == year)
  df_train$Nebulosity_stab[idx] <- seuil_bas + (df_train$Nebulosity[idx] - min(df_train$Nebulosity[idx])) / (max(df_train$Nebulosity[idx]) - min(df_train$Nebulosity[idx])) * (seuil_haut - seuil_bas)
}
idx <- which(df_train$Year >= 2018)
df_train$Nebulosity_stab[idx] <- df_train$Nebulosity[idx]
plot(df_train$Date, df_train$Nebulosity_stab)


# Création du set de validation, on prend l'année 2021
range(df_train_val$Date)
train_idx <- which(df_train_val$Date <= "2021-09-01")
df_train <- df_train_val[train_idx, ]
df_val <- df_train_val[-train_idx, ]
range(df_train$Date)
range(df_val$Date)
