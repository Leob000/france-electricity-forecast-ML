rm(list = objects())
graphics.off()
source("R/score.R")
library(tidyverse)

set.seed(42)
GRAPH <- FALSE

df_train_val <- read_csv("Data/train.csv")
df_test <- read_csv("Data/test.csv")

# Problème avec Nebulosity, non stabilisé pré 2018
plot(df_train_val$Nebulosity)
names(df_train_val)
# Stabilisation de Nebulosity
years <- 2013:2017
seuil_bas <- 58
seuil_haut <- 96
for (year in years) {
  idx <- which(df_train_val$Year == year)
  col <- df_train_val$Nebulosity[idx]
  df_train_val$Nebulosity[idx] <- seuil_bas + (col - min(col)) / (max(col) - min(col)) * (seuil_haut - seuil_bas)
}
plot(df_train_val$Date, df_train_val$Nebulosity)

# Idem pour Nebulosity_weighted
plot(df_train_val$Nebulosity_weighted)
seuil_bas <- 65
seuil_haut <- 96
for (year in years) {
  idx <- which(df_train_val$Year == year)
  col <- df_train_val$Nebulosity_weighted[idx]
  df_train_val$Nebulosity_weighted[idx] <- seuil_bas + (col - min(col)) / (max(col) - min(col)) * (seuil_haut - seuil_bas)
}
plot(df_train_val$Date, df_train_val$Nebulosity_weighted)

# Matrice de corrélation des variables
# TODO Clarifier la matrice?
library(corrplot)
correlation_matrix <- cor(df_train_val %>% select(-Date, -Year, -Month, -Holiday, -Holiday_zone_a, -Holiday_zone_b, -Holiday_zone_c, -toy, -BH_before, -BH, -BH_after, -WeekDays, -DLS, -Summer_break, -Christmas_break, -BH_Holiday) %>% na.omit())
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# Création du set de validation, on prend l'année 2021
range(df_train_val$Date)
train_idx <- which(df_train_val$Date <= "2021-09-01")
df_train <- df_train_val[train_idx, ]
df_val <- df_train_val[-train_idx, ]
range(df_train$Date)
range(df_val$Date)
