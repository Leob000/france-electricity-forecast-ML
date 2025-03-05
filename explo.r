rm(list = objects())
graphics.off()
source("R/score.R")
library(tidyverse)

set.seed(42)
GRAPH <- TRUE

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

# On transforme Nebulosity_weighted et Wind_weighted en ratio avec les non weighted
df_train_val$Nebulosity_weighted <- df_train_val$Nebulosity_weighted / df_train_val$Nebulosity
df_train_val <- df_train_val %>% rename(Nebulosity_weighted_ratio = Nebulosity_weighted)
df_train_val$Wind_weighted <- df_train_val$Wind_weighted / df_train_val$Wind
df_train_val <- df_train_val %>% rename(Wind_weighted_ratio = Wind_weighted)

# On drop "Load","Solar_power","Wind_power" du dataset train, qui ne sont pas présentes dans le dataset test
df_train_val <- df_train_val %>% select(-Load, -Solar_power, -Wind_power)

# Matrice de corrélation des variables
library(corrplot)
correlation_matrix <- cor(df_train_val %>% select(-Date, -Year, -Month, -Holiday, -Holiday_zone_a, -Holiday_zone_b, -Holiday_zone_c, -toy, -BH_before, -BH, -BH_after, -WeekDays, -DLS, -Summer_break, -Christmas_break, -BH_Holiday) %>% na.omit())
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# Ajout de la feature "Net_demand.1_trend" qui prend la moyenne des 7 dernières valeurs
df_train_val <- df_train_val %>%
  arrange(Date) %>%
  mutate(Net_demand.1_trend = zoo::rollapply(Net_demand.1, width = 7, FUN = mean, align = "right", fill = NA))

# On rempli les NaN du début
for (i in 1:6) {
  df_train_val$Net_demand.1_trend[i] <- df_train_val$Net_demand.1_trend[7]
}
plot(df_train_val$Date, df_train_val$Net_demand)
lines(df_train_val$Date, df_train_val$Net_demand.1_trend, col = "red")

# Création du set de validation, on prend la dernière année dans les données d'entraînement
range(df_train_val$Date)
train_idx <- which(df_train_val$Date <= "2021-09-01")
df_train <- df_train_val[train_idx, ]
df_val <- df_train_val[-train_idx, ]
range(df_train$Date)
range(df_val$Date)
