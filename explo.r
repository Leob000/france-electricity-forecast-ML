rm(list = objects())
graphics.off()
source("R/score.R")
library(tidyverse)

set.seed(42)

df_train_val <- read_csv("Data/train.csv")
df_test <- read_csv("Data/test.csv")

# Matrice de corrélation des variables
library(corrplot)
correlation_matrix <- cor(df_train_val %>% select(-Date, -Year, -Month, -Holiday, -Holiday_zone_a, -Holiday_zone_b, -Holiday_zone_c, -toy, -BH_before, -BH, -BH_after, -WeekDays, -DLS, -Summer_break, -Christmas_break, -BH_Holiday) %>% na.omit())
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# On veut joindre les deux datasets pour faire les transformations de données directement sur les deux
# On drop "Load","Solar_power","Wind_power" du dataset train, qui ne sont pas présentes dans le dataset test
df_train_val <- df_train_val %>% select(-Load, -Solar_power, -Wind_power)
# On drop "Id" et "Usage" du test set
df_test <- df_test %>% select(-Id, -Usage)

# Concaténation des deux jeux de données
df_full <- bind_rows(df_train_val, df_test)
# Check NaNs in df_full
na_counts <- sapply(df_full, function(x) sum(is.na(x)))
print(na_counts)

# Problème avec Nebulosity, non stabilisé pré 2018
plot(df_full$Date, df_full$Nebulosity)
names(df_full)
# Stabilisation de Nebulosity
years <- 2013:2017
seuil_bas <- 58
seuil_haut <- 96
for (year in years) {
  idx <- which(df_full$Year == year)
  col <- df_full$Nebulosity[idx]
  df_full$Nebulosity[idx] <- seuil_bas + (col - min(col)) / (max(col) - min(col)) * (seuil_haut - seuil_bas)
}
plot(df_full$Date, df_full$Nebulosity)

# Idem pour Nebulosity_weighted
plot(df_full$Nebulosity_weighted)
seuil_bas <- 65
seuil_haut <- 96
for (year in years) {
  idx <- which(df_full$Year == year)
  col <- df_full$Nebulosity_weighted[idx]
  df_full$Nebulosity_weighted[idx] <- seuil_bas + (col - min(col)) / (max(col) - min(col)) * (seuil_haut - seuil_bas)
}
plot(df_full$Date, df_full$Nebulosity_weighted)

# On transforme Nebulosity_weighted et Wind_weighted en ratio avec les non weighted
df_full$Nebulosity_weighted <- df_full$Nebulosity_weighted / df_full$Nebulosity
df_full <- df_full %>% rename(Nebulosity_weighted_ratio = Nebulosity_weighted)
df_full$Wind_weighted <- df_full$Wind_weighted / df_full$Wind
df_full <- df_full %>% rename(Wind_weighted_ratio = Wind_weighted)

# Ajout de la feature "Net_demand.1_trend" qui prend la moyenne des 7 dernières valeurs
df_full <- df_full %>%
  arrange(Date) %>%
  mutate(Net_demand.1_trend = zoo::rollapply(Net_demand.1, width = 7, FUN = mean, align = "right", fill = NA))
# On rempli les NaN du début
for (i in 1:6) {
  df_full$Net_demand.1_trend[i] <- df_full$Net_demand.1_trend[7]
}
plot(df_full$Date, df_full$Net_demand)
lines(df_full$Date, df_full$Net_demand.1_trend, col = "red")

# Feature engineering sur les dates
# Weekdays en facteur, idem pour BH_Holiday
df_full$WeekDays <- as.factor(df_full$WeekDays) # 0 = Lundi, 6 = Dimanche
boxplot(Net_demand ~ WeekDays, data = df_full)
df_full$BH_Holiday <- as.factor(df_full$BH_Holiday)
# dayofyear trigonométrique
df_full$dayofyear <- yday(df_full$Date)
df_full$angle <- 2 * pi * (df_full$dayofyear - 1) / 366
df_full$x_dayofyear <- cos(df_full$angle)
df_full$y_dayofyear <- sin(df_full$angle)
df_full <- df_full %>% select(-dayofyear, -angle)
# dayofweek trigonométrique
df_full$dayofweek <- wday(df_full$Date)
df_full$angle_week <- 2 * pi * (df_full$dayofweek - 1) / 7
df_full$x_dayofweek <- cos(df_full$angle_week)
df_full$y_dayofweek <- sin(df_full$angle_week)
df_full <- df_full %>% select(-dayofweek, -angle_week)
# Drop des anciennes variables de date
df_full <- df_full %>% select(-toy, -Month)
# Variable catégorielle pour lundi ou vendredi
df_full$lundi_vendredi <- as.numeric((df_full$WeekDays == 0) | (df_full$WeekDays == 4))
# Variable catégorielle pour température inférieure à 15 degrés
# 15 degrés celcius = 288.15 K
df_full$flag_temp <- as.numeric(df_full$Temp <= 288.15)


# Transformation minmax de l'année
df_full$Year <- (df_full$Year - min(df_full$Year)) / (max(df_full$Year) - min(df_full$Year))

# Création juste des données train
idx_train <- which(df_full$Date <= "2022-09-01")
df_train <- df_full[idx_train, ]

# Normalisation des variables classiques
# On normalisera toujours en prenant la moyenne/écart type à partir des données train, puis en les appliquant sur le jeu de donné global
features_to_normalize <- c(
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
  "Net_demand.1_trend"
)
for (feature in features_to_normalize) {
  f_mean <- mean(df_train[[feature]])
  f_sd <- sd(df_train[[feature]])
  df_full[[feature]] <- (df_full[[feature]] - f_mean) / f_sd
}

# Normalisation des variables qui ont un lag, on normalisera les différentes Net_demand plus tard pour garder en mémoire la moyenne et l'écart type, pour déstandardiser les prédictions
features_to_normalize_multiple <- list(
  c("Load.1", "Load.7"),
  c("Solar_power.1", "Solar_power.7"),
  c("Wind_power.1", "Wind_power.7")
)
for (feature_pair in features_to_normalize_multiple) {
  first_feature <- feature_pair[1]
  second_feature <- feature_pair[2]
  f_mean <- mean(df_train[[first_feature]])
  f_sd <- sd(df_train[[first_feature]])
  df_full[[first_feature]] <- (df_full[[first_feature]] - f_mean) / f_sd
  df_full[[second_feature]] <- (df_full[[second_feature]] - f_mean) / f_sd
}

# On sauvegarde les résultats en csv pour les réutiliser ailleurs
write_csv(df_full, "Data/treated_data.csv")
